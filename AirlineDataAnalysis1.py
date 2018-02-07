## Spark Application - execute with spark-submit
## Airline Data Analysis with visualisation

import calendar
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession
from pyspark.sql import SQLContext
from pyspark.sql.types import StringType
from pyspark.sql import functions as F

conf = SparkConf()
sc = SparkContext(conf = conf)
sqlContext = SQLContext(sc)
spark = SparkSession(sc)

## This function is used for changing the datatypes of few columns in the input dataframe
def convertColumn(df, name, new_type):
    df_1 = df.withColumnRenamed(name, "swap")
    return df_1.withColumn(name, df_1.swap.cast(new_type)).drop("swap") 

## Location of data: contains 'airline on time' performance data from dataexpo 2009
air_file_loc = "/usr/local/airlinedata/inputdata/*.csv"
airports_loc = "/usr/local/airlinedata/airports.csv"
carriers_loc = "/usr/local/airlinedata/carriers.csv"

## Read in Airline DataFrame from csv
df = sqlContext.read.load(air_file_loc, 
                          format='com.databricks.spark.csv', 
                          header='true', 
                          inferSchema='true')

## Read in Carriers DataFrame from csv                          
df_carrName = sqlContext.read.load(carriers_loc, 
                          format='com.databricks.spark.csv', 
                          header='true', 
                          inferSchema='true')

## Read in Airports DataFrame from csv                          
df_Airports = sqlContext.read.load(airports_loc, 
                          format='com.databricks.spark.csv', 
                          header='true', 
                          inferSchema='true')                          
                          
## Casting the fields from string type to int type. The last dataframe will have all the fields casted to int
# observe the pattern   
df_2 = convertColumn(df, "ArrDelay", "int")    
df_3 = convertColumn(df_2, "DepDelay", "int")
df_4 = convertColumn(df_3, "CarrierDelay", "int")
df_5 = convertColumn(df_4, "WeatherDelay", "int")
df_6 = convertColumn(df_5, "NASDelay", "int")
df_7 = convertColumn(df_6, "SecurityDelay", "int")
df_8 = convertColumn(df_7, "Month", "int")
airlineinfo_df = convertColumn(df_8, "LateAircraftDelay", "int")

## Fill the null/NA values in the int columns
airlineinfo_df.fillna(0)

## Register the Dataframes as tables
sqlContext.registerDataFrameAsTable(airlineinfo_df,"AirlineTable")
sqlContext.registerDataFrameAsTable(df_carrName, "carrierName")
sqlContext.registerDataFrameAsTable(df_Airports, "AirportName")

## new dataframe with carrier name column mapped
df_mappedCarrierNames = sqlContext.sql \
                        ("select t.Year,t.Month,t.DayOfWeek,t.UniqueCarrier,t.Origin,t.Dest,t.ArrDelay,t.DepDelay,c.Description as AirlineName \
                          from carrierName c, AirlineTable t \
                          where t.UniqueCarrier = c.Code")

## Filtering only positive ArrDelay and Postive dep delay in different dataframes                      
df_ArrDelay = df_mappedCarrierNames.filter(df_mappedCarrierNames.ArrDelay > 0) 
df_DepDelay = df_mappedCarrierNames.filter(df_mappedCarrierNames.DepDelay > 0) 
                       
## User defined Function is used for mapping month numbers to respective Month Strings
## calendar.month_abbr[int(x)] is from calendar API of Python
udf = F.UserDefinedFunction(lambda x: calendar.month_abbr[int(x)], StringType())
df_MonthlyArrDelay = df_ArrDelay.withColumn('MonthName', udf("Month"))
df_ArrDelay_MonthName = df_MonthlyArrDelay.sort(F.asc("Month")) 
df_MonthlyDepDelay = df_DepDelay.withColumn('MonthName', udf("Month"))
df_DepDelay_MonthName = df_MonthlyDepDelay.sort(F.asc("Month")) 

## User defined Function is used for mapping weekday numbers to respective weekday Strings
## calendar.day_name[int(x)] is from calendar API of Python
udf = F.UserDefinedFunction(lambda x: calendar.day_name[int(x) - 1], StringType())
df_WeeklyArrDelay = df_ArrDelay_MonthName.withColumn('WeekDay', udf("DayOfWeek"))
df_ArrDelay_WeekDay = df_WeeklyArrDelay.sort(F.asc("DayOfWeek")) 

# Register the dataframe into a table
sqlContext.registerDataFrameAsTable(df_DepDelay_MonthName,"DepDelayAnalysis")
sqlContext.registerDataFrameAsTable(df_ArrDelay_WeekDay,"ArrDelayAnalysis")

## 1... ##############################################################################################
## ArrDelay Analysis for SFO Airport for 12 months of a year 

df_Delay_SFO = df_ArrDelay.where(df_ArrDelay.Dest == 'SFO') 

df_SFOMonthArrDelay = sqlContext.sql \
                      ("select sfo.Month,sfo.MonthName,avg(sfo.ArrDelay) as TotalArrivalDelay \
                      from ArrDelayAnalysis sfo \
                      where (sfo.Dest = 'SFO') \
                      group by sfo.Month, sfo.MonthName")
                      
df_SFOMonthArrDelay_result =  df_SFOMonthArrDelay.toPandas()

## Departure Delay Analysis for SFO Airport for 12 months of a year 
df_Delay_SFO = df_DepDelay.where(df_DepDelay.Dest == 'SFO') 
df_SFOMonthDepDelay = sqlContext.sql \
                      ("select sfo.Month,sfo.MonthName,avg(sfo.DepDelay) as AvgDepDelay \
                      from DepDelayAnalysis sfo \
                      where (sfo.Dest = 'SFO') \
                      group by sfo.Month, sfo.MonthName")
                      
df_SFOMonthDepDelay_result =  df_SFOMonthDepDelay.toPandas()


## 2.....##############################################################################################
## ArrDelay Analysis for Chicago Airport for 12 months of a year 
df_ChicagoMonthDepDelay = sqlContext.sql \
                      ("select chicago.Month,chicago.MonthName, avg(chicago.DepDelay) as AvgDepDelay \
                      from DepDelayAnalysis chicago \
                      where (chicago.Dest = 'ORD') \
                      group by chicago.Month, chicago.MonthName")
                      
df_ChicagoMonthDepDelay_result =  df_ChicagoMonthDepDelay.toPandas()

# Register the dataframe into a table

## 3.....##############################################################################################
## ArrDelay Analysis for SFO Airport for 7 days of a week 

df_SFOWeeklyArrDelay = sqlContext.sql \
                      ("select sfo.DayOfWeek,sfo.WeekDay,avg(sfo.ArrDelay) as AvgArrivalDelay \
                      from ArrDelayAnalysis sfo \
                      where (sfo.Dest = 'SFO') \
                      group by sfo.DayOfWeek, sfo.WeekDay")
                      
df_SFOWeeklyArrDelay_result =  df_SFOWeeklyArrDelay.toPandas()


## 4.....##############################################################################################
## Average Arrival Delay analysis by unique carrier accross all airports
## Convert spark dataframe to pandas dataframe inorder to plot a bargraph with seaborn
df_CarrierWiseArrDelay_result = df_ArrDelay.groupBy("UniqueCarrier",df_ArrDelay.AirlineName) \
                                            .agg(F.avg(df_ArrDelay.ArrDelay) \
                                            .alias("AverageArrivalDelay")) \
                                            .sort(F.asc("UniqueCarrier")) \
                                            .toPandas()
print(df_CarrierWiseArrDelay_result)
## 5..... ##############################################################################################
## Finding best carrier with minimal arrival Delay in SFO Airport

df_CarrierWiseSFOArrDelay_result = df_Delay_SFO.groupBy("UniqueCarrier",df_ArrDelay.AirlineName) \
                                            .agg(F.avg(df_ArrDelay.ArrDelay) \
                                            .alias("AverageArrivalDelay")) \
                                            .sort(F.asc("AverageArrivalDelay")) \
                                            .toPandas()
print(df_CarrierWiseSFOArrDelay_result)

## 6... ##############################################################################################

df_mappedAirportNames = sqlContext.sql \
    ("select f.AirlineName, a.city as OriginCity, b.city as DestCity, f.ArrDelay \
    from ArrDelayAnalysis f, AirportName a,AirportName b \
    where f.Origin = a.iata and f.Dest = b.iata")

df_result = df_mappedAirportNames \
            .groupBy("AirlineName",df_mappedAirportNames.OriginCity,df_mappedAirportNames.DestCity) \
            .agg(F.avg("ArrDelay").alias("AverageArrivalDelay")) \
            .sort(F.asc("AverageArrivalDelay"))
df_result.show(5,truncate=7)

## 7... ##############################################################################################
## Finding the Month with minimal Arrival and Departure Delay in East Coast 

# Selecting major airports in Eastcoast for delay analysis
df_EastCoastDealy_Arr = df_ArrDelay_MonthName.where(F.col("Dest").isin ({"IAD","DCA","ATL","BWI","BOS","JFK"}))             
df_EastCoastDelay_Dep = df_DepDelay_MonthName.where(F.col("Dest").isin ({"IAD","DCA","ATL","BWI","BOS","JFK"}))             
df_MonthlyDepAvg = df_EastCoastDelay_Dep.groupBy("Month","MonthName") \
                                            .agg(F.avg(df_EastCoastDelay_Dep.DepDelay) \
                                            .alias("AverageDepDelay")) \
                                            .sort(F.asc("Month"))
                                            
#rdd_MonthlyDepAvg = df_MonthlyDepAvg.rdd
#df_MonthlyDepAvg.write.format("org.elasticsearch.spark.sql").option("es.resource", "airline/ecdep").option("es.nodes", "10.0.0.7").save()

df_MonthlyDepDelay_result = df_MonthlyDepAvg.toPandas()
                                  
df_MonthlyArrDelay_result = df_EastCoastDealy_Arr.groupBy("Month","MonthName") \
                                            .agg(F.avg(df_EastCoastDealy_Arr.ArrDelay) \
                                            .alias("AverageArrDelay")) \
                                            .sort(F.asc("Month")) \
                                            .toPandas()                                            


## Seaborn is a Python visualization library based on matplotlib. It provides a high-level interface for drawing attractive statistical graphics.
fig = plt.figure()
fig.set_size_inches(18,10)
ax1 = fig.add_subplot(321)
ax2 = fig.add_subplot(322)
ax3 = fig.add_subplot(323)
ax4 = fig.add_subplot(324)
ax5 = fig.add_subplot(325)
ax6 = fig.add_subplot(326)
fig.subplots_adjust(hspace=.5)

## Finding the Month with minimal Arrival Delay in East Coast 
sns_bar = sns.barplot(x="MonthName", y="AverageArrDelay", data=df_MonthlyArrDelay_result, ax=ax1)
sns_bar.set(xlabel='Month', ylabel='Avg ArrivalDelay(mins)')
sns_bar.set_title('1. Arrival Delay EastCoast')

## Finding the Month with minimal Departure Delay in East Coast 
sns_bar = sns.barplot(x="MonthName", y="AverageDepDelay", data=df_MonthlyDepDelay_result, ax=ax2)
sns_bar.set(xlabel='Month', ylabel='Avg DepartureDelay(mins)')
sns_bar.set_title('2. Departure Delay EastCoast')

## Finding the best carrier with minimal arrival Delay in SFO Airport
sns_bar = sns.barplot(x="UniqueCarrier", y="AverageArrivalDelay", data=df_CarrierWiseSFOArrDelay_result, ax=ax3)
sns_bar.set(xlabel='AirlineCode', ylabel='Avg ArrivalDelay(mins)')
sns_bar.set_title('3. SFO - Arrival Delay')

## Average Arrival Delay analysis by unique carrier accross all airports
sns_bar = sns.barplot(x="UniqueCarrier", y="AverageArrivalDelay", data=df_CarrierWiseArrDelay_result, ax=ax4)
sns_bar.set(xlabel='AirlineCode', ylabel='Avg ArrivalDelay(mins)')
sns_bar.set_title('4. Arrival Delays of carriers')

## ArrDelay Analysis for SFO Airport for 7 days of a week                  
sns_bar = sns.barplot(x="WeekDay", y="AvgArrivalDelay", data=df_SFOWeeklyArrDelay_result, ax=ax5)
sns_bar.set(xlabel='Day of the week', ylabel='Avg ArrivalDelay(mins)')
sns_bar.set_title('5. SFO - Weekly Arrival Delay')

## Plotting ArrDelay graph for chicago Airport for 12 months of a year                   
sns_bar = sns.barplot(x="MonthName", y="AvgDepDelay", data=df_ChicagoMonthDepDelay_result, ax=ax6)
sns_bar.set(xlabel='Month', ylabel='Avg DepDelay (mins)')
sns_bar.set_title('6. Chicago - Monthly Departure Delay')

## Save the output figure with group of subplots in a png file
fig.savefig("/tmp/Output1.png")

################################################################################################



