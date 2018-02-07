# -*- coding: utf-8 -*-
## Spark Application - execute with spark-submit
## Airline Data Analysis with visualisation

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession
from pyspark.sql import SQLContext
from pyspark.sql.functions import *
from graphframes import *


conf = SparkConf()
sc = SparkContext(conf = conf)
sqlContext = SQLContext(sc)
spark = SparkSession(sc)

## This function is used for changing the datatypes of few columns in the input dataframe
def convertColumn(df, name, new_type):
    df_1 = df.withColumnRenamed(name, "swap")
    return df_1.withColumn(name, df_1.swap.cast(new_type)).drop("swap") 

## Location of data: contains 'airline on time' performance data from dataexpo 2009
air_file_loc = "/usr/local/airlinedata/inputdata/2008.csv"
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
df_9 = convertColumn(df_8, "LateAircraftDelay", "int")

## Fill the null/NA values in the int columns
df_9.fillna(0)

airline_df = df_9.join(df_carrName, df_9.UniqueCarrier == df_carrName.Code).join(df_Airports, df_9.Dest ==  df_Airports.iata)

## Filtering only positive ArrDelay and Postive dep delay in different dataframes                      
df_ArrDelay = airline_df.filter(airline_df.ArrDelay > 0) 
df_DepDelay = airline_df.filter(airline_df.DepDelay > 0) 

################################################################################################
#df_mappedCarrierNames
## What flights departing from SFO are most likely to have significant delays?
vertices = df_Airports.select(col("iata"),col("airport"), col("state"), col("city")).withColumnRenamed("iata","id").distinct()
edges =  airline_df.withColumnRenamed("Origin","src").withColumnRenamed("Dest","dst").withColumnRenamed("city","dstCity")

airlineGraph1 = GraphFrame(vertices, edges)

SFO_DepartingFlightsDelay = airlineGraph1.edges.filter("src = 'SFO' and ArrDelay > 0") \
                          .groupBy("src", "dst","dstCity") \
                          .avg("ArrDelay") \
                          .sort(desc("avg(ArrDelay)")).limit(10)
                          
SFO_DepartingFlightsDelay.show(10)
SFO_DepartingFlightsDelay_result = SFO_DepartingFlightsDelay.toPandas()                          

###############################################################################################################################
## BFS algorithm to find the shortest path from Reno to St.Louis

vertices = df_Airports.select(col("iata"), col("city")).withColumnRenamed("iata","id").distinct()
edges =  airline_df.select(col("Origin"), col("Dest"), col("FlightNum"), col("UniqueCarrier")).distinct().withColumnRenamed("Origin","src").withColumnRenamed("Dest","dst").withColumnRenamed("city","dstCity")
airlineGraph2 = GraphFrame(vertices, edges)
paths1 = airlineGraph2.bfs("id = 'ORD'", "id = 'RNO'", maxPathLength=3)
paths1.show(5)
df_BFS = paths1.toPandas()
paths2 = airlineGraph2.bfs("id = 'RNO'", "id = 'STL'", edgeFilter="dst != 'SJC'", maxPathLength=10)
paths2.show(5)      

###############################################################################################################################
## PageRank algorithm to find the most connected Airport
edges =  airline_df.filter(airline_df.Month < 3).withColumnRenamed("Origin","src").withColumnRenamed("Dest","dst").withColumnRenamed("city","dstCity")
airlineGraph1 = GraphFrame(vertices, edges)
ranks = airlineGraph1.pageRank(resetProbability=0.15, maxIter=5)
df_PageRank = ranks.vertices.orderBy(ranks.vertices.pagerank.desc()).limit(20)
df_PageRank.show()
df_PageRank_result = df_PageRank.toPandas()

## Seaborn is a Python visualization library based on matplotlib. It provides a high-level interface for drawing attractive statistical graphics.

fig = plt.figure()
fig.set_size_inches(18,10)

ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)
'''ax3 = fig.add_subplot(223,frame_on=False) # no visible frame
ax3.xaxis.set_visible(False)  # hide the x axis
ax3.yaxis.set_visible(False)  # hide the y axis
'''
#ax4 = fig.add_subplot(224)
fig.subplots_adjust(hspace=.5)

## PageRank algorithm plot to find the Visualize the most connected Airport
sns_bar = sns.barplot(x="id", y="pagerank", data=df_PageRank_result, ax=ax1)
sns_bar.set(xlabel='Airport', ylabel='PageRank')
sns_bar.set_title('Which is the most connected Airport?')

## What flights departing from SFO are most likely to have significant delays?
sns_bar = sns.barplot(x="dst", y="avg(ArrDelay)", data=SFO_DepartingFlightsDelay_result, ax=ax2)
sns_bar.set(xlabel='Destination Airport', ylabel='Average Delay')
sns_bar.set_title('Which flights departing from SFO are most likely to have significant delays?')

## Display dataframe table in seaborn 
#table(ax3, df_BFS)  # where df is your data frame

## Save the output figure with group of subplots in a png file
fig.savefig("/tmp/Output2.png")

