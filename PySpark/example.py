$ sudo apt install default-jre
$ sudo apt install openjdk-11-jre-headless


import os
os.getcwd()
os.environ["SPARK_HOME"]="/home/anaconda3/lib/python3.7/site-packages/pyspark"

import findspark
findspark.init()


import pyspark # only run after findspark.init()
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("PySpark").getOrCreate()
spark

cores = spark._jsc.sc().getExecutorMemoryStatus().keySet().size()
print("You are working with", cores, "core(s)")
spark


data = [['tom', 10], ['nick', 15], ['juli', 14]] 

df = spark.createDataFrame(data,['Name', 'Age']) 

df.show()
############################

path = "/home/anaconda3/work/Python Files and Datasets AS of 22DEC20/PySpark DataFrame Essentials/Datasets/students.csv"
df = spark.read.csv(path,,inferSchema=True,header=True)
df.toPandas()

df.groupBy("gender").agg({'math score':'mean'}).show()

# It's not until we change the df in some way, that the ID changes
# These kinds of commands won't actually be run...
df = df.withColumn('new_col', df['math score'] * 2)

# Until we executute a command like this
collect = df.collect()

# Even if we duplicate the dataframe, the ID remains the same
df2 = df
df2.rdd.id()

# Iterate over a column

def square_float(x):
    return float(x**2)

square_udf_float2=udf(lambda z: square_float(z),FloatType())

(df.select('integers',square_udf_float2('integers').alias('int_squared')))

students = spark.read.csv(path+'students.csv',inferSchema=True,header=True)
print(type(students))

studentsPdf = students.toPandas()
print(type(studentsPdf))

students.schema['math score'].dataType

students.describe(['math score']).show()

students.select("math score", "reading score","writing score").summary("count", "min", "25%", "75%", "max").show()

# **Parquet Files**

parquet = spark.read.parquet(path+'users1.parquet')
parquet.show(2)

# **Partitioned Parquet Files**
# Actually most big datasets will be partitioned. Here is how you can collect all the pieces (parts) of the dataset in one simple command.

partitioned = spark.read.parquet(path+'users*')
partitioned.show(2)

users1_2 = spark.read.option("basePath", path).parquet(path+'users1.parquet', path+'users2.parquet')
users1_2.show()

# However you often have to set the schema yourself if you aren't dealing with a .read method that doesn't have inferSchema() built-in.

from pyspark.sql.types import StructField,StringType,IntegerType,StructType,DateType

data_schema = [StructField("name", StringType(), True),
               StructField("email", StringType(), True),
               StructField("city", StringType(), True),
               StructField("mac", StringType(), True),
               StructField("timestamp", DateType(), True),
               StructField("creditcard", StringType(), True)]

final_struc = StructType(fields=data_schema)

people = spark.read.json(path+'people.json', schema=final_struc)

people.printSchema()
