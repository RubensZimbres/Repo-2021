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

df.select("gender", "math_score").summary("count", "min", "max").show()
df.select(['Name','gender']).orderBy('Name').show(5,False) #not truncated
df.select("Name","math_score").where(df.Name.like("A%")).show(5, False)
df[df.Name.isin("L. Messi", "Cristiano Ronaldo")].limit(4).toPandas()
df.select("Photo",df.Photo.substr(-4,4)).show(5,False) #png
df.select("Name","math_score").where(df.name.startswith("L")).where(df.Name.like("A%")).show(5, False)

df.filter("Age>40").limit(4).toPandas()
## spark starts with 1 != Python = 0 
## filer BEFORE select

# It's not until we change the df in some way, that the ID changes
# These kinds of commands won't actually be run...
df = df.withColumn('new_col', df['math score'] * 2)

df2 = df.withColumnRenamed('Rolling year total number of offences','Count')

df.createOrReplaceTempView("tempview")
spark.sql("SELECT Region, sum(Count) AS Total FROM tempview GROUP BY Region").limit(5).toPandas()


col_list= df.columns[0:5]
df3=df.select(col_list)



from pyspark.ml.feature import SQLTransformer

sqlTrans = SQLTransformer(
    statement="SELECT PFA,Region,Offence FROM __THIS__") ## placeholder
sqlTrans.transform(df).show(5)

type(sqlTrans)

df4=sqlTrans.transform(df)


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

# Note the strange naming convention of the output file in the path that you specified. 
# Spark uses Hadoop File Format, which requires data to be partitioned - that's why you have part- files. 
# If you want to rename your written files to a more user friendly format, you can do that using the method below:

from py4j.java_gateway import java_import
java_import(spark._jvm, 'org.apache.hadoop.fs.Path')

fs = spark._jvm.org.apache.hadoop.fs.FileSystem.get(spark._jsc.hadoopConfiguration())
file = fs.globStatus(spark._jvm.Path('write_test.csv/part*'))[0].getPath().getName()
fs.rename(spark._jvm.Path('write_test.csv/' + file), spark._jvm.Path('write_test2.csv')) #these two need to be different
fs.delete(spark._jvm.Path('write_test.csv'), True)

# WRITE in Data

students.write.mode("overwrite").csv('write_test.csv')

users1_2.write.mode("overwrite").parquet('parquet/')

users1_2.write.mode("overwrite").partitionBy("gender").parquet('part_parquet/')

