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
