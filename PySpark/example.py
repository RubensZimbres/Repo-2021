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
df = spark.read.csv(path,inferSchema=True,header=True)

df.limit(5).toPandas()

df.groupBy("gender").agg({'math score':'mean'}).show()
df.groupBy("product").agg(min(df.price).alias("Min Price"),max(df.price).alias("Max Price")).show(5)
df.groupBy("host_id").sum('number_of_reviews').show(10)

df.agg({'minimum_nights':'avg'}).withColumnRenamed("avg(minimum_nights)", "Avg Min Nights").show()

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

df = df.withColumn('publish_time_2',regexp_replace(df.publish_time, 'T', ' '))
df = df.withColumn('publish_time_2',regexp_replace(df.publish_time_2, 'Z', ''))
df = df.withColumn("publish_time_3", to_timestamp(df.publish_time_2, 'yyyy-MM-dd HH:mm:ss.SSS'))

df = sales.select('Date',split(sales.Date, '/')[0].alias('Month'),'Total')

from pyspark.sql.functions import when
clean = tweets.withColumn('Party', when(tweets.Party == 'Democrat', 'Democrat').when(tweets.Party == 'Republican', 'Republican').otherwise('Other'))
counts = clean.groupBy("Party").count()
counts.orderBy(desc("count")).show(16)

from pyspark.sql.functions import year, month
# Other options: dayofmonth, dayofweek, dayofyear, weekofyear
df.select("trending_date",year("trending_date"),month("trending_date")).show(5)

from pyspark.sql.functions import datediff
df.select("trending_date","publish_time_3",(datediff(df.trending_date,df.publish_time_3)/365).alias('diff')).show(5)

df = df.withColumn('title',lower(df.title)) # or rtrim/ltrim
df.select("title").show(5,False)

###### OR

import pyspark.sql.functions as f
df.select("publish_time",f.translate(f.col("publish_time"), "TZ", " ").alias("translate_func")).show(5,False)

df.createOrReplaceTempView("tempview")
spark.sql("SELECT Region, sum(Count) AS Total FROM tempview GROUP BY Region").limit(5).toPandas()
spark.sql("SELECT * FROM tempview WHERE App LIKE '%dating%'").limit(5).toPandas()


print("Option#1: select or withColumn() using when-otherwise")
from pyspark.sql.functions import when
df.select("likes","dislikes",(when(df.likes > df.dislikes, 'Good').when(df.likes < df.dislikes, 'Bad').otherwise('Undetermined')).alias("Favorability")).show(3)

print("Option2: select or withColumn() using expr function")
from pyspark.sql.functions import expr 
df.select("likes","dislikes",expr("CASE WHEN likes > dislikes THEN  'Good' WHEN likes < dislikes THEN 'Bad' ELSE 'Undetermined' END AS Favorability")).show(3)

print("Option 3: selectExpr() using SQL equivalent CASE expression")
df.selectExpr("likes","dislikes","CASE WHEN likes > dislikes THEN  'Good' WHEN likes < dislikes THEN 'Bad' ELSE 'Undetermined' END AS Favorability").show(3)

from pyspark.sql.functions import concat_ws
df.select(concat_ws(' ', df.title,df.channel_title,df.tags).alias('text')).show(1,False)

from pyspark.sql.functions import split
df.select("title").show(1,False)
df.select(split(df.title, ' ').alias('new')).show(1,False)

# CLEAN
df = reviews.withColumn("cleaned_reviews", trim(lower(regexp_replace(col('review_txt'),'[^\sa-zA-Z0-9]', ''))))

col_list= df.columns[0:5]
df3=df.select(col_list)


###### SQL

inner_join = eats_plants.join(eats_meat, ["name","id"],"inner")

left_join = eats_plants.join(eats_meat, ["name","id"], how='left') # Could also use 'left_outer'

+-----------+---+-----------+---------+
|       name| id|eats_plants|eats_meat|
+-----------+---+-----------+---------+
|       deer|  3|        yes|     null|
|      human|  4|        yes|      yes|
|      koala|  1|        yes|     null|
|caterpillar|  2|        yes|     null|
+-----------+---+-----------+---------+

conditional_join = eats_plants.join(eats_meat, ["name","id"], how='left').filter(eats_meat.name.isNotNull())

### DIFF NAMES 2 TABLES

step1 = teachings.join(instructors, teachings.instructor_id == instructors.id, how='left').select(['instructor_id','name','section_uuid'])

df_list = []
for filename in os.listdir(path):
    if filename.endswith(".csv"):
        filename_list = filename.split(".") #separate path from .csv
        df_name = filename_list[0]
        df = spark.read.csv(path+filename,inferSchema=True,header=True)
        df.name = df_name
        df_list.append(df_name)
        exec(df_name + ' = df')

from pyspark.sql.functions import levenshtein

df0 = spark.createDataFrame([('Aple', 'Apple','Microsoft','IBM')], ['Input', 'Option1','Option2','Option3'])
print("Correct this company name: Aple")
df0.select(levenshtein('Input', 'Option1').alias('Apple')).show()
df0.select(levenshtein('Input', 'Option2').alias('Microsoft')).show()
df0.select(levenshtein('Input', 'Option3').alias('IBM')).show()

# MISSING

df.filter(df.cuisines.isNull()).select(['name','cuisines']).show(5)

from pyspark.sql.functions import *

def null_value_calc(df):
    null_columns_counts = []
    numRows = df.count()
    for k in df.columns:
        nullRows = df.where(col(k).isNull()).count()
        if(nullRows > 0):
            temp = k,nullRows,(nullRows/numRows)*100
            null_columns_counts.append(temp)
    return(null_columns_counts)

null_columns_calc_list = null_value_calc(df)
spark.createDataFrame(null_columns_calc_list, ['Column_Name', 'Null_Values_Count','Null_Value_Percent']).show()

df.na.drop().limit(4).toPandas() 

# OR

drop_len = df.na.drop(subset=["votes"]).count() 

df.na.fill(999).limit(10).toPandas()

df.filter(df.name.isNull()).na.fill('No Name',subset=['name']).limit(5).toPandas()

def fill_with_mean(df, include=set()): 
    stats = df.agg(*(avg(c).alias(c) for c in df.columns if c in include))
    return df.na.fill(stats.first().asDict())

updated_df = fill_with_mean(df, ["votes"])

from pyspark.ml.feature import SQLTransformer

sqlTrans = SQLTransformer(
    statement="SELECT PFA,Region,Offence FROM __THIS__") ## placeholder
sqlTrans.transform(df).show(5)

type(sqlTrans)

df4=sqlTrans.transform(df)

from pyspark.sql.functions import expr 

sqlTrans = SQLTransformer(
    statement="SELECT SUM(Count) as Total FROM __THIS__") 
sqlTrans.transform(df).show(5)

df.withColumn("percent",expr("round((count/244720928)*100,2)")).show()
df.select("*",expr("round((count/244720928)*100,2) AS percent")).show()

from pyspark.sql.types import *
df = googlep.withColumn("Rating", googlep["Rating"].cast(FloatType()))  .withColumn("Reviews", googlep["Reviews"].cast(IntegerType()))  .withColumn("Price", googlep["Price"].cast(IntegerType()))
print(df.printSchema())

##### Available types:
#     - DataType
#     - NullType
#     - StringType
#     - BinaryType
#     - BooleanType
#     - DateType
#     - TimestampType
#     - DecimalType
#     - DoubleType
#     - FloatType
#     - ByteType
#     - IntegerType
#     - LongType
#     - ShortType
#     - ArrayType
#     - MapType
#     - StructField
#     - StructType

################################################################################

# Until we executute a command like this
collect = df.collect()

# Even if we duplicate the dataframe, the ID remains the same
df2 = df
df2.rdd.id()

# Iterate over a column

from pyspark.sql.functions import udf
from pyspark.sql.types import IntegerType

def square(x):
    return int(x**2)
square_udf = udf(lambda z: square(z), IntegerType())

df.select('dislikes',square_udf('dislikes').alias('likes_sq')).where(col('dislikes').isNotNull()).show()

#######################################################################

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

############## MACHINE LEARNING

## LABEL

dependent_var = 'Class/ASD Traits '
renamed = df.withColumn("label_str", df[dependent_var].cast(StringType())) #Rename and change to string type
indexer = StringIndexer(inputCol="label_str", outputCol="label") #Pyspark is expecting the this naming convention 
indexed = indexer.fit(renamed).transform(renamed)
indexed.toPandas()['label']

# STRING TO NUMERICAL

input_columns = df.columns
numeric_inputs = []
string_inputs = []
for column in input_columns:
    # First identify the string vars in your input column list
    if str(indexed.schema[column].dataType) == 'StringType':
        # Set up your String Indexer function
        indexer = StringIndexer(inputCol=column, outputCol=column+"_num") 
        # Then call on the indexer you created here
        indexed = indexer.fit(indexed).transform(indexed)
        # Rename the column to a new name so you can disinguish it from the original
        new_col_name = column+"_num"
        # Add the new column name to the string inputs list
        string_inputs.append(new_col_name)
    else:
        # If no change was needed, take no action 
        # And add the numeric var to the num list
        numeric_inputs.append(column)

## SKEWNESS
        
d = {}
### TOP AND BOTTOM 1%
for col in numeric_inputs: 
    d[col] = indexed.approxQuantile(col,[0.01,0.99],0.25) #if you want to make it go faster increase the last number

for col in numeric_inputs:
    skew = indexed.agg(skewness(indexed[col])).collect() #check for skewness
    skew = skew[0][0]
    if skew > 1: # If right skew, floor, cap and log(x+1)
        indexed = indexed.withColumn(col,         log(when(df[col] < d[col][0],d[col][0])        .when(indexed[col] > d[col][1], d[col][1])        .otherwise(indexed[col] ) +1).alias(col))
        print(col+" has been treated for positive (right) skewness. (skew =)",skew,")")
    elif skew < -1: # If left skew floor, cap and exp(x)
        indexed = indexed.withColumn(col,         exp(when(df[col] < d[col][0],d[col][0])        .when(indexed[col] > d[col][1], d[col][1])        .otherwise(indexed[col] )).alias(col))
        print(col+" has been treated for negative (left) skewness. (skew =",skew,")")

## FEATURES AND LABEL

indexed.toPandas()[features_list]

indexed.toPandas()['label']

features_list = numeric_inputs + string_inputs
assembler = VectorAssembler(inputCols=features_list,outputCol='features')
output = assembler.transform(indexed).select('features','label')

output.toPandas()

                                               features  label
0     (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, ...    1.0
1     (1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, ...    0.0
2     (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, ...    0.0
3     [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, ...    0.0
4     [1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, ...    0.0
...                                                 ...    ...
1049  (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ...    1.0
1050  (0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, ...    0.0
1051  [1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, ...    0.0
1052  (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, ...    1.0
1053  [1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, ...    0.0
       
scaler = MinMaxScaler(inputCol="features", outputCol="scaledFeatures",min=0,max=1000)
# Compute summary statistics and generate MinMaxScalerModel
scalerModel = scaler.fit(output)
scaled_data = scalerModel.transform(output)
final_data = scaled_data.select('label','scaledFeatures')
final_data = final_data.withColumnRenamed("scaledFeatures","features")
final_data.show()

+-----+--------------------+
|label|            features|
+-----+--------------------+
|  1.0|(17,[6,7,9,10,11,...|
|  0.0|(17,[0,1,5,6,10,1...|
|  0.0|(17,[0,6,7,9,10,1...|
|  0.0|[1000.0,1000.0,10...|
|  0.0|[1000.0,1000.0,0....|
|  0.0|[1000.0,1000.0,0....|
|  0.0|(17,[0,3,4,5,8,10...|
+-----+--------------------+
    
train,test = final_data.randomSplit([0.7,0.3])

from pyspark.ml.classification import *
from pyspark.ml.evaluation import *
from pyspark.sql.functions import *
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

Bin_evaluator = BinaryClassificationEvaluator(rawPredictionCol='predictions') #labelCol='label'
MC_evaluator = MulticlassClassificationEvaluator(metricName="accuracy") # redictionCol="prediction",

from pyspark.ml.evaluation import BinaryClassificationEvaluator
classifier = LogisticRegression()
fitModel=classifier.fit(train)

predictions = fitModel.transform(test)
evaluator = BinaryClassificationEvaluator()
print('Test Area Under ROC', evaluator.evaluate(predictions))

#####################

# Amount labels

class_count=final_data.select(countDistinct("label")).collect()

classifier = LogisticRegression()
# Then Set up your parameter grid for the cross validator to conduct hyperparameter tuning
paramGrid = (ParamGridBuilder().addGrid(classifier.maxIter, [10, 15,20]).build())
# Then set up the Cross Validator which requires all of the following parameters:
crossval = CrossValidator(estimator=classifier,
                          estimatorParamMaps=paramGrid,
                          evaluator=MC_evaluator,
                          numFolds=2) # 3 + is best practice

fitModel = crossval.fit(train)

            BestModel = fitModel.bestModel
print("Intercept: " + str(BestModel.interceptVector))
print("Coefficients: \n" + str(BestModel.coefficientMatrix))

LR_BestModel = BestModel

predictions = fitModel.transform(test)

accuracy = (MC_evaluator.evaluate(predictions))*100
print(accuracy)

####################### MLP
            
from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

train.toPandas().shape[0]

layers=[17,128,128,train.toPandas().shape[1]]
classifier = MultilayerPerceptronClassifier(maxIter=100,layers=layers,blockSize=128,seed=4)
fitModel = classifier.fit(train)

predictionAndLabels = fitModel.transform(test)

accuracy = (MC_evaluator.evaluate(predictions))*100
print(accuracy)

##### NAIVE BAYES
classifier = NaiveBayes()
paramGrid = (ParamGridBuilder().addGrid(classifier.smoothing, [0,.2, .4,.6]).build())

##### SVM
classifier = LinearSVC()
paramGrid = (ParamGridBuilder().addGrid(classifier.maxIter, [10, 15]).addGrid(classifier.regParam, [.1, .01]).build())

###### DECISION TREES
classifier = DecisionTreeClassifier()
paramGrid = (ParamGridBuilder().addGrid(classifier.maxBins, [10, 20,40,80,100]).build())

###### RANDOM FOREST
classifier = RandomForestClassifier()
paramGrid = (ParamGridBuilder().addGrid(classifier.maxDepth, [2,5,10]).build())
BestModel.featureImportances.toArray()
array([7.42616028e-02, 2.33261315e-02, 1.57644374e-02, 1.83837109e-02,
       2.15133684e-02, 1.33275126e-01, 5.47811046e-02, 1.58630216e-02,
       1.19127272e-01, 3.54069589e-03, 4.33353182e-03, 5.12711914e-01,
       1.16754626e-03, 1.01202259e-03, 0.00000000e+00, 6.80487967e-04,
       2.58025630e-04])
            
####### Gradient Boosted Trees
classifier = GBTClassifier()
paramGrid = (ParamGridBuilder().addGrid(classifier.maxDepth, [2,5,10]).build())

############# REGRESSION
from pyspark.ml.stat import Correlation
from pyspark.ml.regression import *
from pyspark.ml.evaluation import *
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
          
pearsonCorr = Correlation.corr(final_data, 'features', 'pearson').collect()[0][0]
array = pearsonCorr.toArray()

for item in array:
    print(item[0])
    print(" ")
    print(item[1])
    print(" ")
    print(item[2])
    #etc

regressor = LinearRegression()
fitModel = regressor.fit(train)

evaluator = RegressionEvaluator(metricName="rmse")
predictions = fitModel.transform(test)

rmse = evaluator.evaluate(predictions)
print("Root Mean Squared Error (RMSE) on test data = %g" % rmse)

trainingSummary = fitModel.summary

print('\033[1m' + "Linear Regression Model Summary without cross validation:"+ '\033[0m')
print(" ")
print("Intercept: %s" % str(fitModel.intercept))
print("")
coeff_array = fitModel.coefficients.toArray()
coeff_scores = []
for x in coeff_array:
    coeff_scores.append(float(x))

result = spark.createDataFrame(zip(input_columns,coeff_scores), schema=['feature','coeff'])
print(result.orderBy(result["coeff"].desc()).show(truncate=False))
            
-------------
            
paramGrid = (ParamGridBuilder()              .addGrid(regressor.maxIter, [10, 15])              .addGrid(regressor.regParam, [0.1, 0.01])              .build())

evaluator = RegressionEvaluator(metricName="rmse")

crossval = CrossValidator(estimator=regressor,
                          estimatorParamMaps=paramGrid,
                          evaluator=evaluator,
                          numFolds=2) # 3 is best practice

print('\033[1m' + "Linear Regression Model Summary WITH cross validation:"+ '\033[0m')
print(" ")

fitModel = crossval.fit(train)

LR_BestModel = fitModel.bestModel

ModelSummary = LR_BestModel.summary
print("Coefficient Standard Errors: ")
coeff_ste = ModelSummary.coefficientStandardErrors
result = spark.createDataFrame(zip(input_columns,coeff_ste), schema=['feature','coeff std error'])
print(result.orderBy(result["coeff std error"].desc()).show(truncate=False))
print(" ")
print("P Values: ") 
pvalues = ModelSummary.pValues
result = spark.createDataFrame(zip(input_columns,pvalues), schema=['feature','P-Value'])
print(result.orderBy(result["P-Value"].desc()).show(truncate=False))
print(" ")

############### K-MEANS - LDA

from pyspark.sql.functions import *

def null_value_calc(df):
    null_columns_counts = []
    numRows = df.count()
    for k in df.columns:
        nullRows = df.where(col(k).isNull()).count()
        if(nullRows > 0):
            temp = k,nullRows,(nullRows/numRows)*100
            null_columns_counts.append(temp)
    return(null_columns_counts)

null_columns_calc_list = null_value_calc(df)
spark.createDataFrame(null_columns_calc_list, ['Column_Name', 'Null_Values_Count','Null_Value_Percent']).show()

from pyspark.sql.functions import *
def fill_with_mean(df, include=set()): 
    stats = df.agg(*(avg(c).alias(c) for c in df.columns if c in include))
    return df.na.fill(stats.first().asDict())

columns = df.columns
columns = columns[1:]
df = fill_with_mean(df, columns)

from pyspark.ml.feature import VectorAssembler
input_columns = df.columns # Collect the column names as a list
input_columns = input_columns[1:] # keep only relevant columns: from column 8 until the end
vecAssembler = VectorAssembler(inputCols=input_columns, outputCol="features")
df_kmeans = vecAssembler.transform(df) #.select('CUST_ID', 'features')
df_kmeans.limit(4).toPandas()

from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
import numpy as np

kmax = 50
kmcost = np.zeros(kmax)
for k in range(2,kmax):
    kmeans = KMeans().setK(k).setSeed(1).setFeaturesCol("features")
    model = kmeans.fit(df_kmeans)
    # Fill in the zeros of your array with cost....
    # Computes the "cost" (sum of squared distances) between the input points and their corresponding cluster centers.
    predictions = model.transform(df_kmeans)
    evaluator = ClusteringEvaluator()
    kmcost[k] = evaluator.evaluate(predictions) #computing Silhouette score

## ELBOW

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig, ax = plt.subplots(1,1, figsize =(8,6))
ax.plot(range(2,kmax),kmcost[2:kmax])
ax.set_xlabel('k')
ax.set_ylabel('cost')
plt.show()

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig, ax = plt.subplots(1,1, figsize =(8,6))
ax.plot(range(2,kmax),kmcost[2:kmax])
ax.set_xlabel('k')
ax.set_ylabel('cost')
plt.show()

            
#### BISECTING K-MEANS - tree

from pyspark.ml.clustering import BisectingKMeans

kmax = 50
bkmcost = np.zeros(kmax)
for k in range(2,kmax):
    bkmeans = BisectingKMeans().setK(k).setSeed(1).setFeaturesCol("features")
    model = bkmeans.fit(df_kmeans)
    predictions = model.transform(df_kmeans)
    evaluator = ClusteringEvaluator()
    bkmcost[k] = evaluator.evaluate(predictions) #computes Silhouette score

#################
            
k = 15
bkmeans = BisectingKMeans().setK(k).setSeed(1).setFeaturesCol("features")
model = bkmeans.fit(df_kmeans)

predictions = model.transform(df_kmeans)

evaluator = ClusteringEvaluator()

silhouette = evaluator.evaluate(predictions)
print("Silhouette with squared euclidean distance = " + str(silhouette))
print(" ")

centers = model.clusterCenters()
print("Cluster Centers: ")
for center in centers:
    print(center)

import pandas as pd
import numpy as np
center_pdf = pd.DataFrame(list(map(np.ravel,centers)))
center_pdf.columns = columns
center_pdf

predictions.limit(5).toPandas()

########## LDA

df = spark.read.json(path+'recipes.json')

            # Tokenize
regex_tokenizer = RegexTokenizer(inputCol="Description", outputCol="words", pattern="\\W")
raw_words = regex_tokenizer.transform(df_clean)

# Remove Stop words
remover = StopWordsRemover(inputCol="words", outputCol="filtered")
words_df = remover.transform(raw_words)

# Zero Index Label Column
cv = CountVectorizer(inputCol="filtered", outputCol="features")
cvmodel = cv.fit(words_df)
df_vect = cvmodel.transform(words_df)
            
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

kmax = 30
ll = np.zeros(kmax)
lp = np.zeros(kmax)
for k in range(2,kmax):
    lda = LDA(k=k, maxIter=10)
    model = lda.fit(df_vect)
    ll[k] = model.logLikelihood(df_vect)
    lp[k] = model.logPerplexity(df_vect)
    
fig, ax = plt.subplots(1,1, figsize =(8,6))
ax.plot(range(2,kmax),ll[2:kmax])
ax.set_xlabel('k')
ax.set_ylabel('ll')

fig, ax = plt.subplots(1,1, figsize =(8,6))
ax.plot(range(2,kmax),lp[2:kmax])
ax.set_xlabel('k')
ax.set_ylabel('lp')

print("Recap of ll and lp:")
ll = model.logLikelihood(df_vect)
lp = model.logPerplexity(df_vect)
print("The lower bound on the log likelihood of the entire corpus: " + str(ll))
print("The upper bound on perplexity: " + str(lp))
print("Vocab Size: ", model.vocabSize())

print("The topics described by their top-weighted terms:")
topics = model.describeTopics(maxTermsPerTopic = 4)
topics = topics.collect()
vocablist = cvmodel.vocabulary
for x, topic in enumerate(topics):
    print(" ")
    print('TOPIC: ' + str(x))
    # This is like a temp holder
    topic = topics
    # Then we extract the words from the topics
    words = topic[x][1]
    # Then print the words by topics
    for n in range(len(words)):
        print(vocablist[words[n]]) # + ' ' + str(weights[n])

# Make predictions
transformed = model.transform(df_vect)
transformed.toPandas()

# Convert topicdistribution col from vector to array
to_array = udf(lambda x: x.toArray().tolist(), ArrayType(DoubleType()))
recommendations = transformed.withColumn('array', to_array('topicDistribution'))

# Find the best topic value that we will call "max"
max_vals = recommendations.withColumn("max",array_max("array"))

# Find the index of the max value found above which translates to our topic!
argmaxUdf = udf(lambda x,y: [i for i, e in enumerate(x) if e==y ])
results = max_vals.withColumn('topic', argmaxUdf(max_vals.array,max_vals.max))
results.printSchema()
results.limit(4).toPandas()

############ GAUSSIAN MIXTURE MODEL

kmax = 50
ll = np.zeros(kmax)
for k in range(2,kmax):
    gm = GaussianMixture(k=k, tol=0.0001,maxIter=10, seed=10)
    model = gm.fit(final_df)
    summary = model.summary
    ll[k] = summary.logLikelihood

fig, ax = plt.subplots(1,1, figsize =(8,6))
ax.plot(range(2,kmax),ll[2:kmax])
ax.set_xlabel('k')
ax.set_ylabel('ll')

gm = GaussianMixture(k=5, maxIter=10, seed=10)
model = gm.fit(final_df)

summary = model.summary
print("Clusters: ",summary.k)
print("Cluster Sizes: ",summary.clusterSizes)
print("Log Likelihood: ",summary.logLikelihood)

weights = model.weights
print("Model Weights: :",len(weights))

print("Means: ", model.gaussiansDF.select("mean").head())

print("Cov: ",model.gaussiansDF.select("cov").head())

transformed = model.transform(final_df)#.select("features", "prediction")

transformed.limit(7).toPandas()
transformed.show(1,False)

transformed.groupBy("prediction").agg({"prediction":"count",'QUANTITYORDERED':'min','PRICEEACH':'min','SALES':'min'}).orderBy("prediction").show()

limited = transformed.filter("prediction == 0")
aggregates = limited.summary("min", "mean", "max")
print("Total Cases in this Cluster: ",limited.count())
aggregates.toPandas()

