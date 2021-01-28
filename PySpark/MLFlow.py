from pyspark.ml.classification import *
from pyspark.ml.evaluation import *
from pyspark.sql.functions import *
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

import warnings

# Mlflow libaries
import mlflow
from mlflow import spark


# In[81]:


# Set experiment
# This will actually automatically create one if the one you call on doesn't exist
mlflow.set_experiment(experiment_name = "Experiment-3")

# set up your client
from  mlflow.tracking import MlflowClient
client = MlflowClient()


# In[137]:


# Create a run and attach it to the experiment you just created
experiments = client.list_experiments() # returns a list of mlflow.entities.Experiment

experiment_name = "Experiment-3"
def create_run(experiment_name):
    mlflow.set_experiment(experiment_name = experiment_name)
    for x in experiments:
        if experiment_name in x.name:
#             print(experiment_name)
#             print(x)
            experiment_index = experiments.index(x)
            run = client.create_run(experiments[experiment_index].experiment_id) # returns mlflow.entities.Run
#             print(run)
            return run

# Example run command
# run = create_run('Experiment-3')
# run = create_run(experiment_name)


# In[97]:


# test the functionality here
run = create_run('Experiment-3')

# Add tag to a run
client.set_tag(run.info.run_id, "Algorithm", "Gradient Boosted Tree")
client.set_tag(run.info.run_id,"Random Seed",908)
client.set_tag(run.info.run_id,"Train Perct",0.7)

# Add params and metrics to a run
client.log_param(run.info.run_id, "Max Depth", 90)
client.log_param(run.info.run_id, "Max Bins", 50)
client.log_metric(run.info.run_id, "Accuracy", 0.87)

# Terminate the client
client.set_terminated(run.info.run_id)


# In[30]:

from pyspark.ml.evaluation import MulticlassClassificationEvaluator,BinaryClassificationEvaluator

# Set up our classification and evaluation objects
Bin_evaluator = BinaryClassificationEvaluator(rawPredictionCol='prediction') #labelCol='label'
MC_evaluator = MulticlassClassificationEvaluator(metricName="accuracy") # redictionCol="prediction",


# ### Logistic Regression without Cross Validation
# 
# **Review**
# The Logistic Regression Algorithm, also known as "Logit", is used to estimate (guess) the probability (a number between 0 and 1) of an event occurring having been given some previous data to “learn” from. It works with either binary or multinomial (more than 2 categories) data and uses logistic function (ie. log) to find a model that fits with the data points.
# 
# **Example**
# You may want to predict the likelihood of a student passing or failing an exam based on a set of biographical factors. The model you create will provide a probability (i.e a number between 0 and 1) that you can use to determine the likelihood of each student passing.
# 
# PySpark Documentation Link: https://spark.apache.org/docs/latest/api/python/pyspark.ml.html#pyspark.ml.classification.LogisticRegression

# In[139]:


# Create a run
run = create_run(experiment_name)

# Simplist Method
classifier = LogisticRegression()
fitModel = classifier.fit(train)

# Evaluate
predictionAndLabels = fitModel.transform(test)
# predictionAndLabels = predictionAndLabels.predictions.select('label','prediction')
auc = Bin_evaluator.evaluate(predictionAndLabels)
print("AUC:",auc)

predictions = fitModel.transform(test)
accuracy = (MC_evaluator.evaluate(predictions))*100
print("Accuracy: {0:.2f}".format(accuracy),"%") #     print("Test Error = %g " % (1.0 - accuracy))
print(" ")

# Log metric to MLflow
client.log_metric(run.info.run_id, "Accuracy", accuracy)

# Extract params of Best Model
paramMap = fitModel.extractParamMap()

# Log parameters to the client
for key, val in paramMap.items():
    if 'maxIter' in key.name:
        client.log_param(run.info.run_id, "Max Iter", val)
for key, val in paramMap.items():
    if 'regParam' in key.name:
        client.log_param(run.info.run_id, "Reg Param", val)
        
# Set a runs status to finished (best practice)
client.set_terminated(run.info.run_id)

