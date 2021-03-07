import datetime
from googleapiclient import discovery
from oauth2client.client import GoogleCredentials

id = 'machinelearning-XXX'
bucket_name = "keras-XXX-folder"
project_id = 'projects/{}'.format(id)
job_name = "training_" + datetime.datetime.now().strftime("%y%m%d_%H%M%S")

def main(event, context):

     training_inputs = {
     'scaleTier': 'BASIC',
     'packageUris': ["gs://{}/trainer".format(bucket_name)],
     'pythonModule': 'trainer.task',
     'region': 'us-central1',
     'jobDir': "gs://{}".format(bucket_name),
     'runtimeVersion': '2.2',
     'pythonVersion': '3.7',
          }

     job_spec = {"jobId":job_name, "trainingInput": training_inputs}
     cloudml = discovery.build("ml" , "v1" ,cache_discovery=False)
     request = cloudml.projects().jobs().create(body=job_spec,parent=project_id)
     response = request.execute()
