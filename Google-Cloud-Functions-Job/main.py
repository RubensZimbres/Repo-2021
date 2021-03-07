import datetime
from googleapiclient import discovery
from oauth2client.client import GoogleCredentials

id = 'machinelearning-XXXX'
bucket_name = "bucket-XXX"
project_id = 'projects/{}'.format(id)
job_name = "training_" + datetime.datetime.now().strftime("%y%m%d_%H%M%S")

def main(event, context):

     training_inputs = {
     'scaleTier': 'BASIC',
     'packageUris': [f"gs://{bucket_name}/package/trainer-0.1.tar.gz"],
     'pythonModule': 'trainer.task',
     'region': 'asia-northeast1',
     'jobDir': f"gs://{bucket_name}",
     'runtimeVersion': '2.2',
     'pythonVersion': '3.7',
          }

     job_spec = {"jobId":job_name, "trainingInput": training_inputs}
     cloudml = discovery.build("ml" , "v1" ,cache_discovery=False)
     request = cloudml.projects().jobs().create(body=job_spec,parent=project_id)
     response = request.execute()
