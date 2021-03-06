import time
import io
import os
from google.cloud import speech
from google.cloud.speech import enums
from google.cloud.speech import types
from google.cloud import storage
import os
from google.oauth2 import service_account
import os

os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="/home/anaconda3/work/machinelearning-XXXXX.json"

client = storage.Client()

