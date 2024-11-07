# inference

import mlflow.pyfunc
import numpy as np

import dagshub
dagshub.init(repo_owner='Abhi-0088', repo_name='dagshub-demo', mlflow=True)

mlflow.set_tracking_uri("https://dagshub.com/Abhi-0088/dagshub-demo.mlflow")

data = np.array([1,85,66,29,0,26.6,0.351,31]).reshape(1,-1)

model_name = "diabetes-rf"
model_version = 1

model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/{model_version}")

print(model.predict(data))