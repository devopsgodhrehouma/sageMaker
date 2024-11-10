# Lab 3.5 - Student Notebook

## Overview

This lab is a continuation of the guided labs in Module 3. 

In this lab, you will deploy a trained model and perform a prediction against the model. You will then delete the endpoint and perform a batch transform on the test dataset.


## Introduction to the business scenario

You work for a healthcare provider, and want to improve the detection of abnormalities in orthopedic patients. 

You are tasked with solving this problem by using machine learning (ML). You have access to a dataset that contains six biomechanical features and a target of *normal* or *abnormal*. You can use this dataset to train an ML model to predict if a patient will have an abnormality.


## About this dataset

This biomedical dataset was built by Dr. Henrique da Mota during a medical residence period in the Group of Applied Research in Orthopaedics (GARO) of the Centre Médico-Chirurgical de Réadaptation des Massues, Lyon, France. The data has been organized in two different, but related, classification tasks. 

The first task consists in classifying patients as belonging to one of three categories: 

- *Normal* (100 patients)
- *Disk Hernia* (60 patients)
- *Spondylolisthesis* (150 patients)

For the second task, the categories *Disk Hernia* and *Spondylolisthesis* were merged into a single category that is labeled as *abnormal*. Thus, the second task consists in classifying patients as belonging to one of two categories: *Normal* (100 patients) or *Abnormal* (210 patients).


## Attribute information

Each patient is represented in the dataset by six biomechanical attributes that are derived from the shape and orientation of the pelvis and lumbar spine (in this order): 

- Pelvic incidence
- Pelvic tilt
- Lumbar lordosis angle
- Sacral slope
- Pelvic radius
- Grade of spondylolisthesis

The following convention is used for the class labels: 
- DH (Disk Hernia)
- Spondylolisthesis (SL)
- Normal (NO) 
- Abnormal (AB)

For more information about this dataset, see the [Vertebral Column dataset webpage](http://archive.ics.uci.edu/ml/datasets/Vertebral+Column).


## Dataset attributions

This dataset was obtained from:
Dua, D. and Graff, C. (2019). UCI Machine Learning Repository (http://archive.ics.uci.edu/ml). Irvine, CA: University of California, School of Information and Computer Science.


# Lab setup

Because this solution is split across several labs in the module, you run the following cells so that you can load the data and train the model to be deployed.

**Note:** The setup can take up to 5 minutes to complete.

## Importing the data

By running the following cells, the data will be imported and ready for use. 

**Note:** The following cells represent the key steps in the previous labs.



```python
bucket='c124417a3052644l8324406t1w858105440430-labbucket-jkgcbnt6qt7t'
```


```python
print(bucket)
```

    c124417a3052644l8324406t1w858105440430-labbucket-jkgcbnt6qt7t



```python
import warnings, requests, zipfile, io
warnings.simplefilter('ignore')
import pandas as pd
from scipy.io import arff

import os
import boto3
import sagemaker
from sagemaker.image_uris import retrieve
from sklearn.model_selection import train_test_split
```

    sagemaker.config INFO - Not applying SDK defaults from location: /etc/xdg/sagemaker/config.yaml
    sagemaker.config INFO - Not applying SDK defaults from location: /home/ec2-user/.config/sagemaker/config.yaml



```python
f_zip = 'http://archive.ics.uci.edu/ml/machine-learning-databases/00212/vertebral_column_data.zip'
r = requests.get(f_zip, stream=True)
Vertebral_zip = zipfile.ZipFile(io.BytesIO(r.content))
Vertebral_zip.extractall()

data = arff.loadarff('column_2C_weka.arff')
df = pd.DataFrame(data[0])

class_mapper = {b'Abnormal':1,b'Normal':0}
df['class']=df['class'].replace(class_mapper)

cols = df.columns.tolist()
cols = cols[-1:] + cols[:-1]
df = df[cols]

train, test_and_validate = train_test_split(df, test_size=0.2, random_state=42, stratify=df['class'])
test, validate = train_test_split(test_and_validate, test_size=0.5, random_state=42, stratify=test_and_validate['class'])

prefix='lab3'

train_file='vertebral_train.csv'
test_file='vertebral_test.csv'
validate_file='vertebral_validate.csv'

s3_resource = boto3.Session().resource('s3')
def upload_s3_csv(filename, folder, dataframe):
    csv_buffer = io.StringIO()
    dataframe.to_csv(csv_buffer, header=False, index=False )
    s3_resource.Bucket(bucket).Object(os.path.join(prefix, folder, filename)).put(Body=csv_buffer.getvalue())

upload_s3_csv(train_file, 'train', train)
upload_s3_csv(test_file, 'test', test)
upload_s3_csv(validate_file, 'validate', validate)

container = retrieve('xgboost',boto3.Session().region_name,'1.0-1')

hyperparams={"num_round":"42",
             "eval_metric": "auc",
             "objective": "binary:logistic"}

s3_output_location="s3://{}/{}/output/".format(bucket,prefix)
xgb_model=sagemaker.estimator.Estimator(container,
                                       sagemaker.get_execution_role(),
                                       instance_count=1,
                                       instance_type='ml.m4.xlarge',
                                       output_path=s3_output_location,
                                        hyperparameters=hyperparams,
                                        sagemaker_session=sagemaker.Session())

train_channel = sagemaker.inputs.TrainingInput(
    "s3://{}/{}/train/".format(bucket,prefix,train_file),
    content_type='text/csv')

validate_channel = sagemaker.inputs.TrainingInput(
    "s3://{}/{}/validate/".format(bucket,prefix,validate_file),
    content_type='text/csv')

data_channels = {'train': train_channel, 'validation': validate_channel}

xgb_model.fit(inputs=data_channels, logs=False)

print('ready for hosting!')
```

    INFO:sagemaker:Creating training-job with name: sagemaker-xgboost-2024-11-10-13-47-44-006


    
    2024-11-10 13:47:45 Starting - Starting the training job...
    2024-11-10 13:48:09 Starting - Preparing the instances for training.........
    2024-11-10 13:48:59 Downloading - Downloading input data....
    2024-11-10 13:49:25 Downloading - Downloading the training image..............
    2024-11-10 13:50:36 Training - Training image download completed. Training in progress...
    2024-11-10 13:50:51 Uploading - Uploading generated training model.
    2024-11-10 13:51:04 Completed - Training job completed
    ready for hosting!


# Step 1: Hosting the model

Now that you have a trained model, you can host it by using Amazon SageMaker hosting services.

The first step is to deploy the model. Because you have a model object, *xgb_model*, you can use the **deploy** method. For this lab, you will use a single ml.m4.xlarge instance.




```python
xgb_predictor = xgb_model.deploy(initial_instance_count=1,
                serializer = sagemaker.serializers.CSVSerializer(),
                instance_type='ml.m4.xlarge')
```

    INFO:sagemaker:Creating model with name: sagemaker-xgboost-2024-11-10-13-53-20-391
    INFO:sagemaker:Creating endpoint-config with name sagemaker-xgboost-2024-11-10-13-53-20-391
    INFO:sagemaker:Creating endpoint with name sagemaker-xgboost-2024-11-10-13-53-20-391


    ------!

# Step 2: Performing predictions

Now that you have a deployed model, you will run some predictions.

First, review the test data and re-familiarize yourself with it.


```python
test.shape
```




    (31, 7)



You have 31 instances, with seven attributes. The first five instances are:


```python
test.head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>class</th>
      <th>pelvic_incidence</th>
      <th>pelvic_tilt</th>
      <th>lumbar_lordosis_angle</th>
      <th>sacral_slope</th>
      <th>pelvic_radius</th>
      <th>degree_spondylolisthesis</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>136</th>
      <td>1</td>
      <td>88.024499</td>
      <td>39.844669</td>
      <td>81.774473</td>
      <td>48.179830</td>
      <td>116.601538</td>
      <td>56.766083</td>
    </tr>
    <tr>
      <th>230</th>
      <td>0</td>
      <td>65.611802</td>
      <td>23.137919</td>
      <td>62.582179</td>
      <td>42.473883</td>
      <td>124.128001</td>
      <td>-4.083298</td>
    </tr>
    <tr>
      <th>134</th>
      <td>1</td>
      <td>52.204693</td>
      <td>17.212673</td>
      <td>78.094969</td>
      <td>34.992020</td>
      <td>136.972517</td>
      <td>54.939134</td>
    </tr>
    <tr>
      <th>130</th>
      <td>1</td>
      <td>50.066786</td>
      <td>9.120340</td>
      <td>32.168463</td>
      <td>40.946446</td>
      <td>99.712453</td>
      <td>26.766697</td>
    </tr>
    <tr>
      <th>47</th>
      <td>1</td>
      <td>41.352504</td>
      <td>16.577364</td>
      <td>30.706191</td>
      <td>24.775141</td>
      <td>113.266675</td>
      <td>-4.497958</td>
    </tr>
  </tbody>
</table>
</div>



You don't need to include the target value (class). This predictor can take data in the comma-separated values (CSV) format. You can thus get the first row *without the class column* by using the following code:

`test.iloc[:1,1:]` 

The **iloc** function takes parameters of [*rows*,*cols*]

To only get the first row, use `0:1`. If you want to get row 2, you could use `1:2`.

To get all columns *except* the first column (*col 0*), use `1:`




```python
row = test.iloc[0:1,1:]
row.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>pelvic_incidence</th>
      <th>pelvic_tilt</th>
      <th>lumbar_lordosis_angle</th>
      <th>sacral_slope</th>
      <th>pelvic_radius</th>
      <th>degree_spondylolisthesis</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>136</th>
      <td>88.024499</td>
      <td>39.844669</td>
      <td>81.774473</td>
      <td>48.17983</td>
      <td>116.601538</td>
      <td>56.766083</td>
    </tr>
  </tbody>
</table>
</div>



You can convert this to a comma-separated values (CSV) file, and store it in a string buffer.


```python
batch_X_csv_buffer = io.StringIO()
row.to_csv(batch_X_csv_buffer, header=False, index=False)
test_row = batch_X_csv_buffer.getvalue()
print(test_row)
```

    88.0244989,39.84466878,81.77447308,48.17983012,116.6015376,56.76608323
    


Now, you can use the data to perform a prediction.


```python
xgb_predictor.predict(test_row)
```




    b'0.9966071844100952'



The result you get isn't a *0* or a *1*. Instead, you get a *probability score*. You can apply some conditional logic to the probability score to determine if the answer should be presented as a 0 or a 1. You will work with this process when you do batch predictions.

For now, compare the result with the test data.


```python
test.head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>class</th>
      <th>pelvic_incidence</th>
      <th>pelvic_tilt</th>
      <th>lumbar_lordosis_angle</th>
      <th>sacral_slope</th>
      <th>pelvic_radius</th>
      <th>degree_spondylolisthesis</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>136</th>
      <td>1</td>
      <td>88.024499</td>
      <td>39.844669</td>
      <td>81.774473</td>
      <td>48.179830</td>
      <td>116.601538</td>
      <td>56.766083</td>
    </tr>
    <tr>
      <th>230</th>
      <td>0</td>
      <td>65.611802</td>
      <td>23.137919</td>
      <td>62.582179</td>
      <td>42.473883</td>
      <td>124.128001</td>
      <td>-4.083298</td>
    </tr>
    <tr>
      <th>134</th>
      <td>1</td>
      <td>52.204693</td>
      <td>17.212673</td>
      <td>78.094969</td>
      <td>34.992020</td>
      <td>136.972517</td>
      <td>54.939134</td>
    </tr>
    <tr>
      <th>130</th>
      <td>1</td>
      <td>50.066786</td>
      <td>9.120340</td>
      <td>32.168463</td>
      <td>40.946446</td>
      <td>99.712453</td>
      <td>26.766697</td>
    </tr>
    <tr>
      <th>47</th>
      <td>1</td>
      <td>41.352504</td>
      <td>16.577364</td>
      <td>30.706191</td>
      <td>24.775141</td>
      <td>113.266675</td>
      <td>-4.497958</td>
    </tr>
  </tbody>
</table>
</div>



**Question:** Is the prediction accurate?

**Challenge task:** Update the previous code to send the second row of the dataset. Are those predictions correct? Try this task with a few other rows.

It can be tedious to send these rows one at a time. You could write a function to submit these values in a batch, but SageMaker already has a batch capability. You will examine that feature next. However, before you do, you will terminate the model.

# Step 3: Terminating the deployed model

To delete the endpoint, use the **delete_endpoint** function on the predictor.


```python
xgb_predictor.delete_endpoint(delete_endpoint_config=True)
```

    INFO:sagemaker:Deleting endpoint configuration with name: sagemaker-xgboost-2024-11-10-13-53-20-391
    INFO:sagemaker:Deleting endpoint with name: sagemaker-xgboost-2024-11-10-13-53-20-391


# Step 4: Performing a batch transform

When you are in the training-testing-feature engineering cycle, you want to test your holdout or test sets against the model. You can then use those results to calculate metrics. You could deploy an endpoint as you did earlier, but then you must remember to delete the endpoint. However, there is a more efficient way.

You can use the transformer method of the model to get a transformer object. You can then use the transform method of this object to perform a prediction on the entire test dataset. SageMaker will: 

- Spin up an instance with the model
- Perform a prediction on all the input values
- Write those values to Amazon Simple Storage Service (Amazon S3) 
- Finally, terminate the instance

You will start by turning your data into a CSV file that the transformer object can take as input. This time, you will use **iloc** to get all the rows, and all columns *except* the first column.



```python
test.head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>class</th>
      <th>pelvic_incidence</th>
      <th>pelvic_tilt</th>
      <th>lumbar_lordosis_angle</th>
      <th>sacral_slope</th>
      <th>pelvic_radius</th>
      <th>degree_spondylolisthesis</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>136</th>
      <td>1</td>
      <td>88.024499</td>
      <td>39.844669</td>
      <td>81.774473</td>
      <td>48.179830</td>
      <td>116.601538</td>
      <td>56.766083</td>
    </tr>
    <tr>
      <th>230</th>
      <td>0</td>
      <td>65.611802</td>
      <td>23.137919</td>
      <td>62.582179</td>
      <td>42.473883</td>
      <td>124.128001</td>
      <td>-4.083298</td>
    </tr>
    <tr>
      <th>134</th>
      <td>1</td>
      <td>52.204693</td>
      <td>17.212673</td>
      <td>78.094969</td>
      <td>34.992020</td>
      <td>136.972517</td>
      <td>54.939134</td>
    </tr>
    <tr>
      <th>130</th>
      <td>1</td>
      <td>50.066786</td>
      <td>9.120340</td>
      <td>32.168463</td>
      <td>40.946446</td>
      <td>99.712453</td>
      <td>26.766697</td>
    </tr>
    <tr>
      <th>47</th>
      <td>1</td>
      <td>41.352504</td>
      <td>16.577364</td>
      <td>30.706191</td>
      <td>24.775141</td>
      <td>113.266675</td>
      <td>-4.497958</td>
    </tr>
  </tbody>
</table>
</div>




```python
batch_X = test.iloc[:,1:];
batch_X.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>pelvic_incidence</th>
      <th>pelvic_tilt</th>
      <th>lumbar_lordosis_angle</th>
      <th>sacral_slope</th>
      <th>pelvic_radius</th>
      <th>degree_spondylolisthesis</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>136</th>
      <td>88.024499</td>
      <td>39.844669</td>
      <td>81.774473</td>
      <td>48.179830</td>
      <td>116.601538</td>
      <td>56.766083</td>
    </tr>
    <tr>
      <th>230</th>
      <td>65.611802</td>
      <td>23.137919</td>
      <td>62.582179</td>
      <td>42.473883</td>
      <td>124.128001</td>
      <td>-4.083298</td>
    </tr>
    <tr>
      <th>134</th>
      <td>52.204693</td>
      <td>17.212673</td>
      <td>78.094969</td>
      <td>34.992020</td>
      <td>136.972517</td>
      <td>54.939134</td>
    </tr>
    <tr>
      <th>130</th>
      <td>50.066786</td>
      <td>9.120340</td>
      <td>32.168463</td>
      <td>40.946446</td>
      <td>99.712453</td>
      <td>26.766697</td>
    </tr>
    <tr>
      <th>47</th>
      <td>41.352504</td>
      <td>16.577364</td>
      <td>30.706191</td>
      <td>24.775141</td>
      <td>113.266675</td>
      <td>-4.497958</td>
    </tr>
  </tbody>
</table>
</div>



Next, write your data to a CSV file.


```python
batch_X_file='batch-in.csv'
upload_s3_csv(batch_X_file, 'batch-in', batch_X)
```

Last, before you perform a transform, configure your transformer with the input file, output location, and instance type.


```python
batch_output = "s3://{}/{}/batch-out/".format(bucket,prefix)
batch_input = "s3://{}/{}/batch-in/{}".format(bucket,prefix,batch_X_file)

xgb_transformer = xgb_model.transformer(instance_count=1,
                                       instance_type='ml.m4.xlarge',
                                       strategy='MultiRecord',
                                       assemble_with='Line',
                                       output_path=batch_output)

xgb_transformer.transform(data=batch_input,
                         data_type='S3Prefix',
                         content_type='text/csv',
                         split_type='Line')
xgb_transformer.wait()
```

    INFO:sagemaker:Creating model with name: sagemaker-xgboost-2024-11-10-14-16-12-690
    INFO:sagemaker:Creating transform job with name: sagemaker-xgboost-2024-11-10-14-16-13-414


    ........................................[34m[2024-11-10:14:22:54:INFO] No GPUs detected (normal if no gpus installed)[0m
    [34m[2024-11-10:14:22:54:INFO] No GPUs detected (normal if no gpus installed)[0m
    [34m[2024-11-10:14:22:54:INFO] nginx config: [0m
    [34mworker_processes auto;[0m
    [34mdaemon off;[0m
    [34mpid /tmp/nginx.pid;[0m
    [34merror_log  /dev/stderr;[0m
    [34mworker_rlimit_nofile 4096;[0m
    [34mevents {
      worker_connections 2048;[0m
    [34m}[0m
    [34mhttp {
      include /etc/nginx/mime.types;
      default_type application/octet-stream;
      access_log /dev/stdout combined;
      upstream gunicorn {
        server unix:/tmp/gunicorn.sock;
      }
      server {
        listen 8080 deferred;
        client_max_body_size 0;
        keepalive_timeout 3;
        location ~ ^/(ping|invocations|execution-parameters) {
          proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
          proxy_set_header Host $http_host;
          proxy_redirect off;
          proxy_read_timeout 60s;
          proxy_pass http://gunicorn;
        }
        location / {
          return 404 "{}";
        }
      }[0m
    [34m}[0m
    [34m[2024-11-10 14:22:55 +0000] [18] [INFO] Starting gunicorn 19.10.0[0m
    [34m[2024-11-10 14:22:55 +0000] [18] [INFO] Listening at: unix:/tmp/gunicorn.sock (18)[0m
    [34m[2024-11-10 14:22:55 +0000] [18] [INFO] Using worker: gevent[0m
    [34m[2024-11-10 14:22:55 +0000] [25] [INFO] Booting worker with pid: 25[0m
    [34m[2024-11-10 14:22:55 +0000] [26] [INFO] Booting worker with pid: 26[0m
    [34m[2024-11-10 14:22:55 +0000] [27] [INFO] Booting worker with pid: 27[0m
    [34m[2024-11-10 14:22:55 +0000] [28] [INFO] Booting worker with pid: 28[0m
    [34m[2024-11-10:14:23:01:INFO] No GPUs detected (normal if no gpus installed)[0m
    [34m169.254.255.130 - - [10/Nov/2024:14:23:01 +0000] "GET /ping HTTP/1.1" 200 0 "-" "Go-http-client/1.1"[0m
    [34m169.254.255.130 - - [10/Nov/2024:14:23:01 +0000] "GET /execution-parameters HTTP/1.1" 200 84 "-" "Go-http-client/1.1"[0m
    [34m[2024-11-10:14:23:01:INFO] Determined delimiter of CSV input is ','[0m
    [34m169.254.255.130 - - [10/Nov/2024:14:23:01 +0000] "POST /invocations HTTP/1.1" 200 598 "-" "Go-http-client/1.1"[0m
    [35m[2024-11-10:14:23:01:INFO] No GPUs detected (normal if no gpus installed)[0m
    [35m169.254.255.130 - - [10/Nov/2024:14:23:01 +0000] "GET /ping HTTP/1.1" 200 0 "-" "Go-http-client/1.1"[0m
    [35m169.254.255.130 - - [10/Nov/2024:14:23:01 +0000] "GET /execution-parameters HTTP/1.1" 200 84 "-" "Go-http-client/1.1"[0m
    [35m[2024-11-10:14:23:01:INFO] Determined delimiter of CSV input is ','[0m
    [35m169.254.255.130 - - [10/Nov/2024:14:23:01 +0000] "POST /invocations HTTP/1.1" 200 598 "-" "Go-http-client/1.1"[0m
    [32m2024-11-10T14:23:01.484:[sagemaker logs]: MaxConcurrentTransforms=4, MaxPayloadInMB=6, BatchStrategy=MULTI_RECORD[0m
    
    [34m[2024-11-10:14:22:54:INFO] No GPUs detected (normal if no gpus installed)[0m
    [34m[2024-11-10:14:22:54:INFO] No GPUs detected (normal if no gpus installed)[0m
    [34m[2024-11-10:14:22:54:INFO] nginx config: [0m
    [34mworker_processes auto;[0m
    [34mdaemon off;[0m
    [35m[2024-11-10:14:22:54:INFO] No GPUs detected (normal if no gpus installed)[0m
    [35m[2024-11-10:14:22:54:INFO] No GPUs detected (normal if no gpus installed)[0m
    [35m[2024-11-10:14:22:54:INFO] nginx config: [0m
    [35mworker_processes auto;[0m
    [35mdaemon off;[0m
    [34mpid /tmp/nginx.pid;[0m
    [34merror_log  /dev/stderr;[0m
    [34mworker_rlimit_nofile 4096;[0m
    [34mevents {
      worker_connections 2048;[0m
    [34m}[0m
    [34mhttp {
      include /etc/nginx/mime.types;
      default_type application/octet-stream;
      access_log /dev/stdout combined;
      upstream gunicorn {
        server unix:/tmp/gunicorn.sock;
      }
      server {
        listen 8080 deferred;
        client_max_body_size 0;
        keepalive_timeout 3;
        location ~ ^/(ping|invocations|execution-parameters) {
          proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
          proxy_set_header Host $http_host;
          proxy_redirect off;
          proxy_read_timeout 60s;
          proxy_pass http://gunicorn;
        }
        location / {
          return 404 "{}";
        }
      }[0m
    [34m}[0m
    [34m[2024-11-10 14:22:55 +0000] [18] [INFO] Starting gunicorn 19.10.0[0m
    [34m[2024-11-10 14:22:55 +0000] [18] [INFO] Listening at: unix:/tmp/gunicorn.sock (18)[0m
    [35mpid /tmp/nginx.pid;[0m
    [35merror_log  /dev/stderr;[0m
    [35mworker_rlimit_nofile 4096;[0m
    [35mevents {
      worker_connections 2048;[0m
    [35m}[0m
    [35mhttp {
      include /etc/nginx/mime.types;
      default_type application/octet-stream;
      access_log /dev/stdout combined;
      upstream gunicorn {
        server unix:/tmp/gunicorn.sock;
      }
      server {
        listen 8080 deferred;
        client_max_body_size 0;
        keepalive_timeout 3;
        location ~ ^/(ping|invocations|execution-parameters) {
          proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
          proxy_set_header Host $http_host;
          proxy_redirect off;
          proxy_read_timeout 60s;
          proxy_pass http://gunicorn;
        }
        location / {
          return 404 "{}";
        }
      }[0m
    [35m}[0m
    [35m[2024-11-10 14:22:55 +0000] [18] [INFO] Starting gunicorn 19.10.0[0m
    [35m[2024-11-10 14:22:55 +0000] [18] [INFO] Listening at: unix:/tmp/gunicorn.sock (18)[0m
    [34m[2024-11-10 14:22:55 +0000] [18] [INFO] Using worker: gevent[0m
    [34m[2024-11-10 14:22:55 +0000] [25] [INFO] Booting worker with pid: 25[0m
    [34m[2024-11-10 14:22:55 +0000] [26] [INFO] Booting worker with pid: 26[0m
    [34m[2024-11-10 14:22:55 +0000] [27] [INFO] Booting worker with pid: 27[0m
    [34m[2024-11-10 14:22:55 +0000] [28] [INFO] Booting worker with pid: 28[0m
    [35m[2024-11-10 14:22:55 +0000] [18] [INFO] Using worker: gevent[0m
    [35m[2024-11-10 14:22:55 +0000] [25] [INFO] Booting worker with pid: 25[0m
    [35m[2024-11-10 14:22:55 +0000] [26] [INFO] Booting worker with pid: 26[0m
    [35m[2024-11-10 14:22:55 +0000] [27] [INFO] Booting worker with pid: 27[0m
    [35m[2024-11-10 14:22:55 +0000] [28] [INFO] Booting worker with pid: 28[0m
    [34m[2024-11-10:14:23:01:INFO] No GPUs detected (normal if no gpus installed)[0m
    [34m169.254.255.130 - - [10/Nov/2024:14:23:01 +0000] "GET /ping HTTP/1.1" 200 0 "-" "Go-http-client/1.1"[0m
    [34m169.254.255.130 - - [10/Nov/2024:14:23:01 +0000] "GET /execution-parameters HTTP/1.1" 200 84 "-" "Go-http-client/1.1"[0m
    [34m[2024-11-10:14:23:01:INFO] Determined delimiter of CSV input is ','[0m
    [34m169.254.255.130 - - [10/Nov/2024:14:23:01 +0000] "POST /invocations HTTP/1.1" 200 598 "-" "Go-http-client/1.1"[0m
    [35m[2024-11-10:14:23:01:INFO] No GPUs detected (normal if no gpus installed)[0m
    [35m169.254.255.130 - - [10/Nov/2024:14:23:01 +0000] "GET /ping HTTP/1.1" 200 0 "-" "Go-http-client/1.1"[0m
    [35m169.254.255.130 - - [10/Nov/2024:14:23:01 +0000] "GET /execution-parameters HTTP/1.1" 200 84 "-" "Go-http-client/1.1"[0m
    [35m[2024-11-10:14:23:01:INFO] Determined delimiter of CSV input is ','[0m
    [35m169.254.255.130 - - [10/Nov/2024:14:23:01 +0000] "POST /invocations HTTP/1.1" 200 598 "-" "Go-http-client/1.1"[0m
    [32m2024-11-10T14:23:01.484:[sagemaker logs]: MaxConcurrentTransforms=4, MaxPayloadInMB=6, BatchStrategy=MULTI_RECORD[0m


After the transform completes, you can download the results from Amazon S3 and compare them with the input.

First, download the output from Amazon S3 and load it into a pandas DataFrame.



```python
s3 = boto3.client('s3')
obj = s3.get_object(Bucket=bucket, Key="{}/batch-out/{}".format(prefix,'batch-in.csv.out'))
target_predicted = pd.read_csv(io.BytesIO(obj['Body'].read()),sep=',',names=['class'])
target_predicted.head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.996607</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.777283</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.994641</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.993690</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.939139</td>
    </tr>
  </tbody>
</table>
</div>



You can use a function to convert the probabilty into either a *0* or a *1*.

The first table output will be the *predicted values*, and the second table output is the *original test data*.


```python
def binary_convert(x):
    threshold = 0.65
    if x > threshold:
        return 1
    else:
        return 0

target_predicted['binary'] = target_predicted['class'].apply(binary_convert)

print(target_predicted.head(100))
test.head(100)
```

           class  binary
    0   0.996607       1
    1   0.777283       1
    2   0.994641       1
    3   0.993690       1
    4   0.939139       1
    5   0.997396       1
    6   0.991977       1
    7   0.987518       1
    8   0.993334       1
    9   0.682776       1
    10  0.018407       0
    11  0.002559       0
    12  0.998525       1
    13  0.988788       1
    14  0.743618       1
    15  0.983228       1
    16  0.165929       0
    17  0.997496       1
    18  0.584684       0
    19  0.995587       1
    20  0.792606       1
    21  0.004930       0
    22  0.137988       0
    23  0.998327       1
    24  0.263713       0
    25  0.998124       1
    26  0.033861       0
    27  0.180446       0
    28  0.998020       1
    29  0.002707       0
    30  0.856742       1





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>class</th>
      <th>pelvic_incidence</th>
      <th>pelvic_tilt</th>
      <th>lumbar_lordosis_angle</th>
      <th>sacral_slope</th>
      <th>pelvic_radius</th>
      <th>degree_spondylolisthesis</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>136</th>
      <td>1</td>
      <td>88.024499</td>
      <td>39.844669</td>
      <td>81.774473</td>
      <td>48.179830</td>
      <td>116.601538</td>
      <td>56.766083</td>
    </tr>
    <tr>
      <th>230</th>
      <td>0</td>
      <td>65.611802</td>
      <td>23.137919</td>
      <td>62.582179</td>
      <td>42.473883</td>
      <td>124.128001</td>
      <td>-4.083298</td>
    </tr>
    <tr>
      <th>134</th>
      <td>1</td>
      <td>52.204693</td>
      <td>17.212673</td>
      <td>78.094969</td>
      <td>34.992020</td>
      <td>136.972517</td>
      <td>54.939134</td>
    </tr>
    <tr>
      <th>130</th>
      <td>1</td>
      <td>50.066786</td>
      <td>9.120340</td>
      <td>32.168463</td>
      <td>40.946446</td>
      <td>99.712453</td>
      <td>26.766697</td>
    </tr>
    <tr>
      <th>47</th>
      <td>1</td>
      <td>41.352504</td>
      <td>16.577364</td>
      <td>30.706191</td>
      <td>24.775141</td>
      <td>113.266675</td>
      <td>-4.497958</td>
    </tr>
    <tr>
      <th>135</th>
      <td>1</td>
      <td>77.121344</td>
      <td>30.349874</td>
      <td>77.481083</td>
      <td>46.771470</td>
      <td>110.611148</td>
      <td>82.093607</td>
    </tr>
    <tr>
      <th>100</th>
      <td>1</td>
      <td>84.585607</td>
      <td>30.361685</td>
      <td>65.479486</td>
      <td>54.223922</td>
      <td>108.010218</td>
      <td>25.118478</td>
    </tr>
    <tr>
      <th>89</th>
      <td>1</td>
      <td>71.186811</td>
      <td>23.896201</td>
      <td>43.696665</td>
      <td>47.290610</td>
      <td>119.864938</td>
      <td>27.283985</td>
    </tr>
    <tr>
      <th>297</th>
      <td>0</td>
      <td>45.575482</td>
      <td>18.759135</td>
      <td>33.774143</td>
      <td>26.816347</td>
      <td>116.797007</td>
      <td>3.131910</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>49.712859</td>
      <td>9.652075</td>
      <td>28.317406</td>
      <td>40.060784</td>
      <td>108.168725</td>
      <td>7.918501</td>
    </tr>
    <tr>
      <th>281</th>
      <td>0</td>
      <td>64.261507</td>
      <td>14.497866</td>
      <td>43.902504</td>
      <td>49.763642</td>
      <td>115.388268</td>
      <td>5.951454</td>
    </tr>
    <tr>
      <th>270</th>
      <td>0</td>
      <td>51.311771</td>
      <td>8.875541</td>
      <td>57.000000</td>
      <td>42.436230</td>
      <td>126.472258</td>
      <td>-2.144044</td>
    </tr>
    <tr>
      <th>162</th>
      <td>1</td>
      <td>118.144655</td>
      <td>38.449501</td>
      <td>50.838520</td>
      <td>79.695154</td>
      <td>81.024541</td>
      <td>74.043767</td>
    </tr>
    <tr>
      <th>32</th>
      <td>1</td>
      <td>44.936675</td>
      <td>17.443838</td>
      <td>27.780576</td>
      <td>27.492837</td>
      <td>117.980324</td>
      <td>5.569620</td>
    </tr>
    <tr>
      <th>51</th>
      <td>1</td>
      <td>74.433593</td>
      <td>41.557331</td>
      <td>27.700000</td>
      <td>32.876262</td>
      <td>107.949304</td>
      <td>5.000089</td>
    </tr>
    <tr>
      <th>194</th>
      <td>1</td>
      <td>72.643850</td>
      <td>18.929117</td>
      <td>68.000000</td>
      <td>53.714733</td>
      <td>116.963416</td>
      <td>25.384247</td>
    </tr>
    <tr>
      <th>131</th>
      <td>1</td>
      <td>69.781006</td>
      <td>13.777465</td>
      <td>58.000000</td>
      <td>56.003541</td>
      <td>118.930666</td>
      <td>17.914560</td>
    </tr>
    <tr>
      <th>174</th>
      <td>1</td>
      <td>61.411737</td>
      <td>25.384364</td>
      <td>39.096869</td>
      <td>36.027373</td>
      <td>103.404597</td>
      <td>21.843407</td>
    </tr>
    <tr>
      <th>36</th>
      <td>1</td>
      <td>32.090987</td>
      <td>6.989378</td>
      <td>35.998198</td>
      <td>25.101609</td>
      <td>132.264735</td>
      <td>6.413428</td>
    </tr>
    <tr>
      <th>161</th>
      <td>1</td>
      <td>67.263149</td>
      <td>7.194661</td>
      <td>51.696887</td>
      <td>60.068488</td>
      <td>97.801085</td>
      <td>42.136943</td>
    </tr>
    <tr>
      <th>302</th>
      <td>0</td>
      <td>54.600316</td>
      <td>21.488974</td>
      <td>29.360216</td>
      <td>33.111342</td>
      <td>118.343321</td>
      <td>-1.471067</td>
    </tr>
    <tr>
      <th>294</th>
      <td>0</td>
      <td>46.236399</td>
      <td>10.062770</td>
      <td>37.000000</td>
      <td>36.173629</td>
      <td>128.063620</td>
      <td>-5.100053</td>
    </tr>
    <tr>
      <th>261</th>
      <td>0</td>
      <td>74.976021</td>
      <td>14.921705</td>
      <td>53.730072</td>
      <td>60.054317</td>
      <td>105.645400</td>
      <td>1.594748</td>
    </tr>
    <tr>
      <th>95</th>
      <td>1</td>
      <td>57.522356</td>
      <td>33.647075</td>
      <td>50.909858</td>
      <td>23.875281</td>
      <td>140.981712</td>
      <td>148.753711</td>
    </tr>
    <tr>
      <th>224</th>
      <td>0</td>
      <td>89.834676</td>
      <td>22.639217</td>
      <td>90.563461</td>
      <td>67.195460</td>
      <td>100.501192</td>
      <td>3.040973</td>
    </tr>
    <tr>
      <th>61</th>
      <td>1</td>
      <td>89.680567</td>
      <td>32.704435</td>
      <td>83.130732</td>
      <td>56.976132</td>
      <td>129.955476</td>
      <td>92.027277</td>
    </tr>
    <tr>
      <th>212</th>
      <td>0</td>
      <td>44.362490</td>
      <td>8.945435</td>
      <td>46.902096</td>
      <td>35.417055</td>
      <td>129.220682</td>
      <td>4.994195</td>
    </tr>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>63.027817</td>
      <td>22.552586</td>
      <td>39.609117</td>
      <td>40.475232</td>
      <td>98.672917</td>
      <td>-0.254400</td>
    </tr>
    <tr>
      <th>98</th>
      <td>1</td>
      <td>77.655119</td>
      <td>22.432950</td>
      <td>93.892779</td>
      <td>55.222169</td>
      <td>123.055707</td>
      <td>61.211187</td>
    </tr>
    <tr>
      <th>284</th>
      <td>0</td>
      <td>59.167612</td>
      <td>14.562749</td>
      <td>43.199158</td>
      <td>44.604863</td>
      <td>121.035642</td>
      <td>2.830504</td>
    </tr>
    <tr>
      <th>171</th>
      <td>1</td>
      <td>78.401254</td>
      <td>14.042260</td>
      <td>79.694263</td>
      <td>64.358994</td>
      <td>104.731234</td>
      <td>12.392853</td>
    </tr>
  </tbody>
</table>
</div>



**Note:** The *threshold* in the **binary_convert** function is set to *.65*.

**Challenge task:** Experiment with changing the value of the threshold. Does it impact the results?

**Note:** The initial model might not be good. You will generate some metrics in the next lab, before you tune the model in the final lab.

# Congratulations!

You have completed this lab, and you can now end the lab by following the lab guide instructions.
