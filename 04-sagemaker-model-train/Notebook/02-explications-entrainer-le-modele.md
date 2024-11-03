```python
# Importer les bibliothèques nécessaires
import warnings, requests, zipfile, io
import pandas as pd
from scipy.io import arff
import boto3

warnings.simplefilter('ignore')

# Télécharger et extraire le fichier de données
f_zip = 'http://archive.ics.uci.edu/ml/machine-learning-databases/00212/vertebral_column_data.zip'
r = requests.get(f_zip, stream=True)
Vertebral_zip = zipfile.ZipFile(io.BytesIO(r.content))
Vertebral_zip.extractall()

# Charger le fichier .arff et le convertir en DataFrame Pandas
data = arff.loadarff('column_2C_weka.arff')
df = pd.DataFrame(data[0])

# Mapper les valeurs de la classe en valeurs numériques
class_mapper = {b'Abnormal': 1, b'Normal': 0}
df['class'] = df['class'].replace(class_mapper)

# Dimensions du jeu de données
df.shape

# Afficher les noms des colonnes
df.columns

# Déplacer la colonne cible en première position
cols = df.columns.tolist()
cols = cols[-1:] + cols[:-1]
df = df[cols]

# Vérifier le nouvel ordre des colonnes
df.columns

# Diviser les données en ensembles d'entraînement et de validation/test
from sklearn.model_selection import train_test_split
train, test_and_validate = train_test_split(df, test_size=0.2, random_state=42, stratify=df['class'])

# Diviser l'ensemble de validation/test en deux
test, validate = train_test_split(test_and_validate, test_size=0.5, random_state=42, stratify=test_and_validate['class'])

# Vérifier les dimensions de chaque ensemble
print(train.shape)
print(test.shape)
print(validate.shape)

# Compter les occurrences de chaque classe dans chaque ensemble
print(train['class'].value_counts())
print(test['class'].value_counts())
print(validate['class'].value_counts())

# Définir les paramètres S3
bucket = 'c124417a3052642l8225608t1w975050101910-labbucket-qgynmzp5xcwe'
prefix = 'lab3'
train_file = 'vertebral_train.csv'
test_file = 'vertebral_test.csv'
validate_file = 'vertebral_validate.csv'

# Initialiser la ressource S3
s3_resource = boto3.Session().resource('s3')

# Fonction pour télécharger les fichiers CSV vers S3
def upload_s3_csv(filename, folder, dataframe):
    csv_buffer = io.StringIO()
    dataframe.to_csv(csv_buffer, header=False, index=False)
    s3_resource.Bucket(bucket).Object(os.path.join(prefix, folder, filename)).put(Body=csv_buffer.getvalue())

# Télécharger les ensembles de données vers S3
upload_s3_csv(train_file, 'train', train)
upload_s3_csv(test_file, 'test', test)
upload_s3_csv(validate_file, 'validate', validate)

# Obtenir l'URI du conteneur XGBoost
from sagemaker.image_uris import retrieve
container = retrieve('xgboost', boto3.Session().region_name, '1.0-1')

# Définir les hyperparamètres pour l'entraînement
hyperparams = {
    "num_round": "42",
    "eval_metric": "auc",
    "objective": "binary:logistic"
}

# Définir l'emplacement de sortie des modèles dans S3
import sagemaker
s3_output_location = "s3://{}/{}/output/".format(bucket, prefix)

# Configurer l'estimateur XGBoost
xgb_model = sagemaker.estimator.Estimator(
    container,
    sagemaker.get_execution_role(),
    instance_count=1,
    instance_type='ml.m4.xlarge',
    output_path=s3_output_location,
    hyperparameters=hyperparams,
    sagemaker_session=sagemaker.Session()
)

# Définir les canaux d'entraînement et de validation
train_channel = sagemaker.inputs.TrainingInput(
    "s3://{}/{}/train/".format(bucket, prefix, train_file),
    content_type='text/csv'
)

validate_channel = sagemaker.inputs.TrainingInput(
    "s3://{}/{}/validate/".format(bucket, prefix, validate_file),
    content_type='text/csv'
)

data_channels = {'train': train_channel, 'validation': validate_channel}

# Lancer l'entraînement du modèle
xgb_model.fit(inputs=data_channels, logs=False)
```
