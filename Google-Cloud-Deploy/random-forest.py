from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest
from sklearn.pipeline import Pipeline

iris = datasets.load_iris()
pipeline = Pipeline([
      ('feature_selection', SelectKBest(chi2, k=2)),
      ('classification', RandomForestClassifier())
    ])
pipeline.fit(iris.data, iris.target)

joblib.dump(pipeline, '/home/rubensvectomobile_gmail_com/Deploy/model.joblib')
