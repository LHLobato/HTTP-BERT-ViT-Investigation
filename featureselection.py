from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import io
from sklearn.pipeline import Pipeline
from transformers import BertModel, BertTokenizer
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler 
import torch

def loadData(file):
    with open(file, 'r', encoding="utf8") as f:
        data = f.readlines()
    result = []
    for d in data:
        d = d.strip()
        if (len(d) > 0):
            result.append(d)
    return result

bad_requests = loadData('PreProcessedAnomalous.txt')
good_requests = loadData('PreprocessedNormalTraining.txt')
all_requests = bad_requests + good_requests

labels_Bad = [1] * len(bad_requests)
labels_Good = [0] * len(good_requests)
labels = labels_Bad + labels_Good
print ("Total requests : ",len(all_requests))
print ("Bad requests: ",len(bad_requests))
print ("Good requests: ",len(good_requests))
res = 64
vectorizer = TfidfVectorizer(analyzer="char", sublinear_tf=True,lowercase=False, max_features = res*res)
X = vectorizer.fit_transform(all_requests)

X_train, X_test, y_train, y_test = train_test_split(
    X, labels, test_size=0.2, random_state=42
)


classifier = RandomForestClassifier(n_estimators=200)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
y_probs = classifier.predict_proba(X_test)[:, 1]  # Probabilidades da classe positiva
test_auc = roc_auc_score(y_test, y_probs)
print("-------FOR TFIDVECTORIZER-----------")
print("AUC no conjunto de teste:", test_auc)
print("Acurácia: ", accuracy_score(y_test, y_pred))
print(f"Desempenho atingido com resolução: {res}")

