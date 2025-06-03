from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.decomposition import PCA
import numpy as np
import io
from pyts.image import GramianAngularField, RecurrencePlot
from transformers import BertModel, BertTokenizer
import torch

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler 
def loadData(file):
    with open(file, 'r', encoding="utf8") as f:
        data = f.readlines()
    result = []
    for d in data:
        d = d.strip()
        if (len(d) > 0):
            result.append(d)
    return result

def extract_features(text):t
    input_ids = torch.tensor([tokenizer.encode(text, add_special_tokens=True)])
    with torch.no_grad():
        outputs = model(input_ids)
        hidden_states = outputs[2]
    token_vecs = []
    for layer in range(-4, 0):
        token_vecs.append(hidden_states[layer][0])
    features = []
    for token in token_vecs:
        features.append(torch.mean(token, dim=0))
    return torch.stack(features)

bad_requests = loadData('PreProcessedAnomalous.txt')
good_requests = loadData('PreprocessedNormalTraining.txt')

all_requests = bad_requests + good_requests
labels_Bad = [1] * len(bad_requests)
labels_Good = [0] * len(good_requests)
labels = labels_Bad + labels_Good

print ("Total requests : ",len(all_requests))
print ("Bad requests: ",len(bad_requests))
print ("Good requests: ",len(good_requests))

vectorizer = TfidfVectorizer(analyzer="char", sublinear_tf=True,lowercase=False, ngram_range=(3,3))
X = vectorizer.fit_transform(all_requests).toarray()
pca = PCA(n_components=1)
X_1d = pca.fit_transform(X)


#model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)
#tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
