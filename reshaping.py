from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import io

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

bad_requests = loadData('PreProcessedAnomalous.txt')
good_requests = loadData('PreprocessedNormalTraining.txt')

all_requests = bad_requests + good_requests
labels_Bad = [1] * len(bad_requests)
labels_Good = [0] * len(good_requests)
labels = labels_Bad + labels_Good

print ("Total requests : ",len(all_requests))
print ("Bad requests: ",len(bad_requests))
print ("Good requests: ",len(good_requests))


param_grid = {
    'tfidf__max_features': [32, 64, 128, 224],  # Valores típicos para imagens 224x224
    'tfidf__ngram_range': [(3, 3), (3, 5)],       # Testar diferentes tamanhos de n-grams
}


pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(analyzer="char", sublinear_tf=True,lowercase=False)),
    ('classifier', RandomForestClassifier(n_estimators=200))  # Modelo leve para avaliação
])
#X = vectorizer.fit_transform(all_requests)
pipeline.set_params(classifier__class_weight='balanced')
#for state in [0,42,100,1000]:
    #X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.3, random_state=state)
grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,                # Validação cruzada de 3 folds
    scoring='roc_auc',  # Métrica de avaliação
    n_jobs=-1,           # Usar todos os núcleos do CPU
    verbose=2
)
n_samples = 1000
X_train, X_test, y_train, y_test = train_test_split(
    all_requests, labels, test_size=0.2, random_state=42
)
grid_search.fit(X_train[:n_samples], y_train[:n_samples])
print("Melhores parâmetros (AUC):", grid_search.best_params_)
print("AUC do melhor modelo:", grid_search.best_score_)


best_model = grid_search.best_estimator_
y_probs = best_model.predict_proba(X_test)[:, 1]  # Probabilidades da classe positiva
test_auc = roc_auc_score(y_test, y_probs)
print("\nAUC no conjunto de teste:", test_auc)
