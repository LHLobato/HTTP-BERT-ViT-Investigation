from transformers import BertModel, BertTokenizer
import torch
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def loadData(file):
    with open(file, 'r', encoding="utf8") as f:
        data = f.readlines()
    result = []
    for d in data:
        d = d.strip()
        if (len(d) > 0):
            result.append(d)
    return result

def extract_features(text, model, tokenizer, device='cuda'):
    model = model.to(device)
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)
    
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
        hidden_states = outputs.hidden_states  # Todas as camadas
    
    # Combinação vetorizada das últimas 4 camadas
    last_four_layers = torch.stack(hidden_states[-4:], dim=0)
    token_embeddings = torch.mean(last_four_layers, dim=0)  # Média das camadas
    
    # Pooling com máscara de atenção (ignora padding)
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    features = sum_embeddings / sum_mask
    
    return features

bad_requests = loadData('PreProcessedAnomalous.txt')
good_requests = loadData('PreprocessedNormalTraining.txt')

all_requests = bad_requests + good_requests
labels_Bad = [1] * len(bad_requests)
labels_Good = [0] * len(good_requests)
labels = labels_Bad + labels_Good

print("All request loaded")
print("All labels signed")

model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
print("Model Loaded")

features = []
for i in range(len(all_requests)):
    features.append(extract_features(all_requests[i],model, tokenizer))
features = torch.cat(features).to('cpu').numpy()
print("Shape of unique Feature", features[0].shape)
X_train, X_test, y_train, y_test = train_test_split(
    features, labels, test_size=0.2, random_state=42
)

res = 27

clf = RandomForestClassifier(n_estimators=200)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
y_probs = clf.predict_proba(X_test)[:, 1]  # Probabilidades da classe positiva
test_auc = roc_auc_score(y_test, y_probs)
print("-------FOR BERT TOKENIZER-----------")
print("AUC no conjunto de teste:", test_auc)
print("Acurácia: ", accuracy_score(y_test, y_pred))
print(f"Desempenho atingido com resolução: {res}")

