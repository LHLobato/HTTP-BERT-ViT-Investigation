from transformers import BertModel, BertTokenizer
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

def extract_features(text, model, tokenizer):
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
print("All request loaded")
print("All labels signed")

model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

print("Model Loaded")

features = []
for i in range(len(all_requests[10])):
    features.append(extract_features(all_requests[i]))
features = torch.cat(features).numpy()

print(f"The shape of a single feature: {features[0].shape}")
