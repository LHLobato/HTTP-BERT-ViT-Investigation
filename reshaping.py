from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.decomposition import PCA
import numpy as np
import io
from pyts.image import GramianAngularField, RecurrencePlot
import os 
import matplotlib.pyplot as plt
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

def save_images(dataset_name, X_data, y_data, arg):
    for i, (image, label) in enumerate(zip(X_data, y_data)):
        class_name = "malicious" if label == 1 else "benign"
        file_path = os.path.join(
            base_dir, dataset_name, class_name, f"{class_name}_{i}.png"
        )
        if arg == "RPLOT":
            colors = "binary"
        else:
            colors = "rainbow"
        plt.imsave(file_path, image, cmap=colors)
    print(f"Imagens salvas em {base_dir}/{dataset_name}")

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

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(analyzer = 'char', sublinear_tf = True, lowercase=False, max_features= res*res, ngram_range=(3,3))),
    ('pca', PCA(n_components=res)),
    ('scaler', MinMaxScaler())
])

X_processed = pipeline.fit_transform(all_requests)


image_reshapes = {
    "GASF": GramianAngularField(method = "summation"),
    "GADF": GramianAngularField(method = "difference"),
    "RPLOT": RecurrencePlot(dimension=1,threshold='point', percentage=20) 
}
states = [0,100,1000]
num_datasets = 3

for i in range(num_datasets):
    for state in states:   
        for image_type, transformer in image_reshapes.items():
            X_1d_transformed = transformer.fit_transform(X_processed)
            X_train, X_temp, y_train, y_temp = train_test_split(
                X_1d_transformed, labels, test_size=0.3, stratify=labels, random_state=state
            )
            X_val, X_test, y_val, y_test = train_test_split(
                X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=state
            )
            base_dir = f"datasets/TFIDV/{str(image_type)}+{i}"
            train_dir, val_dir, test_dir = [
                os.path.join(base_dir, d) for d in ["train", "val", "test"]
            ]
            for subdir in [train_dir, val_dir, test_dir]:
                os.makedirs(os.path.join(subdir, "benign"), exist_ok=True)
                os.makedirs(os.path.join(subdir, "malicious"), exist_ok=True)
            
            datasets = {
                "train": (X_train, y_train),
                "val": (X_val, y_val),
                "test": (X_test, y_test),
            }
            for dataset_name, (X_data, y_data) in datasets.items():
                save_images(dataset_name, X_data, y_data, image_type)
            print(f"Imagens {image_type}, geradas e salvas com sucesso!")


