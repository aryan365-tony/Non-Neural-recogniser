from src import gpu_check, data_loader, preprocessing, feature_extraction, train, ensemble
import numpy as np

gpu_check.check_gpu()

X_train, y_train, X_test, y_test = data_loader.load_emnist_digits()
X_train_p = preprocessing.batch_preprocess(X_train)
X_test_p  = preprocessing.batch_preprocess(X_test)

H_train = feature_extraction.extract_hog(X_train_p)
H_test  = feature_extraction.extract_hog(X_test_p)

pca, H_train_pca, H_test_pca = feature_extraction.reduce_pca(H_train, H_test)

scaler, svc, rf, knn = train.train_models(H_train_pca, y_train)

y_pred = ensemble.soft_vote_ensemble((svc, rf, knn), scaler, H_test_pca)
acc = (y_pred == y_test).mean()
print(f"✅ Ensemble Accuracy: {acc*100:.2f}%")
