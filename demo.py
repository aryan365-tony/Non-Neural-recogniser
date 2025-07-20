import numpy as np
import matplotlib.pyplot as plt
from src.inference import recognize_image
from src import feature_extraction, preprocessing, data_loader, train

# Load + preprocess
_, _, X_test, y_test = data_loader.load_emnist_digits()
X_test_p = preprocessing.batch_preprocess(X_test)

# Feature + PCA
H_test = feature_extraction.extract_hog(X_test_p)
pca, _, H_test_pca = feature_extraction.reduce_pca(H_test, H_test)
scaler, svc, rf, knn = train.train_models(H_test_pca, y_test)

# Demo
idx = np.random.randint(0, len(X_test_p))
img = X_test_p[idx]
plt.imshow(img, cmap='gray')
plt.axis('off')
pred = recognize_image(img, pca, scaler, svc, rf, knn)
plt.title(f"True: {y_test[idx]} | Pred: {pred}")
plt.show()
