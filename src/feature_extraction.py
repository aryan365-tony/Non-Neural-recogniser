from skimage.feature import hog
from sklearn.decomposition import PCA
from config import PCA_COMPONENTS, RANDOM_STATE
import numpy as np

def extract_hog(images):
    features=[]
    for im in images:
        h=hog(im, orientations=9, pixels_per_cell=(7,7), cells_per_block=(2,2),
                block_norm='L2-Hys', feature_vector=True)
        features.append(h)
    return np.stack(features)

def reduce_pca(X_train, X_test):
    pca=PCA(n_components=PCA_COMPONENTS, whiten=True, svd_solver='randomized', random_state=RANDOM_STATE)
    return pca, pca.fit_transform(X_train), pca.transform(X_test)
