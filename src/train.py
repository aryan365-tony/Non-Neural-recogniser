import cudf
from cuml.preprocessing import StandardScaler
from cuml.svm import SVC
from cuml.ensemble import RandomForestClassifier as cuRF
from cuml.neighbors import KNeighborsClassifier as cuKNN

def train_models(H_train_pca, y_train):
    gdf_X = cudf.DataFrame(H_train_pca)
    gdf_y = cudf.Series(y_train)

    scaler = StandardScaler()
    gX_s = scaler.fit_transform(gdf_X)

    svc = SVC(kernel='rbf', C=10, probability=True)
    rf  = cuRF(n_estimators=100, max_depth=16)
    knn = cuKNN(n_neighbors=5)

    svc.fit(gX_s, gdf_y)
    rf.fit(gX_s, gdf_y)
    knn.fit(gX_s, gdf_y)

    return scaler, svc, rf, knn
