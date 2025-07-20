from src.preprocessing import preprocess
from src.feature_extraction import extract_hog
import cudf, cupy as cp

def recognize_image(img, pca, scaler, svc, rf, knn):
    img_p = preprocess(img)
    hog_f = extract_hog([img_p])[0].reshape(1, -1)
    pca_f = pca.transform(hog_f)
    gdf = cudf.DataFrame(pca_f)
    gdf_s = scaler.transform(gdf)

    p1 = svc.predict_proba(gdf_s)
    p2 = rf.predict_proba(gdf_s)
    p3 = knn.predict_proba(gdf_s)

    avg = (p1 + p2 + p3) / 3.0
    return int(cp.argmax(avg.to_cupy(), axis=1).get()[0])
