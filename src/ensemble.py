import cupy as cp

def soft_vote_ensemble(models, scaler, X_test):
    svc, rf, knn=models
    gX=cudf.DataFrame(X_test)
    gX_s=scaler.transform(gX)

    p1=svc.predict_proba(gX_s)
    p2=rf.predict_proba(gX_s)
    p3=knn.predict_proba(gX_s)

    avg=(p1+p2+p3)/3.0
    return cp.argmax(avg.to_cupy(), axis=1).get()
