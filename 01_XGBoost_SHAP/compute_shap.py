"""Tree SHAP contributions and pair interactions from the archived workflow."""
from common.analysis import load, shap_values, FEATURES, OUT
import numpy as np
import pandas as pd
import xgboost as xgb
if __name__ == '__main__':
    d, x, y, model = load()
    shap_values(model, x)
    # Same reference selection as the archived response/interaction calculation.
    ref=np.random.default_rng(773).choice(len(d),2048,replace=False)
    inter=model.get_booster().predict(xgb.DMatrix(x[ref]),pred_interactions=True)[:,:-1,:-1]
    strength=np.abs(inter).mean(0)
    table=pd.DataFrame([{'factor1':FEATURES[i],'factor2':FEATURES[j],'mean_abs_interaction':float(strength[i,j])} for i in range(10) for j in range(i+1,10)]).sort_values('mean_abs_interaction',ascending=False).reset_index(drop=True)
    table['rank']=np.arange(1,46)
    table.to_csv(OUT/'spatial_model_interactions_45.csv',index=False)
    np.savez_compressed(OUT/'SHAP_interactions_reference.npz',reference_ID=d.ID.iloc[ref].to_numpy(),values=inter)
