"""Code-only portable entry point. Run --help for stages."""
import argparse,importlib.util,json,os,runpy,sys,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
ap=argparse.ArgumentParser()
ap.add_argument('--stage',required=True,choices=['model-validation','fit-fixed','shap','1d','2d','3d','plot-1d','plot-2d','plot-3d','slices','plot-3d-slices'])
ap.add_argument('--input',type=Path)
ap.add_argument('--font',type=Path)
ap.add_argument('--dpi',type=int,default=400)
args=ap.parse_args()
if args.input:
    source=args.input.resolve()
if args.font:
    os.environ['TNR_FONT']=str(args.font.resolve())
os.chdir(ROOT)
for p in ['validation/data','validation/models','validation_inputs','replotted_figures']:
    (ROOT/p).mkdir(parents=True,exist_ok=True)
if args.input:
    target=ROOT/'validation_inputs/modeling_data_12_factor_comparison.csv'
    if source!=target:
        shutil.copy2(source,target)

def execute(path):
    runpy.run_path(str(ROOT/path),run_name='__main__')

def assemble():
    import pandas as pd,numpy as np
    out=ROOT/'validation/data'
    zones=[]
    for dim in [1,2,3]:
        p=out/f'priority_zones_{dim}d.csv'
        if p.exists():
            z=pd.read_csv(p)
            if zones:
                if not np.array_equal(zones[0].ID,z.ID):
                    raise ValueError('Zone IDs/order differ across stages')
                z=z.drop(columns=['ID','x','y_coord','TOP'])
            zones.append(z)
    if zones:
        pd.concat(zones,axis=1).to_csv(out/'priority_zones_all_cells.csv',index=False)
    arrays={}
    for dim in [2,3]:
        p=out/f'joint_surfaces_{dim}d.npz'
        if p.exists():
            with np.load(p) as a:
                arrays.update({k:a[k] for k in a.files})
    np.savez_compressed(out/'joint_surfaces.npz',**arrays)

stage=args.stage
if stage=='model-validation':
    execute('01_XGBoost_SHAP/train_validate.py')
elif stage=='fit-fixed':
    from common.analysis import load,FEATURES,OUT
    import pandas as pd,numpy as np,xgboost as xgb
    d=pd.read_csv('validation_inputs/modeling_data_12_factor_comparison.csv')
    if d.ID.duplicated().any() or not np.isfinite(d[FEATURES+['GP-WY TOP']].to_numpy()).all():
        raise ValueError('Duplicate IDs or missing/non-finite model values')
    params=json.loads((ROOT/'01_XGBoost_SHAP/archived_final_parameters.json').read_text())
    model=xgb.XGBRegressor(**params)
    model.fit(d[FEATURES].to_numpy(np.float32),d['GP-WY TOP'].to_numpy())
    model.save_model(ROOT/'validation/models/final_spatial_validated.ubj')
    (OUT/'final_parameters.json').write_text(json.dumps(params,indent=2))
    print('Fixed archived configuration fitted. No new cross-validation results generated.')
elif stage=='shap':
    execute('01_XGBoost_SHAP/compute_shap.py')
elif stage in ['1d','2d','3d']:
    if stage=='1d':
        # The original response prefix reads spatial blocks even without bootstrapping.
        # fit-fixed route only: create the same nominal block IDs required by that prefix.
        p=ROOT/'validation/data/spatial_blocks.csv'
        if not p.exists():
            import numpy as np,pandas as pd
            d=pd.read_csv('validation_inputs/modeling_data_12_factor_comparison.csv')
            bx=np.floor((d.x-d.x.min())/50000).astype(int)
            by=np.floor((d.y_coord-d.y_coord.min())/50000).astype(int)
            block=pd.factorize(list(zip(bx,by)))[0]
            pd.DataFrame({'ID':d.ID,'block_50km':block}).to_csv(p,index=False)
        execute('02_1D_PDP_Priority/calculate_pdp.py')
    folder={'1d':'02_1D_PDP_Priority','2d':'03_2D_PDP_Priority','3d':'04_3D_PDP_Priority'}[stage]
    execute(folder+'/calculate_priority.py')
    assemble()
elif stage.startswith('plot-') and stage!='plot-3d-slices':
    assemble()
    from common.plot_helpers import setup
    setup(os.environ.get('TNR_FONT'),False)
    import pandas as pd,numpy as np
    d=pd.read_csv('validation_inputs/modeling_data_12_factor_comparison.csv')
    z=pd.read_csv('validation/data/priority_zones_all_cells.csv')
    if not np.array_equal(d.ID,z.ID):
        raise ValueError('Input and zoning ID order differ')
    folder={'plot-1d':'02_1D_PDP_Priority','plot-2d':'03_2D_PDP_Priority','plot-3d':'04_3D_PDP_Priority'}[stage]
    spec=importlib.util.spec_from_file_location('figure',ROOT/folder+'/plot_pdp_priority.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    with np.load('validation/data/joint_surfaces.npz') as arrays:
        module.plot(d,z,arrays,args)
elif stage=='slices':
    execute('04_3D_PDP_Priority/contour_slices.py')
elif stage=='plot-3d-slices':
    assemble()
    execute('04_3D_PDP_Priority/plot_3d_contour_slices.py')
