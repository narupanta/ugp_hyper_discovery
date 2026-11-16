import sys, json, numpy as np
sys.path.insert(0,'.'); sys.path.insert(0,'validation')
from forward_fem_full_specimen import load_reference, solve, to_dic_nodes, grip_reaction, FullVectorMaterial, distilled_full_vectors
model, out = sys.argv[1], sys.argv[2]; ds_list=[float(a) for a in sys.argv[3:]]
m=np.load('mesh/TTf_full_mesh.npz',allow_pickle=True)
X=np.asarray(m['X'],float); cells=np.asarray(m['tris']); Xd=np.asarray(m['X_dic'],float)
A=np.linalg.lstsq(np.c_[X,np.ones(len(X))],Xd,rcond=None)[0][:2]
ds=np.load('dataset/preprocessed/experiment/ttf_20_steps_ps_incomp.npz')
xq=ds['mesh_pos'][:,:2].astype(float); uo=ds['u_exp']; steps=ds['steps']; b=ds['bcx'].ravel()
grip=100.0*steps/6000.0; L=X[:,1].max()
mat = FullVectorMaterial(distilled_full_vectors(model).mean(0)) if model.endswith('distilled') else load_reference(model)
res={}
for d in ds_list:
    yb,yt=max(d,1e-9),L-max(d,1e-9)
    held=(lambda y:(lambda p:p[1]<=y+1e-6))(yb); pull=(lambda y:(lambda p:p[1]>=y-1e-6))(yt)
    u=solve(mat,X,cells,held,pull,1,1.0,grip,f'd={d}')
    ud,_=to_dic_nodes(u,Xd,cells,xq,Amat=A)
    R=np.abs(grip_reaction(mat,X,cells,u,np.where(X[:,1]<=yb+1e-6)[0])[:,1])
    res[d]=dict(e1=ud[:,b==1,0].mean(1).tolist(),e2=ud[:,b==2,0].mean(1).tolist(),R=R.tolist(),
                err=np.linalg.norm(ud-uo,axis=-1).mean(1).tolist())
json.dump(res,open(out,'w'))
