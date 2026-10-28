import json
from pathlib import Path
import cv2
import numpy as np
root=Path(__file__).resolve().parent/'camera/calibration'
views=[]
for path in root.glob('*/metadata.json'):
 data=json.loads(path.read_text())
 for view in data.get('views',[]):
  if view.get('corners_found'):views.append((path.parent/view['file'],view))
for path in (root/'manual_views').glob('*.json'):
 view=json.loads(path.read_text())
 if view.get('corners_found'):views.append((path.parent/view['file'],view))
views=[(p,v) for p,v in views if v['image_size']==[1280,720]]
if len(views)<6:raise RuntimeError('Need at least six detected views')
obj=np.zeros((8*13,3),np.float32);obj[:,:2]=np.mgrid[0:8,0:13].T.reshape(-1,2)*.04
objects=[obj.copy() for _ in views]
images=[np.array(v['corners'],np.float32).reshape(-1,1,2) for _,v in views]
result=cv2.calibrateCameraExtended(objects,images,(1280,720),None,None,flags=cv2.CALIB_FIX_K3)
rms,K,dist,rvecs,tvecs,std_intr,std_ext,per_view=result
report={'status':'preliminary','image_size':[1280,720],'inner_corners':[8,13],'square_size_m':.04,'views':len(views),'rms_reprojection_px':float(rms),'camera_matrix':K.tolist(),'distortion_coefficients':dist.ravel().tolist(),'intrinsic_standard_deviations':std_intr.ravel().tolist(),'per_view':[{'file':str(p.relative_to(root)), 'rms_px':float(e)} for (p,_),e in zip(views,per_view.ravel())],'model':'OpenCV pinhole, k3 fixed to zero'}
(root/'camera_intrinsics_preliminary.json').write_text(json.dumps(report,indent=2)+'\n')
np.savez(root/'camera_intrinsics_preliminary.npz',camera_matrix=K,dist_coeffs=dist,image_size=np.array([1280,720]))
print(json.dumps(report,indent=2))
