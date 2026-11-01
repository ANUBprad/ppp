import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2, json, math
from pathlib import Path
from pupil_tracking.core.detector import UnifiedDetector
from pupil_tracking.interface.gui_app import PupilTrackingGUI

detector = UnifiedDetector()
img_dir = Path('clinical_data/clean')
with open(img_dir / 'annotations' / 'annotations.json') as f:
    gt_data = json.load(f)

for img_name in sorted(gt_data.keys()):
    img_file = img_dir / img_name
    if not img_file.is_file():
        continue
    img = cv2.imread(str(img_file))
    h, w = img.shape[:2]
    rw, rh = min(float(w), 847.0), min(float(h), 542.0)
    ix0 = int(round((float(w) - rw) / 2.0))
    iy0 = int(round((float(h) - rh) / 2.0))
    ix1, iy1 = ix0 + int(rw), iy0 + int(rh)
    
    # Try full first
    res = detector.detect(img, frame_number=-1, source=f'pentacam:{img_file}')
    rel = PupilTrackingGUI._pentacam_detection_reliable(res)
    used = 'FULL'
    
    # If not reliable or center not within ROI, fallback to crop
    cx = getattr(getattr(res, 'pupil', None), 'ellipse', None)
    if not rel or cx is None or not (ix0 <= cx.center_x <= ix1 and iy0 <= cx.center_y <= iy1):
        crop = img[iy0:iy1, ix0:ix1]
        res_crop = detector.detect(crop, frame_number=-1, force_mode='pre_docked', source=f'pentacam:{img_file}')
        res = PupilTrackingGUI._offset_detection_result(res_crop, float(ix0), float(iy0))
        rel = PupilTrackingGUI._pentacam_detection_reliable(res)
        used = 'CROP'
    
    ann = gt_data[img_name].get('annotations', {})
    gt_p = ann.get('PUPIL')
    gt_l = ann.get('LIMBUS')
    
    pe = getattr(res.pupil, 'ellipse', None)
    le = getattr(res.limbus, 'ellipse', None)
    p_err = f"{math.hypot(pe.center_x - gt_p['center_x'], pe.center_y - gt_p['center_y']):.1f}px" if pe and gt_p else 'N/A'
    l_err = f"{math.hypot(le.center_x - gt_l['center_x'], le.center_y - gt_l['center_y']):.1f}px" if le and gt_l else 'N/A'
    
    print(f"{img_name:12s} ({used:4s}): P_ok={str(res.pupil.detected):5s} P_err={p_err:7s} | L_ok={str(res.limbus.detected):5s} L_err={l_err:7s} | rel={str(rel):5s}")
