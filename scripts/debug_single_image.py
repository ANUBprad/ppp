"""
Debug single image — find exactly where None comes from.
"""

import cv2
import traceback
import sys
import numpy as np
from pathlib import Path


def debug_image(image_path: str):
    """Run full pipeline with detailed error catching."""
    
    print(f"{'═' * 60}")
    print(f"DEBUG: {image_path}")
    print(f"{'═' * 60}")
    
    # ── Load image ───────────────────────────────────
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Cannot load image: {image_path}")
        return
    
    h, w = image.shape[:2]
    print(f"✅ Image loaded: {w}×{h}")
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # ── Test 1: Enhanced Detection ───────────────────
    print(f"\n{'─' * 40}")
    print("TEST 1: EnhancedPupilDetector")
    print(f"{'─' * 40}")
    try:
        from pupil_tracking.enhanced_detection import (
            EnhancedPupilDetector,
        )
        det = EnhancedPupilDetector()
        result = det.detect(gray)
        if result is None:
            print("  Result: None (no detection)")
        else:
            print(f"  center: ({result.center_x}, "
                  f"{result.center_y})")
            print(f"  radius: {result.radius}")
            print(f"  semi_major: {result.semi_major}")
            print(f"  semi_minor: {result.semi_minor}")
            print(f"  confidence: {result.confidence}")
            
            # Check for None fields
            for field in ['center_x', 'center_y', 
                          'radius', 'semi_major',
                          'semi_minor', 'angle_deg',
                          'confidence']:
                val = getattr(result, field, "MISSING")
                if val is None:
                    print(f"  ⚠️  {field} is None!")
                elif val == "MISSING":
                    print(f"  ⚠️  {field} MISSING!")
                    
    except Exception as e:
        print(f"  ❌ Error: {e}")
        traceback.print_exc()
    
    # ── Test 2: Clinical Pipeline ────────────────────
    print(f"\n{'─' * 40}")
    print("TEST 2: ClinicalDetector")
    print(f"{'─' * 40}")
    try:
        from pupil_tracking.clinical_pipeline import (
            ClinicalDetector,
        )
        clinical = ClinicalDetector()
        
        # Try detect_all
        result = clinical.detect_all(image)
        print(f"  Keys: {list(result.keys())}")
        
        # Check pupil
        pc = result.get("pupil_center")
        pr = result.get("pupil_radius")
        print(f"  pupil_center: {pc}")
        print(f"  pupil_radius: {pr}")
        
        if pc is not None:
            print(f"    pc[0] type: {type(pc[0])}")
            print(f"    pc[1] type: {type(pc[1])}")
            if pc[0] is None or pc[1] is None:
                print("    ⚠️  CENTER CONTAINS None!")
                
        if pr is None:
            print("    ⚠️  RADIUS IS None!")
            
        # Check limbus
        lc = result.get("limbus_center")
        lr = result.get("limbus_radius")
        print(f"  limbus_center: {lc}")
        print(f"  limbus_radius: {lr}")
        
        if lc is not None:
            if lc[0] is None or lc[1] is None:
                print("    ⚠️  LIMBUS CENTER CONTAINS None!")
                
        # Check all numeric values
        for key, val in result.items():
            if val is None and key not in [
                    'ring_mask', 'pupil_contour',
                    'limbus_contour', 'mask',
                    'pupil_ellipse']:
                print(f"    ⚠️  {key} is None")
            elif isinstance(val, tuple):
                for i, v in enumerate(val):
                    if v is None:
                        print(f"    ⚠️  {key}[{i}] "
                              f"is None")
                              
    except Exception as e:
        print(f"  ❌ Error: {e}")
        traceback.print_exc()
    
    # ── Test 3: ML Detector ──────────────────────────
    print(f"\n{'─' * 40}")
    print("TEST 3: ML Detector")
    print(f"{'─' * 40}")
    try:
        from pupil_tracking.ml_segmentation import (
            EyeMLDetector,
        )
        ml_path = Path("./model/eye_segmentation.pth")
        if not ml_path.exists():
            ml_path = Path("./model/eye_segmentation.onnx")
        
        if ml_path.exists():
            ml_det = EyeMLDetector(str(ml_path))
            if ml_det.is_available:
                ml_result = ml_det.detect(image)
                print(f"  Grade: "
                      f"{ml_result.measurement_grade}")
                if ml_result.pupil:
                    p = ml_result.pupil
                    print(f"  Pupil: ({p.center_x:.1f}, "
                          f"{p.center_y:.1f}) "
                          f"r={p.radius:.1f}")
                else:
                    print("  Pupil: None")
                if ml_result.limbus:
                    li = ml_result.limbus
                    print(f"  Limbus: ({li.center_x:.1f},"
                          f" {li.center_y:.1f}) "
                          f"r={li.radius:.1f}")
                else:
                    print("  Limbus: None")
            else:
                print("  ML model not available")
        else:
            print(f"  No model at {ml_path}")
            
    except Exception as e:
        print(f"  ❌ Error: {e}")
        traceback.print_exc()
    
    # ── Test 4: Limbus Detector ──────────────────────
    print(f"\n{'─' * 40}")
    print("TEST 4: LimbusDetector")
    print(f"{'─' * 40}")
    try:
        from pupil_tracking.limbus_detector import (
            LimbusDetector,
        )
        limbus_det = LimbusDetector()
        
        # Need pupil first
        if pc is not None and pr is not None:
            limbus = limbus_det.detect(
                image, 
                pupil_center=pc, 
                pupil_radius=pr)
            if limbus is None:
                print("  Result: None")
            else:
                print(f"  center: ({limbus.center_x}, "
                      f"{limbus.center_y})")
                print(f"  radius: {limbus.radius}")
                print(f"  confidence: "
                      f"{limbus.confidence}")
                
                for field in ['center_x', 'center_y',
                              'radius', 'semi_major',
                              'semi_minor']:
                    val = getattr(limbus, field, None)
                    if val is None:
                        print(f"  ⚠️  {field} is None!")
        else:
            print("  Skipped (no pupil detected)")
            
    except Exception as e:
        print(f"  ❌ Error: {e}")
        traceback.print_exc()
    
    # ── Test 5: Annotation (where crash likely is) ───
    print(f"\n{'─' * 40}")
    print("TEST 5: Annotation (likely crash point)")
    print(f"{'─' * 40}")
    try:
        from pupil_tracking.annotation import (
            EyeAnnotator,
        )
        annotator = EyeAnnotator()
        
        # Build safe detection dict
        safe_result = {}
        if pc is not None and pr is not None:
            # Guard every value
            safe_pc = (
                float(pc[0]) if pc[0] is not None 
                else w / 2.0,
                float(pc[1]) if pc[1] is not None 
                else h / 2.0,
            )
            safe_pr = (
                float(pr) if pr is not None 
                else 10.0)
            
            safe_result = {
                "pupil_center": safe_pc,
                "pupil_radius": safe_pr,
                "pupil_confidence": result.get(
                    "pupil_confidence", 0.5),
                "pupil_semi_major": result.get(
                    "pupil_semi_major") or safe_pr,
                "pupil_semi_minor": result.get(
                    "pupil_semi_minor") or safe_pr,
                "pupil_angle": result.get(
                    "pupil_angle", 0.0) or 0.0,
            }
            
            if lc is not None and lr is not None:
                safe_lc = (
                    float(lc[0]) if lc[0] is not None
                    else safe_pc[0],
                    float(lc[1]) if lc[1] is not None
                    else safe_pc[1],
                )
                safe_lr = (
                    float(lr) if lr is not None 
                    else safe_pr * 3)
                
                safe_result.update({
                    "limbus_center": safe_lc,
                    "limbus_radius": safe_lr,
                    "limbus_confidence": result.get(
                        "limbus_confidence", 0.5),
                    "limbus_semi_major": result.get(
                        "limbus_semi_major") or safe_lr,
                    "limbus_semi_minor": result.get(
                        "limbus_semi_minor") or safe_lr,
                    "limbus_angle": result.get(
                        "limbus_angle", 0.0) or 0.0,
                })
        
        annotated = annotator.annotate(
            image.copy(), safe_result)
        print(f"  ✅ Annotation succeeded!")
        print(f"    Output shape: {annotated.shape}")
        
        # Save for inspection
        out = Path("debug_annotated.png")
        cv2.imwrite(str(out), annotated)
        print(f"    Saved: {out}")
        
    except Exception as e:
        print(f"  ❌ CRASH HERE: {e}")
        traceback.print_exc()
    
    # ── Test 6: Full Static Analyzer ─────────────────
    print(f"\n{'─' * 40}")
    print("TEST 6: Full StaticImageAnalyzer")
    print(f"{'─' * 40}")
    try:
        from pupil_tracking.static_analyzer import (
            StaticImageAnalyzer,
        )
        analyzer = StaticImageAnalyzer()
        analysis = analyzer.analyze(image_path)
        
        print(f"  pupil_detected: "
              f"{analysis.pupil_detected}")
        print(f"  pupil_center: "
              f"{analysis.pupil_center}")
        print(f"  pupil_radius: "
              f"{analysis.pupil_radius}")
        print(f"  limbus_detected: "
              f"{analysis.limbus_detected}")
        print(f"  limbus_center: "
              f"{analysis.limbus_center}")
        print(f"  limbus_radius: "
              f"{analysis.limbus_radius}")
        
        # Check for None in all fields
        for attr in ['pupil_center', 'pupil_radius',
                      'pupil_semi_major', 
                      'pupil_semi_minor',
                      'limbus_center', 'limbus_radius',
                      'limbus_semi_major',
                      'limbus_semi_minor',
                      'corneal_center', 'offset_mm']:
            val = getattr(analysis, attr, "MISSING")
            if val is None:
                print(f"  ⚠️  {attr} is None")
            elif isinstance(val, tuple):
                for i, v in enumerate(val):
                    if v is None:
                        print(f"  ⚠️  {attr}[{i}] "
                              f"is None!")
                              
        print(f"  ✅ Analysis completed")
        
    except Exception as e:
        print(f"  ❌ CRASH: {e}")
        traceback.print_exc()
    
    print(f"\n{'═' * 60}")
    print("DIAGNOSTIC COMPLETE")
    print(f"{'═' * 60}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python debug_single_image.py "
              "<image_path>")
        print("Example: python debug_single_image.py "
              "clinical_data/annotations/eye_01.jpeg")
        sys.exit(1)
    
    debug_image(sys.argv[1])