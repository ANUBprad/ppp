from pupil_tracking.interface.gui_app import PupilTrackingGUI


def test_optimized_gui_preserves_per_structure_confidences():
    frame = PupilTrackingGUI._dict_to_frame_ns(
        {
            "pupil_detected": True,
            "pupil_x": 40.0,
            "pupil_y": 50.0,
            "pupil_radius": 10.0,
            "pupil_confidence": 0.72,
            "limbus_detected": True,
            "limbus_x": 42.0,
            "limbus_y": 52.0,
            "limbus_radius": 30.0,
            "limbus_confidence": 0.61,
            "overall_confidence": 0.67,
            "processing_time_ms": 12.0,
        }
    )
    app = object.__new__(PupilTrackingGUI)

    result = app._adapt_frame_result(frame, (100, 100, 3))
    exported = app._frame_result_to_export_dict(frame, result.calibration, result)

    assert result.pupil.confidence == 0.72
    assert result.limbus.confidence == 0.61
    assert exported["pupil"]["confidence"] == 0.72
    assert exported["limbus"]["confidence"] == 0.61
