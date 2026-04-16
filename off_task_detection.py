import importlib
import json
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from tracker_viz_utils import draw_tracker_history


CONFIG_PATH = Path(__file__).with_name("config_off_task_det.json")

# 얼굴 기준 랜드마크
FACE_NOSE_IDX = 1
FACE_LEFT_EYE_OUTER_IDX = 33
FACE_RIGHT_EYE_OUTER_IDX = 263
MOUTH_LEFT_IDX = 61
MOUTH_RIGHT_IDX = 291
MOUTH_UPPER_IDX = 13
MOUTH_LOWER_IDX = 14


def load_mediapipe_solutions():
    """Load the legacy MediaPipe Solutions API used by this script."""
    try:
        from mediapipe import solutions as mp_solutions
        return mp_solutions
    except ImportError:
        try:
            from mediapipe.python import solutions as mp_solutions
            return mp_solutions
        except ImportError as exc:
            raise ImportError(
                "This script uses MediaPipe Solutions (Holistic/Face Mesh/Pose), but the installed "
                "'mediapipe' package does not expose that API. Reinstall the pip wheel version, for "
                "example: 'pip uninstall mediapipe -y' then 'pip install mediapipe==0.10.14'."
            ) from exc


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_runtime_state(cfg, fps):
    return {
        "tracker_out_counter": 0,
        "face_missing_counter": 0,
        "no_hand_counter": 0,
        "smile_talk_counter": 0,
        "yaw_out_counter": 0,
        "smile_talk_events": deque(),
        "talk_values": deque(),
        "study_started": False,
        "fps": fps,
        "frame_index": 0,
        "last_ts": time.perf_counter(),
        "active_thresholds": dict(cfg["thresholds"]),
        "last_phone_detection": {
            "detected": False,
            "boxes": [],
            "available": False,
        },
        "calibration": {
            "enabled": cfg.get("calibration", {}).get("enabled", False),
            "duration_seconds": cfg.get("calibration", {}).get("duration_seconds", 3.0),
            "min_samples": cfg.get("calibration", {}).get("min_samples", 20),
            "started": False,
            "done": False,
            "start_ts": 0.0,
            "center_x_samples": [],
        },
		"tracker": {
			"kalman": None,
			"initialized": False,
			"lost_frames": 0,
			"tracked_center": None,
			"tracked_area": 0.0,
			"size_ema": 0.0,
			"history": [],  # (center, is_out)
			"history_maxlen": 50,
		},
    }



def get_face_head_yaw(face_landmarks):
    """Signed yaw proxy from face mesh: +right, -left in mirrored screen coordinates."""
    nose = face_landmarks.landmark[FACE_NOSE_IDX]
    left_eye_outer = face_landmarks.landmark[FACE_LEFT_EYE_OUTER_IDX]
    right_eye_outer = face_landmarks.landmark[FACE_RIGHT_EYE_OUTER_IDX]

    eye_center_x = (left_eye_outer.x + right_eye_outer.x) / 2.0
    eye_width = abs(right_eye_outer.x - left_eye_outer.x)
    if eye_width < 1e-6:
        return 0.0
    return (nose.x - eye_center_x) / eye_width


def check_hands_on_desk(pose_landmarks, mp_holistic, desk_y_threshold):
    if not pose_landmarks:
        return False
    left_wrist = pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_WRIST]
    right_wrist = pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_WRIST]
    return (left_wrist.y > desk_y_threshold) or (right_wrist.y > desk_y_threshold)


def has_any_visible_hand(mp_results, pose_landmarks, mp_holistic, min_visibility=0.35):
    if mp_results.left_hand_landmarks or mp_results.right_hand_landmarks:
        return True
    if not pose_landmarks:
        return False

    left_wrist = pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_WRIST]
    right_wrist = pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_WRIST]
    return (left_wrist.visibility > min_visibility) or (right_wrist.visibility > min_visibility)


def _point_box_distance(px, py, box):
    x1, y1, x2, y2 = box[:4]
    dx = max(x1 - px, 0, px - x2)
    dy = max(y1 - py, 0, py - y2)
    return float(np.hypot(dx, dy))


def is_object_held_by_hand(
    boxes,
    mp_results,
    pose_landmarks,
    mp_holistic,
    frame_shape,
    max_distance_ratio=0.05,
    lower_region_threshold=0.8,
):
    if not boxes:
        return False

    h, w = frame_shape[:2]
    hand_points = []

    for hand in [mp_results.left_hand_landmarks, mp_results.right_hand_landmarks]:
        if hand:
            for lm in hand.landmark:
                hand_points.append((lm.x * w, lm.y * h))

    if pose_landmarks:
        for idx in [mp_holistic.PoseLandmark.LEFT_WRIST, mp_holistic.PoseLandmark.RIGHT_WRIST]:
            wrist = pose_landmarks.landmark[idx]
            if wrist.visibility > 0.3:
                hand_points.append((wrist.x * w, wrist.y * h))

    if not hand_points:
        # If object is very low in frame, assume hand may be outside view and keep "not held".
        for box in boxes:
            y2_norm = float(box[3]) / max(float(h), 1.0)
            if y2_norm >= lower_region_threshold:
                return False
        return False

    max_distance = max(w, h) * max_distance_ratio
    for box in boxes:
        for px, py in hand_points:
            if _point_box_distance(px, py, box) <= max_distance:
                return True
    return False


def estimate_smile_talk_features(face_landmarks):
    if not face_landmarks:
        return 0.0, 0.0

    lm = face_landmarks.landmark
    mouth_w = abs(lm[MOUTH_RIGHT_IDX].x - lm[MOUTH_LEFT_IDX].x)
    mouth_h = abs(lm[MOUTH_LOWER_IDX].y - lm[MOUTH_UPPER_IDX].y)
    eye_w = abs(lm[FACE_RIGHT_EYE_OUTER_IDX].x - lm[FACE_LEFT_EYE_OUTER_IDX].x)

    mouth_h_safe = max(mouth_h, 1e-6)
    eye_w_safe = max(eye_w, 1e-6)
    smile_ratio = mouth_w / mouth_h_safe
    mouth_open_ratio = mouth_h / eye_w_safe
    return float(smile_ratio), float(mouth_open_ratio)


def extract_face_measurement(face_landmarks, pose_landmarks=None, mp_holistic=None):
    # 1순위: face mesh
    if face_landmarks:
        xs = [lm.x for lm in face_landmarks.landmark]
        ys = [lm.y for lm in face_landmarks.landmark]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        width = max(1e-6, max_x - min_x)
        height = max(1e-6, max_y - min_y)
        area = width * height
        cx = 0.5 * (min_x + max_x)
        cy = 0.5 * (min_y + max_y)
        return {
            "center": np.array([cx, cy], dtype=np.float32),
            "bbox": (min_x, min_y, max_x, max_y),
            "area": float(area),
        }
    # 2순위: pose landmarks에서 얼굴 중심 추정 (코, 양 눈, 양 귀)
    if pose_landmarks is not None and mp_holistic is not None:
        idxs = []
        for name in ["NOSE", "LEFT_EYE", "RIGHT_EYE", "LEFT_EAR", "RIGHT_EAR"]:
            idx = getattr(mp_holistic.PoseLandmark, name, None)
            if idx is not None:
                idxs.append(idx)
        points = []
        for idx in idxs:
            lm = pose_landmarks.landmark[idx]
            # Tracking fallback from pose head landmarks only when confidence is strong.
            if lm.visibility >= 0.6:
                points.append((lm.x, lm.y))
        if len(points) < 2:
            return None
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        width = max(1e-6, max_x - min_x)
        height = max(1e-6, max_y - min_y)
        area = width * height
        cx = 0.5 * (min_x + max_x)
        cy = 0.5 * (min_y + max_y)
        return {
            "center": np.array([cx, cy], dtype=np.float32),
            "bbox": (min_x, min_y, max_x, max_y),
            "area": float(area),
        }
    return None


def make_kalman_filter(dt):
    kf = cv2.KalmanFilter(4, 2)
    kf.measurementMatrix = np.array(
        [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
        dtype=np.float32,
    )
    kf.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-3
    kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1e-2
    kf.errorCovPost = np.eye(4, dtype=np.float32)
    update_kalman_dt(kf, dt)
    return kf


def update_kalman_dt(kf, dt):
    kf.transitionMatrix = np.array(
        [[1.0, 0.0, dt, 0.0], [0.0, 1.0, 0.0, dt], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
        dtype=np.float32,
    )


def limit_measurement_speed(prev_center, measured_center, dt, max_speed_per_second):
    if prev_center is None or dt <= 1e-6:
        return measured_center

    displacement = measured_center - prev_center
    dist_norm = float(np.linalg.norm(displacement))
    max_dist = max_speed_per_second * dt
    if dist_norm <= max_dist or dist_norm <= 1e-9:
        return measured_center

    return prev_center + (displacement / dist_norm) * max_dist


def reset_tracker_with_measurement(runtime, measurement, dt):
    """Drop the old track and start a fresh track from the latest face measurement."""
    tracker = runtime["tracker"]
    kf = make_kalman_filter(max(dt, 1.0 / max(runtime["fps"], 1.0)))
    cx, cy = measurement["center"]
    kf.statePost = np.array([[cx], [cy], [0.0], [0.0]], dtype=np.float32)

    tracker["kalman"] = kf
    tracker["initialized"] = True
    tracker["tracked_center"] = np.array([cx, cy], dtype=np.float32)
    tracker["tracked_area"] = measurement["area"]
    tracker["size_ema"] = measurement["area"]
    tracker["lost_frames"] = 0


def update_face_tracker(runtime, measurement, dt, cfg):
    tracker_cfg = cfg.get("tracking", {})
    tracker = runtime["tracker"]

    max_speed_per_second = tracker_cfg.get("max_face_speed_screen_per_second", 2.5)
    max_match_distance = tracker_cfg.get("max_match_distance_norm", 0.25)
    min_area_ratio = tracker_cfg.get("min_area_ratio", 0.6)
    max_area_ratio = tracker_cfg.get("max_area_ratio", 1.7)
    size_ema_alpha = tracker_cfg.get("size_ema_alpha", 0.2)

    if not tracker["initialized"]:
        if measurement is None:
            return None
        reset_tracker_with_measurement(runtime, measurement, dt)
        return {
            "center": tracker["tracked_center"],
            "area": tracker["tracked_area"],
            "matched": True,
            "valid_detection": True,
        }

    kf = tracker["kalman"]
    update_kalman_dt(kf, max(dt, 1e-3))
    pred = kf.predict()
    pred_center = np.array([float(pred[0, 0]), float(pred[1, 0])], dtype=np.float32)

    matched = False
    valid_detection = measurement is not None

    if measurement is not None:
        meas_center = measurement["center"]
        meas_area = measurement["area"]

        size_ref = max(tracker["size_ema"], 1e-6)
        area_ratio = meas_area / size_ref
        dist_to_pred = float(np.linalg.norm(meas_center - pred_center))

        area_ok = (min_area_ratio <= area_ratio <= max_area_ratio)
        distance_ok = dist_to_pred <= max_match_distance
        matched = area_ok and distance_ok

        if matched:
            capped = limit_measurement_speed(
                tracker["tracked_center"],
                meas_center,
                dt,
                max_speed_per_second,
            )
            kf.correct(capped.reshape(2, 1).astype(np.float32))
            post = kf.statePost
            tracker["tracked_center"] = np.array([float(post[0, 0]), float(post[1, 0])], dtype=np.float32)
            tracker["tracked_area"] = meas_area
            tracker["size_ema"] = (1.0 - size_ema_alpha) * tracker["size_ema"] + size_ema_alpha * meas_area
            tracker["lost_frames"] = 0
        else:
            # New/other face detected: discard old track and start tracking the newly observed face.
            reset_tracker_with_measurement(runtime, measurement, dt)
            matched = True
    else:
        tracker["tracked_center"] = pred_center
        tracker["lost_frames"] += 1

    return {
        "center": tracker["tracked_center"],
        "area": tracker["tracked_area"],
        "matched": matched,
        "valid_detection": valid_detection,
    }


def compute_tracker_out_of_screen(tracker_result):
    if tracker_result is None:
        return False

    center = tracker_result["center"]
    x, y = float(center[0]), float(center[1])
    return x < 0.0 or x > 1.0 or y < 0.0 or y > 1.0


def maybe_update_calibration(runtime, status, tracker_result):
    calib = runtime["calibration"]
    if not calib["enabled"] or calib["done"]:
        return

    now_ts = time.perf_counter()
    if not calib["started"]:
        calib["started"] = True
        calib["start_ts"] = now_ts

    if tracker_result is not None:
        calib["center_x_samples"].append(float(tracker_result["center"][0]))

    elapsed = now_ts - calib["start_ts"]
    if elapsed < calib["duration_seconds"]:
        return

    sample_count = len(calib["center_x_samples"])
    if sample_count < calib["min_samples"]:
        return

    calib["done"] = True


def load_tflite_interpreter_cls():
    candidates = [
        "tflite_runtime.interpreter",
        "tensorflow.lite.python.interpreter",
    ]
    for module_name in candidates:
        try:
            module = importlib.import_module(module_name)
            return module.Interpreter
        except (ImportError, AttributeError):
            continue
    return None


def load_onnxruntime_module():
    try:
        return importlib.import_module("onnxruntime")
    except ImportError:
        return None


def build_phone_label_metadata(model_cfg):
    """Build consistent label id set and id->name map from config."""
    label_map_cfg = model_cfg.get("phone_labels")
    if isinstance(label_map_cfg, dict) and label_map_cfg:
        id_to_name = {}
        for k, v in label_map_cfg.items():
            try:
                class_id = int(k)
            except (TypeError, ValueError):
                continue
            id_to_name[class_id] = str(v)
        if id_to_name:
            return set(id_to_name.keys()), id_to_name

    # Backward compatibility for old split arrays.
    label_ids = list(model_cfg.get("phone_label_ids", [67]))
    label_names = list(model_cfg.get("phone_label_names", []))
    id_to_name = {}
    for class_id, class_name in zip(label_ids, label_names):
        try:
            id_to_name[int(class_id)] = str(class_name)
        except (TypeError, ValueError):
            continue

    label_id_set = {int(v) for v in label_ids if isinstance(v, (int, np.integer))}
    if not label_id_set:
        label_id_set = {67}
    return label_id_set, id_to_name



def load_phone_detector(cfg):
    model_cfg = cfg.get("model", {})
    backend = str(model_cfg.get("phone_backend", "tflite")).strip().lower()
    phone_label_ids, phone_label_map = build_phone_label_metadata(model_cfg)

    if backend == "onnx":
        model_path = Path(__file__).with_name(model_cfg.get("phone_onnx_path", "yolo26n.onnx"))
        ort = load_onnxruntime_module()
        if ort is None:
            print("[phone] onnxruntime not installed. Phone detection disabled.")
            return None
        if not model_path.exists():
            print(f"[phone] ONNX model not found at {model_path}. Phone detection disabled.")
            return None

        try:
            session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        except Exception as exc:
            print(f"[phone] Failed to initialize ONNX runtime: {exc}")
            return None

        input_info = session.get_inputs()[0]
        input_name = input_info.name
        shape = input_info.shape

        # Expected NCHW. If dynamic dims are used, fall back to 640.
        input_h = int(shape[2]) if len(shape) >= 4 and isinstance(shape[2], int) else 640
        input_w = int(shape[3]) if len(shape) >= 4 and isinstance(shape[3], int) else 640

        return {
            "backend": "onnx",
            "session": session,
            "input_name": input_name,
            "output_names": [o.name for o in session.get_outputs()],
            "input_width": input_w,
            "input_height": input_h,
            "score_threshold": float(model_cfg.get("phone_score_threshold", 0.45)),
            "phone_label_ids": phone_label_ids,
            "phone_label_map": phone_label_map,
            "available": True,
        }

    model_path = Path(__file__).with_name(model_cfg.get("phone_tflite_path", "phone_detection.tflite"))
    interpreter_cls = load_tflite_interpreter_cls()

    if interpreter_cls is None:
        print("[phone] TFLite runtime not installed. Phone detection disabled.")
        return None
    if not model_path.exists():
        print(f"[phone] TFLite model not found at {model_path}. Phone detection disabled.")
        return None

    interpreter = interpreter_cls(model_path=str(model_path))
    interpreter.allocate_tensors()

    input_detail = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()
    _, input_h, input_w, _ = input_detail["shape"]

    return {
        "backend": "tflite",
        "interpreter": interpreter,
        "input_detail": input_detail,
        "output_details": output_details,
        "input_width": int(input_w),
        "input_height": int(input_h),
        "score_threshold": float(model_cfg.get("phone_score_threshold", 0.45)),
        "phone_label_ids": phone_label_ids,
        "phone_label_map": phone_label_map,
        "available": True,
    }



def preprocess_tflite_frame(frame, detector):
    # 모델 입력 크기에 맞춰 리사이즈 및 RGB 변환
    resized = cv2.resize(frame, (detector["input_width"], detector["input_height"]))
    input_tensor = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    input_tensor = np.expand_dims(input_tensor, axis=0)

    input_detail = detector["input_detail"]
    dtype = input_detail["dtype"]

    if dtype == np.float32:
        # Float32 모델: 0~1 사이로 정규화
        input_tensor = input_tensor.astype(np.float32) / 255.0
    else:
        # INT8/UINT8 모델: Quantization 파라미터 적용
        params = input_detail.get("quantization_parameters", {})
        scales = params.get("scales")
        zero_points = params.get("zero_points")
        
        if scales is not None and len(scales) > 0:
            scale = scales[0]
            zero_point = zero_points[0]
            # 수식: (float_value / scale) + zero_point
            input_tensor = (input_tensor.astype(np.float32) / 255.0 / scale) + zero_point
            input_tensor = input_tensor.astype(dtype)
        else:
            input_tensor = input_tensor.astype(dtype)
            
    return input_tensor


def preprocess_onnx_frame(frame, detector):
    resized = cv2.resize(frame, (detector["input_width"], detector["input_height"]))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    chw = np.transpose(rgb, (2, 0, 1)).astype(np.float32) / 255.0
    return np.expand_dims(chw, axis=0)


def _nms_detections(detections, iou_threshold=0.45):
    if not detections:
        return []

    boxes = []
    scores = []
    for x1, y1, x2, y2, score, _class_id in detections:
        boxes.append([int(x1), int(y1), int(max(1, x2 - x1)), int(max(1, y2 - y1))])
        scores.append(float(score))

    keep = cv2.dnn.NMSBoxes(boxes, scores, score_threshold=0.0, nms_threshold=iou_threshold)
    if keep is None or len(keep) == 0:
        return []

    idxs = [int(i[0]) if isinstance(i, (list, tuple, np.ndarray)) else int(i) for i in keep]
    return [detections[i] for i in idxs]


def parse_onnx_outputs(raw_outputs, frame_shape, detector):
    h, w = frame_shape[:2]
    score_threshold = detector["score_threshold"]
    phone_label_ids = detector["phone_label_ids"]

    if not raw_outputs:
        return False, []

    output = np.array(raw_outputs[0])
    if output.ndim == 3 and output.shape[0] == 1:
        output = output[0]

    candidates = []

    # Case A: [N, 6] => x1,y1,x2,y2,score,class_id
    if output.ndim == 2 and output.shape[1] >= 6:
        for det in output:
            x1, y1, x2, y2, score, class_id = det[:6]
            class_id = int(class_id)
            if score < score_threshold or class_id not in phone_label_ids:
                continue

            # If normalized, convert to pixels.
            if max(abs(x1), abs(y1), abs(x2), abs(y2)) <= 1.5:
                x1, y1, x2, y2 = x1 * w, y1 * h, x2 * w, y2 * h

            x1 = max(0, min(w - 1, int(x1)))
            y1 = max(0, min(h - 1, int(y1)))
            x2 = max(0, min(w - 1, int(x2)))
            y2 = max(0, min(h - 1, int(y2)))
            if x2 <= x1 or y2 <= y1:
                continue
            candidates.append((x1, y1, x2, y2, float(score), class_id))

    # Case B: [C, N] or [N, C], commonly YOLO [84, N] / [N, 84]
    elif output.ndim == 2:
        preds = output
        if preds.shape[0] < preds.shape[1] and preds.shape[0] <= 128:
            preds = preds.T

        if preds.shape[1] >= 6:
            in_w = float(detector["input_width"])
            in_h = float(detector["input_height"])
            sx = w / max(in_w, 1.0)
            sy = h / max(in_h, 1.0)

            for row in preds:
                x, y, bw, bh = row[:4]
                class_scores = row[4:]
                if class_scores.size == 0:
                    continue
                class_id = int(np.argmax(class_scores))
                score = float(class_scores[class_id])
                if score < score_threshold or class_id not in phone_label_ids:
                    continue

                # Heuristic: values <=1.5 are normalized, otherwise assume input-scale units.
                if max(abs(x), abs(y), abs(bw), abs(bh)) <= 1.5:
                    cx, cy, ww, hh = x * w, y * h, bw * w, bh * h
                else:
                    cx, cy, ww, hh = x * sx, y * sy, bw * sx, bh * sy

                x1 = int(cx - ww / 2.0)
                y1 = int(cy - hh / 2.0)
                x2 = int(cx + ww / 2.0)
                y2 = int(cy + hh / 2.0)
                x1 = max(0, min(w - 1, x1))
                y1 = max(0, min(h - 1, y1))
                x2 = max(0, min(w - 1, x2))
                y2 = max(0, min(h - 1, y2))
                if x2 <= x1 or y2 <= y1:
                    continue
                candidates.append((x1, y1, x2, y2, score, class_id))

    detections = _nms_detections(candidates)
    return len(detections) > 0, detections

def parse_tflite_outputs(raw_outputs, frame_shape, detector):
    h, w = frame_shape[:2]
    score_threshold = detector["score_threshold"]
    phone_label_ids = detector["phone_label_ids"]

    # 출력 텐서 가져오기 (보통 첫 번째 텐서에 결과가 포함됨)
    output = raw_outputs[0]
    if output.ndim == 3:
        output = output[0]  # [1, N, 6] -> [N, 6]

    detections = []
    
    for det in output:
        # Ultralytics TFLite 포맷에 따라 인덱스가 다를 수 있음
        # 일반적인 구조: [x1, y1, x2, y2, score, class_id] (정규화된 값)
        if len(det) < 6:
            continue
            
        x1_norm, y1_norm, x2_norm, y2_norm, score, class_id = det[:6]
        
        # 스코어 및 클래스 필터링
        if score < score_threshold or int(class_id) not in phone_label_ids:
            continue

        # 핵심 수정: 정규화된 좌표(0~1)를 실제 이미지 픽셀 크기로 변환
        # 모델에 따라 x, y, w, h 형태일 수도 있으니 확인 필요 (여기서는 x1, y1, x2, y2 가정)
        real_x1 = int(x1_norm * w)
        real_y1 = int(y1_norm * h)
        real_x2 = int(x2_norm * w)
        real_y2 = int(y2_norm * h)

        # 이미지 경계 제한
        real_x1 = max(0, min(w - 1, real_x1))
        real_y1 = max(0, min(h - 1, real_y1))
        real_x2 = max(0, min(w - 1, real_x2))
        real_y2 = max(0, min(h - 1, real_y2))

        if real_x2 <= real_x1 or real_y2 <= real_y1:
            continue
            
        # detections.append((real_x1, real_y1, real_x2, real_y2, float(score)))
        detections.append((real_x1, real_y1, real_x2, real_y2, float(score), int(class_id)))

    return len(detections) > 0, detections

def detect_phone(frame, detector):
    backend = detector.get("backend", "tflite")

    if backend == "onnx":
        session = detector["session"]
        input_name = detector["input_name"]
        output_names = detector["output_names"]

        input_tensor = preprocess_onnx_frame(frame, detector)
        raw_outputs = session.run(output_names, {input_name: input_tensor})
        return parse_onnx_outputs(raw_outputs, frame.shape, detector)

    interpreter = detector["interpreter"]
    input_detail = detector["input_detail"]
    output_details = detector["output_details"]

    input_tensor = preprocess_tflite_frame(frame, detector)
    interpreter.set_tensor(input_detail["index"], input_tensor)
    interpreter.invoke()
    raw_outputs = [interpreter.get_tensor(detail["index"]) for detail in output_details]
    return parse_tflite_outputs(raw_outputs, frame.shape, detector)



def draw_phone_boxes(frame, boxes, cfg=None):
    # cfg: config dict, 필요시 label id-name 매핑에 사용
    label_id_to_name = None
    if cfg is not None:
        model_cfg = cfg.get("model", {})
        _label_ids, label_id_to_name = build_phone_label_metadata(model_cfg)
    for det in boxes:
        # det: (x1, y1, x2, y2, score, [class_id])
        if len(det) == 6:
            x1, y1, x2, y2, score, class_id = det
        else:
            x1, y1, x2, y2, score = det
            class_id = None
        label = None
        if label_id_to_name is not None and class_id is not None:
            label = label_id_to_name.get(class_id, str(class_id))
        elif class_id is not None:
            label = str(class_id)
        else:
            label = "Object"
        text = f"{label} {score:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(
            frame,
            text,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            2,
        )


def draw_landmarks(frame, mp_results, mp_holistic, mp_drawing, mp_drawing_styles):
    if mp_results.face_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            mp_results.face_landmarks,
            mp_holistic.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style(),
        )
        mp_drawing.draw_landmarks(
            frame,
            mp_results.face_landmarks,
            mp_holistic.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style(),
        )
    if mp_results.left_hand_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            mp_results.left_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style(),
        )
    if mp_results.right_hand_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            mp_results.right_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style(),
        )
    if mp_results.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            mp_results.pose_landmarks,
            mp_holistic.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
        )


def draw_ui(frame, status, cfg, runtime=None, tracker_result=None):
    h, w, _ = frame.shape
    thresholds = cfg["thresholds"]

    # desk_y = int(h * thresholds["desk_y_threshold"])
    # cv2.line(frame, (0, desk_y), (w, desk_y), (255, 255, 0), 2)

    panel_cfg = cfg.get("ui", {})
    panel_alpha = float(panel_cfg.get("panel_alpha", 0.42))
    panel_w = int(w * float(panel_cfg.get("panel_width_ratio", 0.55)))
    panel_h = int(h * float(panel_cfg.get("panel_height_ratio", 0.5)))

    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (10 + panel_w, 10 + panel_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, panel_alpha, frame, 1.0 - panel_alpha, 0, frame)
    color_ok = (0, 255, 0)
    color_alert = (0, 0, 255)

    cv2.putText(
        frame,
        f"Phone: {status['phone_detected']}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color_alert if status["phone_detected"] else color_ok,
        2,
    )
    cv2.putText(
        frame,
        f"Head-Align: {'Mismatch' if status['status_yaw_out'] else 'Aligned'}",
        (20, 62),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color_alert if status["status_yaw_out"] else color_ok,
        2,
    )
    cv2.putText(
        frame,
        f"Hands on Desk: {'No' if status['status_no_hands'] else 'Yes'}",
        (20, 89),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color_alert if status["status_no_hands"] else color_ok,
        2,
    )
    cv2.putText(
        frame,
        f"Tracker Out: {status['status_tracker_out']} ({status['tracker_out_sec']:.1f}s)",
        (20, 116),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color_alert if status["status_tracker_out"] else color_ok,
        2,
    )
    cv2.putText(
        frame,
        f"Hand Visible: {status['has_hand_visible']} StartReady: {status['study_started']}",
        (20, 170),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color_alert if status["status_no_hands"] else color_ok,
        1,
    )
    cv2.putText(
        frame,
        f"Smile+Talk: {status['status_smile_talking']} (S:{status['smile_ratio']:.2f}, T:{status['mouth_open_ratio']:.2f})",
        (20, 194),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color_alert if status["status_smile_talking"] else color_ok,
        1,
    )
    cv2.putText(
        frame,
        f"TalkDetect: {status['smile_talk_detect_sec']:.1f}s/{status['smile_talk_window_sec']:.1f}s Stdev:{status['talk_stdev']:.3f}",
        (20, 218),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color_alert if status["status_smile_talking"] else color_ok,
        1,
    )
    cv2.putText(
        frame,
        f"Tracker: {'Matched' if status['tracker_matched'] else 'Predicting'} Lost: {status['tracker_lost_frames']}",
        (20, 242),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color_alert if status["tracker_lost_frames"] > 0 else color_ok,
        1,
    )
    cv2.putText(
        frame,
        f"Track Stdev: {status['tracker_history_std']:.4f}",
        (20, 266),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color_alert if status["tracker_history_std"] > 0.21 else color_ok,
        1,
    )
    cv2.putText(
        frame,
        f"Calibration: {status['calibration_state']}",
        (20, 290),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color_ok,
        1,
    )

    # 트래커 히스토리/위치 시각화
    if runtime is not None and tracker_result is not None:
        draw_tracker_history(frame, runtime, tracker_result, compute_tracker_out_of_screen)

    final_text = "CONCENTRATING" if status["is_concentrating"] else "DISTRACTED"
    final_color = (0, 255, 0) if status["is_concentrating"] else (0, 0, 255)
    cv2.putText(frame, final_text, (10, h - 30), cv2.FONT_HERSHEY_DUPLEX, 1.2, final_color, 2)


def analyze_frame(frame, mp_results, phone_detector, runtime, cfg, mp_holistic, dt, yolo_async=None):
    thresholds = runtime["active_thresholds"]
    features = cfg["features"]
    model_cfg = cfg.get("model", {})
    tracking_cfg = cfg.get("tracking", {})
    h, w, _ = frame.shape

    face_landmarks = mp_results.face_landmarks
    pose_landmarks = mp_results.pose_landmarks

    phone_detected = False
    phone_boxes = []
    phone_interval = max(1, int(model_cfg.get("phone_detect_every_n_frames", 7)))
    phone_detector_ready = phone_detector is not None and phone_detector.get("available", False)
    # YOLO inference 비동기 처리
    if features["enable_phone_detection"] and phone_detector_ready:
        if yolo_async is not None:
            # yolo_async: dict with keys future, last_result, last_frame_index
            if (runtime["frame_index"] - 1) % phone_interval == 0:
                # 새 프레임에 대해 YOLO inference 요청
                if yolo_async["future"] is None or yolo_async["future"].done():
                    # 이전 결과 저장
                    if yolo_async["future"] is not None and yolo_async["future"].done():
                        try:
                            yolo_async["last_result"] = yolo_async["future"].result()
                            yolo_async["last_frame_index"] = runtime["frame_index"]
                        except Exception:
                            yolo_async["last_result"] = (False, [])
                    # 새 inference 요청
                    yolo_async["future"] = yolo_async["executor"].submit(detect_phone, frame.copy(), phone_detector)
            # 항상 최신 결과 사용
            if yolo_async["last_result"] is not None:
                phone_detected, phone_boxes = yolo_async["last_result"]
            else:
                phone_detected, phone_boxes = False, []
            runtime["last_phone_detection"] = {
                "detected": phone_detected,
                "boxes": phone_boxes,
                "available": True,
            }
        else:
            # 기존 동기 방식 fallback
            if (runtime["frame_index"] - 1) % phone_interval == 0:
                phone_detected, phone_boxes = detect_phone(frame, phone_detector)
                runtime["last_phone_detection"] = {
                    "detected": phone_detected,
                    "boxes": phone_boxes,
                    "available": True,
                }
            else:
                phone_detected = runtime["last_phone_detection"]["detected"]
                phone_boxes = runtime["last_phone_detection"]["boxes"]
    else:
        runtime["last_phone_detection"] = {
            "detected": False,
            "boxes": [],
            "available": phone_detector_ready,
        }

    requires_hand_contact = bool(model_cfg.get("phone_requires_hand_contact", True))
    hand_contact_distance = float(tracking_cfg.get("object_hand_max_distance_ratio", 0.05))
    lower_region_threshold = float(tracking_cfg.get("object_bottom_ignore_threshold", 0.8))
    if phone_detected and requires_hand_contact:
        phone_detected = is_object_held_by_hand(
            phone_boxes,
            mp_results,
            pose_landmarks,
            mp_holistic,
            frame.shape,
            max_distance_ratio=hand_contact_distance,
            lower_region_threshold=lower_region_threshold,
        )

    status = {
        "phone_detected": phone_detected,
        "status_no_hands": False,
        "status_face_missing": False,
        "status_tracker_out": False,
        "status_smile_talking": False,
        "status_yaw_out": False,
        "has_hand_visible": False,
        "study_started": runtime["study_started"],
        "tracker_out_sec": 0.0,
        "face_missing_sec": 0.0,
        "smile_ratio": 0.0,
        "mouth_open_ratio": 0.0,
        "talk_variance": 0.0,
        "talk_stdev": 0.0,
        "smile_talk_detect_sec": 0.0,
        "smile_talk_window_sec": float(thresholds.get("smile_talk_window_seconds", 2.0)),
        "tracker_history_var": 0.0,
        "tracker_history_std": 0.0,
        "tracker_matched": False,
        "tracker_lost_frames": runtime["tracker"]["lost_frames"],
        "calibration_state": "off",
        "is_concentrating": True,
    }

    measurement = extract_face_measurement(face_landmarks, pose_landmarks, mp_holistic)
    tracker_result = update_face_tracker(runtime, measurement, dt, cfg)

    if tracker_result is not None:
        status["tracker_matched"] = tracker_result["matched"]
        status["tracker_lost_frames"] = runtime["tracker"]["lost_frames"]

    now_ts = time.perf_counter()
    smile_talk_window_sec = float(thresholds.get("smile_talk_window_seconds", 2.0))
    smile_talk_required_frames = int(max(1, thresholds.get("smile_talk_frames", 5)))
    talk_stdev_threshold = float(thresholds.get("talking_stdev_threshold", 0.01))

    if face_landmarks:
        runtime["face_missing_counter"] = 0
        if features.get("enable_smile_talking_detection", True):
            smile_ratio, mouth_open_ratio = estimate_smile_talk_features(face_landmarks)
            status["smile_ratio"] = smile_ratio
            status["mouth_open_ratio"] = mouth_open_ratio
            is_smile = smile_ratio <= thresholds.get("smile_ratio_threshold", 8.0)
            is_talking = mouth_open_ratio >= thresholds.get("talking_open_ratio_threshold", 0.06)
            runtime["smile_talk_events"].append((now_ts, 1 if (is_smile and is_talking) else 0))
            runtime["talk_values"].append((now_ts, float(mouth_open_ratio)))
    else:
        if features.get("enable_face_missing_detection", True):
            runtime["face_missing_counter"] += 1
        status["smile_ratio"] = 0.0
        status["mouth_open_ratio"] = 0.0

    cutoff_ts = now_ts - smile_talk_window_sec
    while runtime["smile_talk_events"] and runtime["smile_talk_events"][0][0] < cutoff_ts:
        runtime["smile_talk_events"].popleft()
    while runtime["talk_values"] and runtime["talk_values"][0][0] < cutoff_ts:
        runtime["talk_values"].popleft()

    hit_count = int(sum(v for _, v in runtime["smile_talk_events"]))
    runtime["smile_talk_counter"] = hit_count
    talk_series = [v for _, v in runtime["talk_values"]]
    talk_variance = float(np.var(talk_series)) if len(talk_series) >= 2 else 0.0
    talk_stdev = float(np.sqrt(talk_variance))
    status["talk_variance"] = talk_variance
    status["talk_stdev"] = talk_stdev
    status["smile_talk_detect_sec"] = hit_count / max(runtime["fps"], 1)
    status["smile_talk_window_sec"] = smile_talk_window_sec
    status["status_smile_talking"] = (
        hit_count >= smile_talk_required_frames and talk_stdev >= talk_stdev_threshold
    )

    is_tracker_out = compute_tracker_out_of_screen(tracker_result)
    if is_tracker_out:
        runtime["tracker_out_counter"] += 1
    else:
        runtime["tracker_out_counter"] = 0

    tracker_out_frames_threshold = int(max(1, thresholds.get("tracker_out_seconds", 0.6) * runtime["fps"]))
    status["status_tracker_out"] = runtime["tracker_out_counter"] >= tracker_out_frames_threshold
    status["tracker_out_sec"] = runtime["tracker_out_counter"] / max(runtime["fps"], 1)

    if features.get("enable_face_missing_detection", True):
        face_missing_frames_threshold = int(max(1, thresholds.get("face_missing_seconds", 1.2) * runtime["fps"]))
        status["status_face_missing"] = runtime["face_missing_counter"] >= face_missing_frames_threshold
        status["face_missing_sec"] = runtime["face_missing_counter"] / max(runtime["fps"], 1)
    else:
        status["status_face_missing"] = False
        status["face_missing_sec"] = 0.0

    # --- Yaw Calibration & Comparison ---
    calib_yaw = runtime.setdefault("yaw_calib", None)
    mediapipe_yaw = None
    if face_landmarks:
        mediapipe_yaw = get_face_head_yaw(face_landmarks)
        status["mediapipe_yaw"] = mediapipe_yaw
    else:
        status["mediapipe_yaw"] = None
    if calib_yaw is not None and mediapipe_yaw is not None:
        status["yaw_from_calib"] = mediapipe_yaw - calib_yaw
    else:
        status["yaw_from_calib"] = None

    # Yaw calibration/compare 텍스트 표시 (deg 단위)
    def yaw_to_deg(yaw):
        if yaw is None:
            return "-"
        return f"{yaw * 90.0:.1f}"

    # --- Yaw 범위 이탈 감지 (face_mesh 고정) ---
    yaw_max_deg = float(thresholds.get("yaw_max_degrees", 30.0))
    yaw_out_seconds = float(thresholds.get("yaw_out_seconds", 0.6))
    yaw_out_frames_threshold = int(max(1, yaw_out_seconds * runtime["fps"]))
    yaw_deg = status["yaw_from_calib"]
    if yaw_deg is not None and abs(yaw_deg * 90.0) > yaw_max_deg:
        runtime["yaw_out_counter"] += 1
    else:
        runtime["yaw_out_counter"] = 0
    status["status_yaw_out"] = runtime["yaw_out_counter"] >= yaw_out_frames_threshold
    
    ytxt = f"Yaw(calib): {yaw_to_deg(status.get('yaw_from_calib', None))} deg  Yaw(mp): {yaw_to_deg(status.get('mediapipe_yaw', None))} deg"
    color = (0, 0, 255) if status["status_yaw_out"] else (255, 255, 0)
    cv2.putText(frame, ytxt, (20, h - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    min_hand_visibility = float(thresholds.get("min_pose_visibility_for_hand", 0.35))
    has_hand_visible = has_any_visible_hand(mp_results, pose_landmarks, mp_holistic, min_hand_visibility)
    status["has_hand_visible"] = has_hand_visible

    if not runtime["study_started"] and has_hand_visible:
        runtime["study_started"] = True
    status["study_started"] = runtime["study_started"]

    no_hand_frames_threshold = int(max(1, thresholds.get("no_hand_seconds", 0.8) * runtime["fps"]))
    if runtime["study_started"]:
        if has_hand_visible:
            runtime["no_hand_counter"] = 0
        else:
            runtime["no_hand_counter"] += 1
        status["status_no_hands"] = runtime["no_hand_counter"] >= no_hand_frames_threshold
    else:
        status["status_no_hands"] = True

    if pose_landmarks:
        if features["enable_hands_on_desk_detection"]:
            status_hands_on_desk = check_hands_on_desk(
                pose_landmarks,
                mp_holistic,
                thresholds["desk_y_threshold"],
            )
            if not status_hands_on_desk:
                status["status_no_hands"] = True

    # tracker history 분산 (디버그용)
    hist_points = [np.asarray(c, dtype=np.float32) for c, _ in runtime["tracker"].get("history", []) if c is not None]
    if len(hist_points) >= 2:
        arr = np.vstack(hist_points)
        status["tracker_history_var"] = float(np.var(arr[-10:, 0]) + np.var(arr[-10:, 1]))
        status["tracker_history_std"] = float(np.sqrt(np.var(arr[-10:, 0]) + np.var(arr[-10:, 1])))
    else:
        status["tracker_history_var"] = 0.0
        status["tracker_history_std"] = 0.0

    maybe_update_calibration(runtime, status, tracker_result)
    legacy_calib = runtime["calibration"]
    legacy_state = "off"
    if legacy_calib["enabled"]:
        if legacy_calib["done"]:
            legacy_state = "done"
        else:
            elapsed = max(0.0, time.perf_counter() - legacy_calib["start_ts"]) if legacy_calib["started"] else 0.0
            legacy_state = f"running {elapsed:.1f}/{legacy_calib['duration_seconds']:.1f}s"

    status["calibration_state"] = legacy_state

    status["is_concentrating"] = not (
        (not status["study_started"])
        or status["status_no_hands"]
        or status["phone_detected"]
        or status["status_tracker_out"]
        or status["status_face_missing"]
        or status["status_smile_talking"]
        or status["status_yaw_out"]
    )

    if phone_boxes:
        # label name 표시를 위해 cfg 전달
        draw_phone_boxes(frame, phone_boxes, cfg=cfg)
    return status, tracker_result


def run_monitoring(cfg):
    mp_solutions = load_mediapipe_solutions()
    mp_holistic = mp_solutions.holistic
    mp_drawing = mp_solutions.drawing_utils
    mp_drawing_styles = mp_solutions.drawing_styles

    phone_detector = load_phone_detector(cfg) if cfg["features"].get("enable_phone_detection", False) else None

    cap = cv2.VideoCapture(cfg["camera"]["index"])
    raw_fps = cap.get(cv2.CAP_PROP_FPS)
    fps = raw_fps if raw_fps and raw_fps > 0 else 30.0

    runtime = build_runtime_state(cfg, fps)

    # YOLO inference 비동기용 ThreadPoolExecutor 및 상태 dict
    yolo_async = None
    if phone_detector is not None:
        yolo_async = {
            "executor": ThreadPoolExecutor(max_workers=1),
            "future": None,
            "last_result": None,
            "last_frame_index": -1,
        }

    try:
        with mp_holistic.Holistic(
            min_detection_confidence=cfg["mediapipe"]["min_detection_confidence"],
            min_tracking_confidence=cfg["mediapipe"]["min_tracking_confidence"],
            refine_face_landmarks=cfg["mediapipe"].get("refine_face_landmarks", True),
        ) as holistic:
            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    break

                runtime["frame_index"] += 1
                now_ts = time.perf_counter()
                dt = max(1e-3, now_ts - runtime["last_ts"])
                runtime["last_ts"] = now_ts

                if cfg["camera"].get("flip_horizontal", True):
                    frame = cv2.flip(frame, 1)

                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image_rgb.flags.writeable = False
                mp_results = holistic.process(image_rgb)
                image_rgb.flags.writeable = True

                status, tracker_result = analyze_frame(frame, mp_results, phone_detector, runtime, cfg, mp_holistic, dt, yolo_async=yolo_async)

                if cfg["features"].get("draw_landmarks", True):
                    draw_landmarks(frame, mp_results, mp_holistic, mp_drawing, mp_drawing_styles)

                draw_ui(frame, status, cfg, runtime=runtime, tracker_result=tracker_result)
                cv2.imshow("Monitoring", frame)

                key = cv2.waitKey(1) & 0xFF
                if key == 27:
                    break
                elif key == ord('c'):
                    # Yaw 캘리브레이션: 현재 mediapipe yaw를 기준점으로 저장
                    if 'mediapipe_yaw' in status and status['mediapipe_yaw'] is not None:
                        runtime['yaw_calib'] = status['mediapipe_yaw']
                        print(f"[Yaw Calibration] 기준 yaw 저장: {runtime['yaw_calib']:.4f}")
    finally:
        if yolo_async is not None:
            yolo_async["executor"].shutdown(wait=False)
        cap.release()
        cv2.destroyAllWindows()


def main():
    cfg = load_config(CONFIG_PATH)
    run_monitoring(cfg)


if __name__ == "__main__":
    main()