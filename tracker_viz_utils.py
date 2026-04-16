import cv2

def clip_point_to_frame(pt, w, h):
    x = min(max(pt[0], 0.0), 1.0)
    y = min(max(pt[1], 0.0), 1.0)
    return x, y

def draw_tracker_history(frame, runtime, tracker_result, compute_tracker_out_of_screen):
    h, w, _ = frame.shape
    tracker = runtime["tracker"]
    # 히스토리 관리
    if tracker_result is not None and tracker_result["center"] is not None:
        center = tracker_result["center"]
        is_out = compute_tracker_out_of_screen(tracker_result)
        tracker["history"].append((center.copy(), is_out))
        if len(tracker["history"]) > tracker["history_maxlen"]:
            tracker["history"] = tracker["history"][-tracker["history_maxlen"]:]

    # 히스토리 선 그리기 (과거→현재, 투명도 증가)
    n = len(tracker["history"])
    if n < 2:
        return
    overlay = frame.copy()
    for i in range(1, n):
        pt1, out1 = tracker["history"][i-1]
        pt2, out2 = tracker["history"][i]
        # 화면 밖이면 클리핑
        if out1:
            pt1 = clip_point_to_frame(pt1, 1.0, 1.0)
        if out2:
            pt2 = clip_point_to_frame(pt2, 1.0, 1.0)
        x1, y1 = int(pt1[0] * w), int(pt1[1] * h)
        x2, y2 = int(pt2[0] * w), int(pt2[1] * h)
        alpha = (i / n)
        color = (255, 0, 0) if not out2 else (0, 0, 255)
        thickness = 2
        cv2.line(overlay, (x1, y1), (x2, y2), color, thickness)
        # 점도 그리기 (선명도 증가)
        cv2.circle(overlay, (x2, y2), 4, color, -1)
    # 투명도 적용
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    # 현재 위치 강조 (더 큰 원)
    pt, is_out = tracker["history"][-1]
    if is_out:
        pt = clip_point_to_frame(pt, 1.0, 1.0)
    x, y = int(pt[0] * w), int(pt[1] * h)
    color = (255, 0, 0) if not is_out else (0, 0, 255)
    thickness = 3 if not is_out else 2
    radius = 14 if not is_out else 14
    # 화면 밖이면 원 테두리만
    if is_out:
        cv2.circle(frame, (x, y), radius, color, thickness)
    else:
        cv2.circle(frame, (x, y), radius, color, -1)
        cv2.circle(frame, (x, y), radius, (255,255,255), 2)
