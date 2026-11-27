"""
Backend kiểm soát vé bằng dấu X.
  - POST /upload      : ESP32-CAM gửi ảnh JPEG (raw body), server nhận diện dấu X
  - GET  /api/events  : trả lịch sử + thống kê cho giao diện web
  - POST /api/clear   : xóa lịch sử
  - GET  /            : trang web (static/index.html)

Cài đặt:  python -m pip install flask opencv-python numpy
Chạy:     python app.py
"""
import os
import json
import threading
from datetime import datetime

import cv2
import numpy as np
from flask import Flask, request, jsonify, send_from_directory

# ====== CẤU HÌNH (chỉnh khi test với ảnh thật) ======
INK_COLOR = "dark"        # "dark" = nét tối trên nền sáng (mọi màu bút), hoặc "blue", "red", "green", "black"
DARK_THRESH = 9           # chế độ "dark": độ tối tối thiểu của nét so với nền (nhỏ = nhạy hơn)
MIN_INK_AREA = 150        # số pixel mực tối thiểu để coi là có dấu
MIN_COMPONENT = 40        # bỏ các đốm mực nhỏ hơn mức này (nhiễu)
MIN_CROSS_ANGLE = 50      # góc giao nhau tối thiểu giữa 2 nét (độ)
GROUP_DILATE = 25         # gộp các nét đứt gần nhau thành một cụm (pixel)
MAX_GROUP_FRAC = 0.06     # cụm mực không được chiếm quá % diện tích ảnh (loại cảnh lộn xộn)
MAX_GROUP_SIDE = 0.80     # cạnh dài nhất của cụm không quá % cạnh dài của ảnh
MIN_GROUP_SIDE = 0.10     # cạnh dài nhất của cụm tối thiểu % cạnh ngắn của ảnh
MAX_FILL = 0.35           # tỉ lệ mực / hình chữ nhật bao quanh (dấu X thì mảnh nên thấp)
MIN_STROKE_FRAC = 0.40    # mỗi nét phải dài ít nhất % đường chéo của cụm
CENTER_TOL = 0.30         # giao điểm hai nét phải nằm gần tâm cụm (tỉ lệ so với kích thước cụm)
ROI = (0.05, 0.05, 0.95, 0.95)   # chỉ xét vùng này của ảnh (x0, y0, x1, y1 theo tỉ lệ)
DIAG_MARGIN = 20          # mỗi nét phải lệch ít nhất góc này so với phương ngang/dọc (loại dấu +)
MAX_EVENTS = 200          # số sự kiện lưu tối đa
# ====================================================

# Dải màu HSV của OpenCV (H: 0-180, S: 0-255, V: 0-255)
INK_RANGES = {
    "blue":  [((90, 70, 40), (130, 255, 255))],
    "red":   [((0, 90, 60), (10, 255, 255)), ((170, 90, 60), (180, 255, 255))],
    "green": [((40, 60, 40), (85, 255, 255))],
    "black": [((0, 0, 0), (180, 255, 70))],
}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CAP_DIR = os.path.join(BASE_DIR, "captures")
STATIC_DIR = os.path.join(BASE_DIR, "static")
EVENTS_FILE = os.path.join(BASE_DIR, "events.json")
os.makedirs(CAP_DIR, exist_ok=True)

app = Flask(__name__, static_folder=STATIC_DIR, static_url_path="/static")
lock = threading.Lock()


# ---------- Lưu lịch sử ----------
def load_events():
    if os.path.exists(EVENTS_FILE):
        try:
            with open(EVENTS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            pass
    return []


def save_events(events):
    with open(EVENTS_FILE, "w", encoding="utf-8") as f:
        json.dump(events, f, ensure_ascii=False)


events = load_events()


# ---------- Nhận diện dấu X ----------
def build_mask(img):
    if INK_COLOR == "dark":
        # Nét tối mảnh trên nền sáng: dùng black-hat để tách nét khỏi nền
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        k = max(9, int(min(gray.shape) * 0.05)) | 1
        kernel_bh = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        bh = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel_bh)
        _, mask = cv2.threshold(bh, DARK_THRESH, 255, cv2.THRESH_BINARY)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    else:
        hsv = cv2.cvtColor(cv2.GaussianBlur(img, (5, 5), 0), cv2.COLOR_BGR2HSV)
        mask = None
        for lo, hi in INK_RANGES[INK_COLOR]:
            m = cv2.inRange(hsv, np.array(lo), np.array(hi))
            mask = m if mask is None else cv2.bitwise_or(mask, m)
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Bỏ các đốm nhỏ
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    clean = np.zeros_like(mask)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= MIN_COMPONENT:
            clean[labels == i] = 255
    return clean


def seg_intersect(a, b):
    """Giao điểm của 2 đoạn thẳng (cho phép lệch nhẹ ở đầu mút), hoặc None."""
    x1, y1, x2, y2 = a
    x3, y3, x4, y4 = b
    d = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(d) < 1e-6:
        return None
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / d
    u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / d
    if -0.6 <= t <= 1.6 and -0.6 <= u <= 1.6:
        return (x1 + t * (x2 - x1), y1 + t * (y2 - y1))
    return None


def is_diagonal(angle):
    """True nếu nét nghiêng chéo (không gần phương ngang hoặc dọc)."""
    d = min(angle % 90, 90 - angle % 90)
    return d >= DIAG_MARGIN


def seg_angle(s):
    x1, y1, x2, y2 = s
    return np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180


def detect_x(img):
    """Trả về (valid, ink_area, ảnh_đã_vẽ_kết_quả)."""
    out = img.copy()
    H, W = img.shape[:2]
    mask = build_mask(img)

    # Chỉ xét vùng ROI, bỏ phần rìa ảnh
    roi = np.zeros_like(mask)
    x0, y0, x1, y1 = int(ROI[0] * W), int(ROI[1] * H), int(ROI[2] * W), int(ROI[3] * H)
    roi[y0:y1, x0:x1] = 255
    mask = cv2.bitwise_and(mask, roi)
    ink_area = int(cv2.countNonZero(mask))

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, cnts, -1, (0, 255, 255), 1)  # viền vàng = vùng mực tìm được
    cv2.rectangle(out, (x0, y0), (x1, y1), (255, 160, 0), 1)  # khung xanh dương = vùng được xét

    # Gộp các nét đứt thành từng cụm, rồi kiểm tra từng cụm xem có phải dấu X không
    grp = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (GROUP_DILATE, GROUP_DILATE)))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(grp)

    valid = False
    for i in range(1, n):
        gx, gy, gw, gh, _ = stats[i]
        sub = np.where(labels[gy:gy + gh, gx:gx + gw] == i, mask[gy:gy + gh, gx:gx + gw], 0).astype(np.uint8)
        area = int(cv2.countNonZero(sub))

        # 1) Kích thước: không quá nhỏ (nhiễu), không quá lớn (cảnh lộn xộn)
        if area < MIN_INK_AREA or area > MAX_GROUP_FRAC * H * W:
            continue
        side = max(gw, gh)
        if side > MAX_GROUP_SIDE * max(H, W) or side < MIN_GROUP_SIDE * min(H, W):
            continue
        # 2) Hình dáng: gần vuông và mảnh (không đặc)
        if not (0.4 <= gw / gh <= 2.5):
            continue
        if area / float(gw * gh) > MAX_FILL:
            continue

        # 3) Có hai nét chéo dài, cắt nhau gần tâm cụm
        edges = cv2.Canny(sub, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=20,
                                minLineLength=25, maxLineGap=40)
        if lines is None:
            continue
        diag = float(np.hypot(gw, gh))
        segs = []
        for l in np.array(lines).reshape(-1, 4):
            sx1, sy1, sx2, sy2 = [int(v) for v in l]
            if np.hypot(sx2 - sx1, sy2 - sy1) >= MIN_STROKE_FRAC * diag:
                segs.append((sx1, sy1, sx2, sy2))
        segs.sort(key=lambda t: -np.hypot(t[2] - t[0], t[3] - t[1]))
        segs = segs[:30]

        found = None
        for a_i in range(len(segs)):
            for b_i in range(a_i + 1, len(segs)):
                sa, sb = segs[a_i], segs[b_i]
                diff = abs(seg_angle(sa) - seg_angle(sb))
                diff = min(diff, 180 - diff)
                if diff < MIN_CROSS_ANGLE:
                    continue
                if not (is_diagonal(seg_angle(sa)) and is_diagonal(seg_angle(sb))):
                    continue
                p = seg_intersect(sa, sb)
                if p is None:
                    continue
                if abs(p[0] - gw / 2) > CENTER_TOL * gw or abs(p[1] - gh / 2) > CENTER_TOL * gh:
                    continue
                found = (sa, sb, p)
                break
            if found:
                break

        if found:
            valid = True
            sa, sb, p = found
            for sg in (sa, sb):
                cv2.line(out, (sg[0] + gx, sg[1] + gy), (sg[2] + gx, sg[3] + gy), (0, 255, 0), 2)
            cv2.circle(out, (int(p[0] + gx), int(p[1] + gy)), 6, (0, 0, 255), 2)
            cv2.rectangle(out, (gx, gy), (gx + gw, gy + gh), (0, 255, 0), 1)
            break

    label = "VALID" if valid else "INVALID"
    color = (0, 200, 0) if valid else (0, 0, 255)
    cv2.putText(out, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    return valid, ink_area, out


# ---------- Routes ----------
@app.route("/")
def index():
    return send_from_directory(STATIC_DIR, "index.html")


@app.route("/captures/<path:name>")
def get_capture(name):
    return send_from_directory(CAP_DIR, name)


@app.route("/upload", methods=["POST"])
def upload():
    data = request.get_data()
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR) if data else None
    if img is None:
        return jsonify({"error": "Không đọc được ảnh"}), 400

    now = datetime.now()
    eid = now.strftime("%Y%m%d_%H%M%S_%f")[:-3]
    raw_name = f"{eid}_raw.jpg"
    res_name = f"{eid}_res.jpg"

    valid, ink_area, annotated = detect_x(img)
    cv2.imwrite(os.path.join(CAP_DIR, raw_name), img)
    cv2.imwrite(os.path.join(CAP_DIR, res_name), annotated)

    event = {
        "id": eid,
        "time": now.strftime("%H:%M:%S"),
        "date": now.strftime("%d/%m/%Y"),
        "valid": valid,
        "ink_area": ink_area,
        "raw": raw_name,
        "image": res_name,
    }
    with lock:
        events.append(event)
        del events[:-MAX_EVENTS]
        save_events(events)

    print(f"[{event['time']}] {'HỢP LỆ' if valid else 'KHÔNG HỢP LỆ'} (mực: {ink_area} px)")
    return jsonify({"valid": valid})


@app.route("/api/events")
def api_events():
    with lock:
        total = len(events)
        ok = sum(1 for e in events if e["valid"])
        recent = list(reversed(events[-30:]))
    return jsonify({
        "events": recent,
        "stats": {"total": total, "valid": ok, "invalid": total - ok},
    })


@app.route("/api/clear", methods=["POST"])
def api_clear():
    with lock:
        events.clear()
        save_events(events)
    for name in os.listdir(CAP_DIR):
        try:
            os.remove(os.path.join(CAP_DIR, name))
        except OSError:
            pass
    return jsonify({"ok": True})


if __name__ == "__main__":
    print("Mở trình duyệt: http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)
