import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from ultralytics import YOLO

# ============================================================
#  Colab-like torch.load safety patch
# ============================================================
def patch_torch_load_safe():
    import torch as _torch
    if getattr(_torch, "_safe_load_patched", False):
        return
    if not hasattr(_torch, "_orig_load"):
        _torch._orig_load = _torch.load
    def _safe_torch_load(*args, **kwargs):
        if "weights_only" not in kwargs:
            kwargs["weights_only"] = False
        if "map_location" not in kwargs:
            kwargs["map_location"] = "cpu"
        return _torch._orig_load(*args, **kwargs)
    _torch.load = _safe_torch_load
    _torch._safe_load_patched = True

patch_torch_load_safe()

# ============================================================
#  DigitCNN def (same as Colab)
# ============================================================
class DigitCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(64*6*4, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.flatten(1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def load_digit_model(path, device):
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(obj, nn.Module):
        model = obj
    elif isinstance(obj, dict):
        model = DigitCNN()
        if "state_dict" in obj and isinstance(obj["state_dict"], dict):
            model.load_state_dict(obj["state_dict"], strict=False)
        else:
            model.load_state_dict(obj, strict=False)
    else:
        raise RuntimeError("Unsupported DigitCNN format")
    model.to(device)
    model.eval()
    return model

# ============================================================
#  Global model cache (load once)
# ============================================================
_MODELS = {"loaded": False}

def _load_models(weights_dir: str):
    global _MODELS
    if _MODELS.get("loaded"):
        return _MODELS

    y1 = os.path.join(weights_dir, "yolo_card.pt")
    y2 = os.path.join(weights_dir, "yolo_nid.pt")
    d  = os.path.join(weights_dir, "digitcnn.pt")

    if not os.path.exists(y1): raise FileNotFoundError(f"Missing: {y1}")
    if not os.path.exists(y2): raise FileNotFoundError(f"Missing: {y2}")
    if not os.path.exists(d):  raise FileNotFoundError(f"Missing: {d}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    yolo_card = YOLO(y1)
    yolo_nid  = YOLO(y2)
    digit_model = load_digit_model(d, device=device)

    _MODELS = {
        "loaded": True,
        "device": device,
        "yolo_card": yolo_card,
        "yolo_nid": yolo_nid,
        "digit_model": digit_model,
    }
    return _MODELS

# ============================================================
#  Block B helpers (same as Colab)
# ============================================================
def _clamp(v, a, b):
    return max(a, min(b, v))

def _crop_xyxy(img, x1, y1, x2, y2, pad=0):
    H, W = img.shape[:2]
    x1 = _clamp(int(x1) - pad, 0, W - 1)
    y1 = _clamp(int(y1) - pad, 0, H - 1)
    x2 = _clamp(int(x2) + pad, x1 + 1, W)
    y2 = _clamp(int(y2) + pad, y1 + 1, H)
    return img[y1:y2, x1:x2].copy(), (x1, y1, x2, y2)

def _best_box_yolo(res, cls_name=None, min_conf=0.15):
    if res is None or res.boxes is None or len(res.boxes) == 0:
        return None
    boxes = res.boxes
    xyxy = boxes.xyxy.detach().cpu().numpy()
    conf = boxes.conf.detach().cpu().numpy()
    cls  = boxes.cls.detach().cpu().numpy().astype(int)
    names = res.names if hasattr(res, "names") else None

    best = None
    bestc = -1.0
    for i in range(len(xyxy)):
        c = float(conf[i])
        if c < min_conf:
            continue
        cid = int(cls[i])
        if cls_name is not None and names is not None:
            if names.get(cid, None) != cls_name:
                continue
        if c > bestc:
            x1, y1, x2, y2 = xyxy[i]
            best = (x1, y1, x2, y2, c, cid)
            bestc = c
    return best

# ============================================================
#  Block C logic (exactly your Colab code, turned into functions)
# ============================================================
def _iran_national_id_checksum_ok(code10: str) -> bool:
    if code10 is None or len(code10) != 10 or not code10.isdigit():
        return False
    if len(set(code10)) == 1:
        return False
    digits = list(map(int, code10))
    s = sum(digits[i] * (10 - i) for i in range(9))
    r = s % 11
    c = digits[9]
    return (r < 2 and c == r) or (r >= 2 and c == (11 - r))

def _to_gray(img_bgr):
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) if img_bgr.ndim == 3 else img_bgr

def _clahe(gray):
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(gray)

def _binarize_variants(gray):
    outs = []
    blur = cv2.GaussianBlur(gray, (3,3), 0)

    _, t = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    outs.append(t)
    outs.append(255 - t)

    a = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                              cv2.THRESH_BINARY, 25, 8)
    outs.append(a)
    outs.append(255 - a)
    return outs

def _clean_binary(bin_img):
    if np.mean(bin_img) > 127:
        bin_img = 255 - bin_img
    k1 = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
    k2 = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    x = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, k1, iterations=1)
    x = cv2.morphologyEx(x, cv2.MORPH_CLOSE, k2, iterations=1)
    return x

def _safe_deskew_minarearect(bin_img, max_angle=25):
    ys, xs = np.where(bin_img > 0)
    if len(xs) < 80:
        return bin_img, 0.0, False

    pts = np.column_stack([xs, ys]).astype(np.float32)
    rect = cv2.minAreaRect(pts)
    angle = rect[-1]
    if angle < -45:
        angle += 90

    if (not np.isfinite(angle)) or (abs(angle) > max_angle):
        return bin_img, 0.0, False

    H, W = bin_img.shape[:2]
    M = cv2.getRotationMatrix2D((W/2, H/2), angle, 1.0)
    rot = cv2.warpAffine(bin_img, M, (W, H), flags=cv2.INTER_NEAREST, borderValue=0)
    return rot, float(angle), True

def _cc_boxes(bin_img):
    num, labels, stats, _ = cv2.connectedComponentsWithStats((bin_img > 0).astype(np.uint8), connectivity=8)
    boxes = []
    H, W = bin_img.shape[:2]
    for i in range(1, num):
        x, y, w, h, area = stats[i]
        if area < max(20, (H*W)//5000):
            continue
        if h < max(8, H//8) and w > W//2:
            continue
        boxes.append((x, y, w, h, area))
    return boxes

def _merge_and_pick_10_boxes(bin_img, boxes):
    H, W = bin_img.shape[:2]
    if not boxes:
        return None

    hs = [b[3] for b in boxes]
    med_h = np.median(hs) if hs else 0
    good = []
    for (x,y,w,h,area) in boxes:
        if med_h > 0:
            if h < 0.45*med_h:
                continue
            if h > 2.2*med_h:
                continue
        good.append((x,y,w,h,area))
    boxes = good if len(good) >= 5 else boxes

    boxes = sorted(boxes, key=lambda b: b[0])

    if len(boxes) > 10:
        keep = sorted(boxes, key=lambda b: b[4], reverse=True)[:10]
        boxes = sorted(keep, key=lambda b: b[0])

    def split_box(box):
        x,y,w,h,area = box
        roi = bin_img[y:y+h, x:x+w]
        proj = (roi > 0).sum(axis=0).astype(np.int32)
        if w < 14:
            return [box]
        proj_s = cv2.GaussianBlur(proj.reshape(1,-1).astype(np.float32), (1, 9), 0).ravel()
        L = int(0.2*w); R = int(0.8*w)
        if R - L < 6:
            return [box]
        cut = int(L + np.argmin(proj_s[L:R]))
        if cut < 6 or w - cut < 6:
            return [box]
        b1 = (x, y, cut, h, int(area*cut/max(w,1)))
        b2 = (x+cut, y, w-cut, h, int(area*(w-cut)/max(w,1)))
        return [b1, b2]

    tries = 0
    while len(boxes) < 10 and tries < 30:
        idx = int(np.argmax([b[2] for b in boxes]))
        new_boxes = split_box(boxes[idx])
        if len(new_boxes) == 1:
            break
        boxes = boxes[:idx] + new_boxes + boxes[idx+1:]
        boxes = sorted(boxes, key=lambda b: b[0])
        tries += 1

    if len(boxes) != 10:
        return None

    refined = []
    for (x,y,w,h,area) in boxes:
        roi = bin_img[y:y+h, x:x+w]
        ys, xs = np.where(roi > 0)
        if len(xs) < 10:
            return None
        x0, x1 = xs.min(), xs.max()
        y0, y1 = ys.min(), ys.max()
        pad = 2
        rx0 = _clamp(x + x0 - pad, 0, W-1)
        ry0 = _clamp(y + y0 - pad, 0, H-1)
        rx1 = _clamp(x + x1 + pad + 1, rx0+1, W)
        ry1 = _clamp(y + y1 + pad + 1, ry0+1, H)
        refined.append((rx0, ry0, rx1-rx0, ry1-ry0, (rx1-rx0)*(ry1-ry0)))
    refined = sorted(refined, key=lambda b: b[0])
    return refined

def _make_digit_tensor_from_bin(bin_img, box, out_h=48, out_w=32):
    x,y,w,h,_ = box
    roi = bin_img[y:y+h, x:x+w]

    if np.mean(roi) > 127:
        roi = 255 - roi

    ys, xs = np.where(roi > 0)
    if len(xs) < 10:
        return None
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    roi = roi[y0:y1+1, x0:x1+1]
    roi = cv2.copyMakeBorder(roi, 4, 4, 4, 4, borderType=cv2.BORDER_CONSTANT, value=0)

    rh, rw = roi.shape[:2]
    scale = min(out_w / max(rw,1), out_h / max(rh,1))
    nw, nh = max(1, int(rw*scale)), max(1, int(rh*scale))
    resized = cv2.resize(roi, (nw, nh), interpolation=cv2.INTER_NEAREST)

    canvas = np.zeros((out_h, out_w), dtype=np.uint8)
    oy = (out_h - nh)//2
    ox = (out_w - nw)//2
    canvas[oy:oy+nh, ox:ox+nw] = resized

    t = torch.from_numpy(canvas.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0)
    return t

def _predict_digit(model, tensor, device):
    with torch.no_grad():
        tensor = tensor.to(device)
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0)
        conf, pred = torch.max(probs, dim=0)
        return int(pred.item()), float(conf.item())

def _score_candidate(digits, confs, checksum_ok):
    if len(digits) != 10 or len(confs) != 10:
        return -1e9
    mean_c = float(np.mean(confs))
    min_c  = float(np.min(confs))
    score = 2.0*mean_c + 1.5*min_c
    if checksum_ok:
        score += 0.7
    return score

ACCEPT_MIN_CONF  = 0.50
ACCEPT_MEAN_CONF = 0.80

def _is_acceptable(best_dict):
    if best_dict is None:
        return False
    confs = best_dict["confs"]
    if len(confs) != 10:
        return False
    return (float(np.min(confs)) >= ACCEPT_MIN_CONF) and (float(np.mean(confs)) >= ACCEPT_MEAN_CONF)

def _best_for_orientation(obgr, oname, digit_model, device):
    gray0 = _to_gray(obgr)
    grays = [
        ("g", gray0),
        ("clahe", _clahe(gray0)),
        ("blur", cv2.GaussianBlur(gray0, (3,3), 0)),
        ("clahe_blur", cv2.GaussianBlur(_clahe(gray0), (3,3), 0)),
    ]

    best = None
    for gname, g in grays:
        for b0 in _binarize_variants(g):
            b = _clean_binary(b0)
            b, ang, used = _safe_deskew_minarearect(b, max_angle=25)

            boxes0 = _cc_boxes(b)
            boxes10 = _merge_and_pick_10_boxes(b, boxes0)
            if boxes10 is None:
                continue

            digits, confs = [], []
            ok = True
            for bx in boxes10:
                tt = _make_digit_tensor_from_bin(b, bx, out_h=48, out_w=32)
                if tt is None:
                    ok = False
                    break
                d, c = _predict_digit(digit_model, tt, device)
                digits.append(d)
                confs.append(c)
            if not ok:
                continue

            code = "".join(str(d) for d in digits)
            chk = _iran_national_id_checksum_ok(code)
            score = _score_candidate(digits, confs, chk)

            if np.min(confs) < 0.30: score -= 0.3
            if np.mean(confs) < 0.55: score -= 0.2

            if best is None or score > best["score"]:
                best = {
                    "score": score,
                    "orientation": oname,
                    "gray_name": gname,
                    "deskew_angle": ang,
                    "deskew_used": used,
                    "code": code,
                    "digits": digits,
                    "confs": confs,
                    "checksum_ok": chk,
                    "bin": b,
                    "boxes": boxes10,
                    "src_bgr": obgr,
                }
    return best

def _run_block_c(nid_crop_bgr, digit_model, device):
    best0 = _best_for_orientation(nid_crop_bgr, "rot0", digit_model, device)
    if _is_acceptable(best0):
        return best0
    best180 = _best_for_orientation(cv2.rotate(nid_crop_bgr, cv2.ROTATE_180), "rot180", digit_model, device)
    return best180 if best180 is not None else best0

# ============================================================
#  Main entry: app calls this
# ============================================================
def run_pipeline(img_bgr, weights_dir="weights"):
    m = _load_models(weights_dir)
    yolo_card = m["yolo_card"]
    yolo_nid  = m["yolo_nid"]
    digit_model = m["digit_model"]
    device = m["device"]

    debug = {}

    # ----- Block B behavior (same as Colab) -----
    r1 = yolo_card.predict(img_bgr, verbose=False)[0]
    b1 = _best_box_yolo(r1, cls_name=None, min_conf=0.15)

    if b1 is None:
        card_crop = img_bgr.copy()
        card_xyxy = (0, 0, img_bgr.shape[1], img_bgr.shape[0])
        card_vis = img_bgr.copy()
    else:
        x1, y1, x2, y2, c, cid = b1
        card_crop, card_xyxy = _crop_xyxy(img_bgr, x1, y1, x2, y2, pad=8)
        card_vis = img_bgr.copy()
        cv2.rectangle(card_vis, (card_xyxy[0], card_xyxy[1]), (card_xyxy[2], card_xyxy[3]), (0,255,0), 2)
        cv2.putText(card_vis, f"card {c:.2f}", (card_xyxy[0], max(0, card_xyxy[1]-8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    debug["card_vis_rgb"] = cv2.cvtColor(card_vis, cv2.COLOR_BGR2RGB)
    debug["card_crop_rgb"] = cv2.cvtColor(card_crop, cv2.COLOR_BGR2RGB)

    r2 = yolo_nid.predict(card_crop, verbose=False)[0]
    b2 = _best_box_yolo(r2, cls_name="national_id_box", min_conf=0.15)

    if b2 is None:
        return {
            "code": None,
            "checksum_ok": False,
            "mean_conf": None,
            "min_conf": None,
            "picked": None,
            "notes": "national_id_box not found",
            "debug": debug
        }

    x1, y1, x2, y2, c, cid = b2
    nid_crop, nid_xyxy = _crop_xyxy(card_crop, x1, y1, x2, y2, pad=6)
    nid_vis = card_crop.copy()
    cv2.rectangle(nid_vis, (nid_xyxy[0], nid_xyxy[1]), (nid_xyxy[2], nid_xyxy[3]), (0,255,0), 2)
    cv2.putText(nid_vis, f"national_id_box {c:.2f}", (nid_xyxy[0], max(0, nid_xyxy[1]-8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    debug["nid_vis_rgb"] = cv2.cvtColor(nid_vis, cv2.COLOR_BGR2RGB)
    debug["nid_crop_rgb"] = cv2.cvtColor(nid_crop, cv2.COLOR_BGR2RGB)

    # ----- Block C behavior (same as Colab) -----
    best = _run_block_c(nid_crop, digit_model, device)

    if best is None:
        return {
            "code": None,
            "checksum_ok": False,
            "mean_conf": None,
            "min_conf": None,
            "picked": None,
            "notes": "Failed: could not reliably segment 10 digits",
            "debug": debug
        }

    # Debug visualization like colab
    vis = best["src_bgr"].copy()
    for i, bx in enumerate(best["boxes"]):
        x,y,w,h,_ = bx
        cv2.rectangle(vis, (x,y), (x+w, y+h), (0,255,0), 2)
        txt = f"{best['digits'][i]} {best['confs'][i]:.2f}"
        cv2.putText(vis, txt, (x, max(0, y-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    debug["chosen_vis_rgb"] = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
    debug["bin_used"] = best["bin"]

    confs = best["confs"]
    return {
        "code": best["code"],
        "checksum_ok": bool(best["checksum_ok"]),
        "mean_conf": float(np.mean(confs)),
        "min_conf": float(np.min(confs)),
        "picked": f"{best['orientation']} | {best['gray_name']} | deskew_used={best['deskew_used']} angle={best['deskew_angle']:.1f} | score={best['score']:.3f}",
        "notes": "OK",
        "debug": debug
    }
