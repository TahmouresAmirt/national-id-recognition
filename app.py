import streamlit as st
import cv2
import numpy as np
from pipeline import run_pipeline

st.set_page_config(page_title="NID Reader", layout="centered")
st.title("Iran National ID Reader (Local)")

st.caption("Upload an image OR take a webcam photo. Then press Run.")

col1, col2 = st.columns(2)
uploaded = col1.file_uploader("📁 Upload image", type=["jpg", "jpeg", "png"])
cam = col2.camera_input("📷 Webcam photo")

def _decode_to_bgr(file_bytes: bytes):
    arr = np.asarray(bytearray(file_bytes), dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)  # can be BGR/BGRA/GRAY
    if img is None:
        return None
    if img.ndim == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img

def _webcam_make_jpeg_like(bgr: np.ndarray, quality=95):
    # Streamlit webcam sometimes gives PNG/BGRA. We already made it BGR.
    # This step standardizes to a JPEG-like image (helps stability on some webcams).
    ok, enc = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        return bgr
    dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    return dec if dec is not None else bgr

img_bgr = None
src_label = None

if uploaded is not None:
    img_bgr = _decode_to_bgr(uploaded.read())
    src_label = "upload"
elif cam is not None:
    img_bgr = _decode_to_bgr(cam.read())
    if img_bgr is not None:
        img_bgr = _webcam_make_jpeg_like(img_bgr, quality=95)
    src_label = "webcam"

if img_bgr is not None:
    st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), caption=f"Input ({src_label})", use_container_width=True)

    if st.button("Run"):
        with st.spinner("Running..."):
            out = run_pipeline(img_bgr, weights_dir="weights")

        st.subheader("Result")
        st.json({
            "code": out.get("code"),
            "checksum_ok": out.get("checksum_ok"),
            "mean_conf": out.get("mean_conf"),
            "min_conf": out.get("min_conf"),
            "picked": out.get("picked"),
            "notes": out.get("notes"),
        })

        dbg = out.get("debug", {})
        if dbg.get("card_vis_rgb") is not None:
            st.image(dbg["card_vis_rgb"], caption="YOLOv1 result (card box)", use_container_width=True)
        if dbg.get("card_crop_rgb") is not None:
            st.image(dbg["card_crop_rgb"], caption="Card crop", use_container_width=True)
        if dbg.get("nid_vis_rgb") is not None:
            st.image(dbg["nid_vis_rgb"], caption="YOLOv2 result (national_id_box)", use_container_width=True)
        if dbg.get("nid_crop_rgb") is not None:
            st.image(dbg["nid_crop_rgb"], caption="national_id_box crop (nid_crop)", use_container_width=True)
        if dbg.get("chosen_vis_rgb") is not None:
            st.image(dbg["chosen_vis_rgb"], caption="Chosen candidate (digit boxes + preds)", use_container_width=True)
        if dbg.get("bin_used") is not None:
            st.image(dbg["bin_used"], caption="Binary used (white digits on black)", use_container_width=True)
else:
    st.info("Upload an image or take a webcam photo.")
