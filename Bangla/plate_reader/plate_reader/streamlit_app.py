"""Web app. Run with:  streamlit run streamlit_app.py"""
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from plate_reader import PlateReader

st.set_page_config(page_title="Bangla License Plate Reader", page_icon="🚗", layout="wide")


@st.cache_resource
def load_reader():
    return PlateReader()


st.title("Bangla License Plate Reader")
st.caption("Upload a vehicle photo. The app finds the plate and reads its characters.")

try:
    reader = load_reader()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

with st.sidebar:
    st.header("Settings")
    plate_conf = st.slider("Plate detection confidence", 0.1, 0.9, 0.4, 0.05)
    char_conf = st.slider("Character confidence", 0.05, 0.9,
                          float(reader.char_conf_default), 0.05)

file = st.file_uploader("Vehicle photo", type=["jpg", "jpeg", "png"])

if file:
    rgb = np.array(Image.open(file).convert("RGB"))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    with st.spinner("Reading plate..."):
        results = reader.read(bgr, plate_conf, char_conf)
    annotated = cv2.cvtColor(reader.annotate(bgr, results), cv2.COLOR_BGR2RGB)

    left, right = st.columns(2)
    left.image(rgb, caption="Input")
    right.image(annotated, caption="Result")

    if not results:
        st.warning("No plate found. Try lowering the plate detection confidence.")

    for i, r in enumerate(results, 1):
        st.subheader(f"Plate {i}")
        c1, c2 = st.columns([1, 2])
        c1.image(cv2.cvtColor(r["crop"], cv2.COLOR_BGR2RGB), caption="Cropped plate")
        c2.markdown(f"## {r['text_bn'] or r['text']}")
        c2.caption(f"English: {r['text']}")
        c2.progress(min(1.0, r["char_conf"]),
                    text=f"Average character confidence {r['char_conf']:.0%}")
        if r["char_conf"] < 0.5:
            c2.info("Low confidence. Check the result by eye.")

    ok, buf = cv2.imencode(".png", cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
    if ok:
        st.download_button("Download annotated image", buf.tobytes(),
                           file_name="plate_result.png", mime="image/png")
    if results:
        df = pd.DataFrame([{"plate": r["text_bn"] or r["text"], "english": r["text"],
                            "plate_conf": round(r["plate_conf"], 2),
                            "char_conf": round(r["char_conf"], 2)} for r in results])
        st.download_button("Download results (CSV)", df.to_csv(index=False).encode("utf-8-sig"),
                           file_name="plates.csv", mime="text/csv")
