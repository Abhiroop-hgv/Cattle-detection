"""Hackathon demo: register cattle from photos, identify them, and answer "which cow is it?" for a
health alert.   Run:  streamlit run demo_app.py

The health alerts here are SIMULATED (you pick the alert type); the identification is real.
"""
import shutil
from pathlib import Path

import streamlit as st
from PIL import Image, ImageDraw, ImageFont, ImageOps

from cattle_id.config import Config
from cattle_id.service import CattleID

STORE = Path("demo_store")
st.set_page_config(page_title="Cattle ID", page_icon="🐄", layout="wide")


@st.cache_resource(show_spinner="Loading models...")
def service() -> CattleID:
    STORE.mkdir(exist_ok=True)
    cfg = Config(index_path=str(STORE / "index.npz"), registry_path=str(STORE / "registry.json"))
    return CattleID(cfg)


def load(files) -> list[Image.Image]:
    return [ImageOps.exif_transpose(Image.open(f)).convert("RGB") for f in files]


def annotated(img: Image.Image) -> Image.Image:
    """Photo with the detector's boxes: green = face, yellow = ear tag."""
    out = img.copy()
    d = ImageDraw.Draw(out)
    w = max(2, img.width // 200)
    for det in service().detector.detect(img):
        d.rectangle(det.box, outline="lime" if det.label == "cow_face" else "yellow", width=w)
    return out


def show_result(res: dict) -> None:
    s = res.get("status")
    det = res.get("details") or {}
    if s == "found":
        st.success(f"**{det.get('name', res['cattle_id'])}** — id `{res['cattle_id']}`")
        st.write({k: v for k, v in det.items()})
        bits = [f"evidence: {res.get('evidence', 'face')}"]
        if res.get("score") is not None:
            bits.append(f"face similarity {res['score']:.2f}")
        if res.get("tag_number"):
            bits.append(f"tag read: {res['tag_number']}")
        st.caption(" · ".join(bits) + (f" · {res['note']}" if res.get("note") else ""))
    elif s == "not_found":
        st.warning(res["message"])
        if res.get("best_score") is not None:
            st.caption(f"closest registered cow scored {res['best_score']:.2f} (needs ≥ {service().cfg.match_threshold})")
    elif s == "conflict":
        st.error(res["message"])
        st.write({"face says": res["face_candidate"]["details"], "tag says": res["tag_candidate"]["details"]})
    elif s == "duplicate":
        st.error(res["message"])
        st.write(res.get("details"))
    elif s == "rejected":
        st.info(res.get("message") or f"Rejected: {res.get('reason')}")
        st.caption(str(res.get("rejections") or res.get("skipped") or ""))
    else:
        st.write(res)


def group_view(img: Image.Image, results: list[dict]) -> Image.Image:
    """Every detected cow boxed and labelled: green = identified, orange = not registered,
    red = face and tag disagree."""
    out = img.copy()
    d = ImageDraw.Draw(out)
    size = max(14, img.width // 45)
    try:
        font = ImageFont.load_default(size=size)
    except TypeError:  # older Pillow
        font = ImageFont.load_default()
    for r in results:
        colour = {"found": "lime", "conflict": "red"}.get(r["status"], "orange")
        label = (r["details"] or {}).get("name", r["cattle_id"]) if r["status"] == "found" else (
            "CHECK" if r["status"] == "conflict" else "not registered")
        x0, y0, x1, y1 = r["box"]
        d.rectangle(r["box"], outline=colour, width=max(3, img.width // 250))
        d.text((x0, max(0, y0 - size - 4)), label, fill=colour, font=font, stroke_width=2, stroke_fill="black")
    return out


svc = service()
st.title("🐄 Cattle ID — which cow is it?")
reg_tab, id_tab, group_tab, alert_tab, herd_tab = st.tabs(
    ["Register", "Identify", "Group photo", "Health alert", "Registered cattle"])

with reg_tab:
    st.write("Upload **3 or more photos** of one cow's face (ear tag visible if possible).")
    c1, c2 = st.columns([1, 2])
    with c1:
        name = st.text_input("Name")
        height = st.number_input("Height (cm)", 0, 250, 130)
        tag = st.text_input("Ear-tag number (optional)")
    with c2:
        files = st.file_uploader("Photos", type=["jpg", "jpeg", "png"], accept_multiple_files=True, key="reg")
    if st.button("Register", type="primary", disabled=not (files and name)):
        with st.spinner("Registering..."):
            st.session_state["reg_res"] = svc.register_frames(
                load(files), {"name": name, "height_cm": height}, tag or None)
        st.session_state["reg_name"] = name
    res = st.session_state.get("reg_res")
    if res:
        if res["status"] == "registered":
            st.success(f"Registered **{st.session_state.get('reg_name')}** as `{res['cattle_id']}`")
            if res.get("overrode_duplicate_of"):
                st.caption(f"Registered as a different animal from `{res['overrode_duplicate_of']}` (confirmed by you).")
            if res.get("tag_check"):
                st.warning(res["tag_check"]["message"] + f" (photos show {res['tag_check']['read_from_photos']})")
        else:
            show_result(res)
            if res["status"] == "duplicate" and res.get("reason") != "tag_already_registered":
                st.write("If these really are two different cows (look-alikes), you can confirm and register anyway.")
                if st.button("These are different cows - register anyway") and files and name:
                    st.session_state["reg_res"] = svc.register_frames(
                        load(files), {"name": name, "height_cm": height}, tag or None, force=True)
                    st.session_state["reg_name"] = name
                    st.rerun()
        if res.get("skipped_frames"):
            st.caption(f"skipped photos: {res['skipped_frames']}")

with id_tab:
    files = st.file_uploader("Photo(s) of one cow", type=["jpg", "jpeg", "png"],
                             accept_multiple_files=True, key="idf")
    if files:
        imgs = load(files)
        st.image([annotated(i) for i in imgs], width=260, caption=["green = face, yellow = ear tag"] * len(imgs))
        with st.spinner("Identifying..."):
            show_result(svc.identify_frames(imgs))

with group_tab:
    st.write("One photo with **several cows**: every face is found and matched on its own. "
             "Cows whose face is turned away are simply not listed.")
    gf = st.file_uploader("Group photo", type=["jpg", "jpeg", "png"], key="grp")
    if gf:
        img = load([gf])[0]
        with st.spinner("Identifying every cow..."):
            results = svc.identify_frame(img, use_tags=True)
        if not results:
            st.info("No cow face found in this photo.")
        else:
            st.image(group_view(img, results), use_container_width=True)
            names = [(r["details"] or {}).get("name", r["cattle_id"]) for r in results if r["status"] == "found"]
            st.write(f"{len(results)} cows seen - identified: {', '.join(names) or 'none'}; "
                     f"not registered: {sum(r['status'] == 'not_found' for r in results)}; "
                     f"to check: {sum(r['status'] == 'conflict' for r in results)}")
            st.caption("A single photo is weaker than several: a wrong match is possible, so treat names as suggestions.")

with alert_tab:
    st.write("A health model (posture / acoustic) raises an alert. Upload a camera frame from that "
             "moment and we tell the farmer **which cow it is**.  *(The alert itself is simulated.)*")
    kind = st.selectbox("Alert", ["Lying down for more than 3 hours", "Abnormal coughing detected",
                                  "Standing time dropped sharply", "Not eating (no feeding visits)"])
    files = st.file_uploader("Camera frame(s) at the time of the alert", type=["jpg", "jpeg", "png"],
                             accept_multiple_files=True, key="alert")
    if files:
        res = svc.identify_frames(load(files))
        if res["status"] == "found":
            d = res["details"] or {}
            tag_txt = f", tag {d['tag_number']}" if d.get("tag_number") else ""
            st.error(f"🚨 **{kind}** — **{d.get('name', res['cattle_id'])}** (`{res['cattle_id']}`{tag_txt}). "
                     "Please check this cow.")
        elif res["status"] == "conflict":
            st.error(f"🚨 **{kind}** — cow uncertain: face and ear tag disagree. Please check.")
        else:
            st.error(f"🚨 **{kind}** — unidentified cow. " + res.get("message", "Please check the pen."))
        with st.expander("Details"):
            show_result(res)

with herd_tab:
    data = svc.registry.data if hasattr(svc.registry, "data") else {}
    st.write(f"{len(data)} registered")
    if data:
        st.table([{"id": k, **v} for k, v in data.items()])
    if st.button("Reset demo (delete all registered cattle)"):
        service.clear()
        shutil.rmtree(STORE, ignore_errors=True)
        st.rerun()
