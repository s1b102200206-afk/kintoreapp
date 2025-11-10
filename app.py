import streamlit as st
import cv2
import tempfile
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image, ImageDraw, ImageFont
import os

st.title("🏋️ スクワット姿勢解析アプリ（ダウンロード用）")

# --- モデル読み込み ---
@st.cache_resource
def load_model():
    model = hub.load("https://tfhub.dev/google/movenet/singlepose/thunder/4")
    return model

movenet = load_model()

def detect_keypoints(frame):
    input_image = tf.image.resize_with_pad(tf.expand_dims(frame, axis=0), 256, 256)
    input_image = tf.cast(input_image, dtype=tf.int32)
    outputs = movenet(input_image)
    keypoints = outputs['output_0'].numpy()[0,0,:,:]
    return keypoints

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

# --- モード選択 ---
mode = st.radio("解析モードを選択", ("浅めモード", "深めモード"))

# --- ファイルアップロード ---
uploaded_file = st.file_uploader("動画をアップロードしてください", type=["mp4","mov","avi"])

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(uploaded_file.read())
    tfile.close()

    cap = cv2.VideoCapture(tfile.name)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(3))
    height = int(cap.get(4))

    out_path = os.path.join(tempfile.gettempdir(), "squat_result.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    st.write("🔍 解析中です…")

    # 日本語フォント設定
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    font = ImageFont.truetype(font_path, 32)

    frame_idx = 0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        keypoints = detect_keypoints(img_rgb)

        left_hip = keypoints[11][:2] * [width, height]
        left_knee = keypoints[13][:2] * [width, height]
        left_ankle = keypoints[15][:2] * [width, height]

        angle = calculate_angle(left_hip, left_knee, left_ankle)

        # モード別コメント
        if mode == "浅めモード":
            comment = f"深め注意！ {int(angle)}°" if angle <= 90 else f"角度: {int(angle)}°"
        else:  # 深めモード
            comment = f"浅め注意！ {int(angle)}°" if angle >= 100 else f"角度: {int(angle)}°"

        # Pillowで描画
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)
        draw.text((50, 50), comment, font=font, fill=(255,0,0))
        frame_with_text = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        out.write(frame_with_text)

        st.text(f"解析中: {frame_idx}/{frame_count} フレーム")

    cap.release()
    out.release()

    st.success("✅ 解析完了！")

    # ダウンロードボタンのみ
    with open(out_path, "rb") as f:
        st.download_button(
            label="📥 解析動画をダウンロード",
            data=f,
            file_name="squat_result.mp4",
            mime="video/mp4"
        )


