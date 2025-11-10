import streamlit as st
import cv2
import tempfile
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import os

st.set_page_config(page_title="スクワット姿勢解析アプリ", layout="wide")
st.title("🏋️‍♂️ スクワット姿勢解析アプリ")
st.write("動画をアップロードすると、膝の角度を解析し、浅め・深めの注意を表示します。")

# --- モデル読み込み ---
@st.cache_resource
def load_model():
    model = hub.load("https://tfhub.dev/google/movenet/singlepose/thunder/4")
    movenet = model.signatures['serving_default']  # ← ここが重要
    return movenet

movenet = load_model()

# --- 姿勢推定 ---
def detect_keypoints(frame):
    input_image = tf.image.resize_with_pad(tf.expand_dims(frame, axis=0), 256, 256)
    input_image = tf.cast(input_image, dtype=tf.int32)
    outputs = movenet(input_image)
    keypoints = outputs['output_0'].numpy()[0,0,:,:]  # 17 keypoints
    return keypoints

# --- 膝角度計算 ---
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

# --- ファイルアップロード ---
uploaded_file = st.file_uploader("動画をアップロードしてください", type=["mp4","mov","avi"])
mode = st.radio("解析モードを選択", ("浅めモード", "深めモード"))

if uploaded_file is not None:
    # 一時ファイルに保存
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(uploaded_file.read())
    tfile.close()

    cap = cv2.VideoCapture(tfile.name)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 出力動画用
    out_path = os.path.join(tempfile.gettempdir(), "squat_result.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    stframe = st.empty()
    st.write("🔍 解析中です。少々お待ちください…")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        keypoints = detect_keypoints(img_rgb)

        # 左膝の座標取得
        left_hip = keypoints[11][:2] * [width, height]
        left_knee = keypoints[13][:2] * [width, height]
        left_ankle = keypoints[15][:2] * [width, height]

        angle = calculate_angle(left_hip, left_knee, left_ankle)

        # 判定
        if mode == "浅めモード":
            if angle <= 90:
                text = f"深め注意！ {int(angle)}°"
                color = (0,0,255)
            else:
                text = f"角度: {int(angle)}°"
                color = (0,255,0)
        else:  # 深めモード
            if angle > 90:
                text = f"浅め注意！ {int(angle)}°"
                color = (0,0,255)
            else:
                text = f"角度: {int(angle)}°"
                color = (0,255,0)

        # フレームに文字描画
        cv2.putText(frame, text, (50,100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
        out.write(frame)

        # 進捗表示（静止画で確認用）
        stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

    cap.release()
    out.release()

    st.success("✅ 解析完了！")

    # 結果動画を再生はせず、ダウンロードのみ
    with open(out_path, "rb") as f:
        st.download_button("📥 解析動画をダウンロード", data=f, file_name="squat_result.mp4", mime="video/mp4")
