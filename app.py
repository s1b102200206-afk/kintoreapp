import streamlit as st
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import cv2
import tempfile

# --- MoveNet読み込み ---
@st.cache_resource
def load_movenet():
    model = hub.load("https://tfhub.dev/google/movenet/singlepose/lightning/4")
    return model.signatures['serving_default']

movenet = load_movenet()

# --- 角度計算 ---
def calculate_angle(a, b):
    a, b = np.array(a), np.array(b)
    vertical = np.array([0, -1])
    spine = a - b
    cosine_angle = np.dot(spine, vertical) / (np.linalg.norm(spine)*np.linalg.norm(vertical)+1e-6)
    return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))

# --- フレーム解析 ---
def analyze_frame(frame, mode="shallow"):
    orig = frame.copy()
    img_resized = tf.image.resize_with_pad(tf.convert_to_tensor(frame), 192, 192)
    input_img = tf.expand_dims(tf.cast(img_resized, dtype=tf.int32), axis=0)
    keypoints = movenet(input_img)['output_0'].numpy()[0,0,:,:]
    h, w, _ = orig.shape

    points = {}
    kp_idx = {"left_shoulder":5,"right_shoulder":6,"left_hip":11,"right_hip":12,
              "left_knee":13,"right_knee":14,"left_ankle":15,"right_ankle":16}
    for name, idx in kp_idx.items():
        points[name] = (keypoints[idx][1]*w, keypoints[idx][0]*h, keypoints[idx][2])
    conf_thresh = 0.1

    def angle(a,b,c):
        a,b,c = np.array(a[:2]), np.array(b[:2]), np.array(c[:2])
        ba, bc = a-b, c-b
        return np.degrees(np.arccos(np.clip(np.dot(ba,bc)/(np.linalg.norm(ba)*np.linalg.norm(bc)+1e-6), -1.0, 1.0)))

    knee_angle = angle(points["left_hip"], points["left_knee"], points["left_ankle"])
    mid_shoulder = ((points["left_shoulder"][0]+points["right_shoulder"][0])/2,
                    (points["left_shoulder"][1]+points["right_shoulder"][1])/2)
    mid_hip = ((points["left_hip"][0]+points["right_hip"][0])/2,
               (points["left_hip"][1]+points["right_hip"][1])/2)
    back_angle = calculate_angle(mid_shoulder, mid_hip)

    if mode=="shallow":
        knee_comment = "深め注意" if knee_angle <= 90 else "浅めOK" if knee_angle > 100 else "少し浅め"
    else:
        knee_comment = "深めOK" if knee_angle < 80 else "もう少し深く" if knee_angle < 100 else "浅すぎ"
    back_comment = "背中まっすぐ" if back_angle < 15 else f"背中曲がり({int(back_angle)}°)"

    for pt in points.values():
        if pt[2] > conf_thresh:
            cv2.circle(orig, tuple(map(int, pt[:2])), 5, (0,255,0), -1)
    bones = [("left_shoulder","left_hip"),("right_shoulder","right_hip"),
             ("left_hip","left_knee"),("left_knee","left_ankle"),
             ("right_hip","right_knee"),("right_knee","right_ankle")]
    for a,b in bones:
        if points[a][2] > conf_thresh and points[b][2] > conf_thresh:
            cv2.line(orig, tuple(map(int, points[a][:2])), tuple(map(int, points[b][:2])), (255,0,0), 2)
    cv2.line(orig, tuple(map(int, mid_shoulder)), tuple(map(int, mid_hip)), (0,0,255), 2)

    cv2.putText(orig, f"下半身: {knee_comment}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 2)
    cv2.putText(orig, f"上半身: {back_comment}", (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

    return cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)

# --- Streamlit UI ---
st.title("スクワット解析アプリ（軽量版）")
mode = st.radio("解析モードを選択", ("shallow","deep"))
uploaded_file = st.file_uploader("動画をアップロードしてください", type=["mp4","mov","avi"])

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(uploaded_file.read())
    tfile.close()

    cap = cv2.VideoCapture(tfile.name)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    max_frames = 300  # フレーム制限
    frames = []
    progress_text = st.empty()
    
    for i in range(min(frame_count, max_frames)):
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (320,240))
        frames.append(analyze_frame(frame, mode))
        progress_text.text(f"解析中: {i+1}/{min(frame_count, max_frames)} フレーム")
    cap.release()

    # 動画として保存
    out_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_file.name, fourcc, fps, (320,240))
    for f in frames:
        out.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    out.release()

    st.success("動画解析完了！")
    st.video(out_file.name)
    with open(out_file.name, "rb") as f:
        st.download_button("解析動画をダウンロード", f, file_name="squat_analysis.mp4", mime="video/mp4")

   

