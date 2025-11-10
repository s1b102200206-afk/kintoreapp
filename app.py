import streamlit as st

st.title("🎥 動画表示テストアプリ")

uploaded_file = st.file_uploader("動画ファイルをアップロードしてください", type=["mp4", "mov", "avi"])

if uploaded_file is not None:
    st.video(uploaded_file)
    st.success("✅ 動画が表示されました！")
