import streamlit as st
import numpy as np
import torch
import clip
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

data = np.load("image_features.npz")
features = data["features"]
ids = data["ids"]
names = data["names"]
filenames = data["filenames"]

st.title("画像検索デモ（CLIP）")

uploaded_file = st.file_uploader("検索したい画像をアップロードしてください", type=["jpg", "jpeg", "png"])

# プレースホルダーを作成
result_area = st.empty()

if uploaded_file is not None:
    # 検索処理
    image = Image.open(uploaded_file)
    image_input = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        query_feature = model.encode_image(image_input)
        query_feature /= query_feature.norm(dim=-1, keepdim=True)

    similarities = (features @ query_feature.cpu().numpy().T).squeeze()
    top_index = np.argmax(similarities)
    top_indices = np.argsort(similarities)[::-1][:5]

    # ✅ プレースホルダーでUIを再構築
    with result_area.container():
        col_left, col_right = st.columns([1, 2])

        # 左側：検索画像
        with col_left:
            st.image(image, caption="検索画像", use_container_width=True)

        # 右側：最も類似した商品
        with col_right:
            st.markdown("### ✅ 最も類似した商品")
            st.write(f"**ID:** {ids[top_index]}")
            st.write(f"**商品名:** {names[top_index]}")
            st.write(f"**類似度:** {similarities[top_index]:.4f}")

            try:
                st.image(Image.open(f"saved_images/{filenames[top_index]}"), caption="類似商品画像", width=250)
            except FileNotFoundError:
                st.warning("画像が見つかりません")

            st.markdown("### 🔍 類似した商品（上位5件）")
            cols = st.columns(5)
            for i, idx in enumerate(top_indices):
                with cols[i]:
                    try:
                        st.image(Image.open(f"saved_images/{filenames[idx]}"), width=120)
                    except FileNotFoundError:
                        st.warning("画像なし")
                    st.write(f"**ID:** {ids[idx]}")
                    st.write(names[idx])
                    st.write(f"{similarities[idx]:.4f}")