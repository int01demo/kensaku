import os
import re
import numpy as np
import torch
import clip
from PIL import Image, UnidentifiedImageError
import pandas as pd
import requests
from io import BytesIO
import urllib.parse

Image.MAX_IMAGE_PIXELS = None

excel_path = r"C:\Users\inter01\Desktop\bcart_productsこぴー.csv"
df = pd.read_csv(excel_path, encoding='utf-8-sig')

product_ids = df.iloc[1:, 1]  # B列
product_names = df.iloc[1:, 2]  # C列
main_image_paths = df.iloc[1:, 29]  # AD列（メイン画像）

# サブ画像列（CI〜CN → インデックス80〜89）
sub_image_paths_list = [df.iloc[1:, i] for i in range(80, 90)]

save_dir = "saved_images"
os.makedirs(save_dir, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

features = []
valid_ids = []
valid_names = []
valid_filenames = []

def process_image(image_path, product_id, product_name, suffix=""):
    if not pd.notna(image_path):
        return

    image_path = str(image_path)
    safe_name = re.sub(r'[\\/:*?"<>|]', '_', product_name)
    filename = f"{product_id}_{safe_name}{suffix}.jpg"
    filepath = os.path.join(save_dir, filename)

    try:
        if image_path.startswith("http"):
            encoded_url = urllib.parse.quote(image_path, safe=':/')
            response = requests.get(encoded_url, timeout=10)
            if response.status_code == 200:
                image = Image.open(BytesIO(response.content)).convert("RGB")
                image.save(filepath)
                print(f"保存しました: {filepath}")
            else:
                print(f"画像取得失敗: {image_path} (Status: {response.status_code})")
                return
        elif os.path.exists(image_path):
            image = Image.open(image_path).convert("RGB")
            image.save(filepath)
            print(f"保存しました: {filepath}")
        else:
            print(f"画像ファイルが見つかりません: {image_path}")
            return

        image = Image.open(filepath)
        if image.width < 10 or image.height < 10:
            print(f"画像サイズが小さすぎるためスキップ: {filepath}")
            return

        image_input = preprocess(image).unsqueeze(0).to(device)
        with torch.no_grad():
            image_features = model.encode_image(image_input)
            image_features /= image_features.norm(dim=-1, keepdim=True)

        features.append(image_features.cpu().numpy())
        valid_ids.append(product_id)
        valid_names.append(product_name)
        valid_filenames.append(filename)

    except UnidentifiedImageError:
        print(f"画像読み込み失敗（形式不正）: {filepath}")
    except Exception as e:
        print(f"特徴量抽出失敗: {filename} → {type(e).__name__}: {e}")

# メイン画像処理
for pid, name, path in zip(product_ids, product_names, main_image_paths):
    process_image(path, str(pid), str(name), suffix="_main")

# サブ画像処理
for i, sub_paths in enumerate(sub_image_paths_list):
    for pid, name, path in zip(product_ids, product_names, sub_paths):
        process_image(path, str(pid), str(name), suffix=f"_sub{i+1}")

# 特徴量保存
if features:
    features_array = np.concatenate(features, axis=0)
    np.savez("image_features.npz",
                features=features_array,
                ids=np.array(valid_ids),
                names=np.array(valid_names),
                filenames=np.array(valid_filenames))
    print("CLIP特徴量を image_features.npz に保存しました。")
else:
    print("保存する特徴量がありません")