import os
import io
import torch
import numpy as np
import pandas as pd
import pickle
from PIL import Image
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
from torchvision import transforms, models
import open_clip
from concurrent.futures import ThreadPoolExecutor
from functools import partial

# Detectar CPU ou GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Carregar modelos com cache
@st.cache_resource
def load_resnet_model():
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model.eval()
    return torch.nn.Sequential(*list(model.children())[:-1]).to(device)

@st.cache_resource
def load_clip_model():
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
    model.eval()
    return model.to(device), preprocess

resnet_model = load_resnet_model()
clip_model, clip_preprocess = load_clip_model()

# Transformações para o ResNet
resnet_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Extração de características com cache
@st.cache_data
def extract_features_cached(image_bytes, model_choice):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    if model_choice == "CLIP":
        image_tensor = clip_preprocess(image).unsqueeze(0).to(device)
        with torch.no_grad():
            features = clip_model.encode_image(image_tensor).squeeze().cpu().numpy()
    else:
        image_tensor = resnet_transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            features = resnet_model(image_tensor).squeeze().cpu().numpy()
    return features / np.linalg.norm(features)

# Carregamento de cache
@st.cache_data
def load_cached_features(cache_file):
    with open(cache_file, "rb") as f:
        return pickle.load(f)

# Extração individual de uma imagem
def extract_single_feature(path, model_choice):
    try:
        img = Image.open(path).convert("RGB")
        if model_choice == "CLIP":
            image_tensor = clip_preprocess(img).unsqueeze(0).to(device)
            with torch.no_grad():
                features = clip_model.encode_image(image_tensor).squeeze().cpu().numpy()
        else:
            image_tensor = resnet_transform(img).unsqueeze(0).to(device)
            with torch.no_grad():
                features = resnet_model(image_tensor).squeeze().cpu().numpy()
        features /= np.linalg.norm(features)
        return os.path.basename(path), features
    except Exception:
        return os.path.basename(path), None

# Função para extrair e armazenar características com pasta de cache separada
def cache_features(images_folder, model_choice):
    cache_dir = os.path.join(images_folder, ".cache")
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"features_{model_choice.lower()}.pkl")

    if os.path.exists(cache_file):
        return load_cached_features(cache_file)
    else:
        st.info(f"Nenhum cache encontrado para o modelo {model_choice}. Extraindo características…")

        feature_dict = {}
        image_files = [f for f in os.listdir(images_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        image_paths = [os.path.join(images_folder, f) for f in image_files]
        total_images = len(image_paths)
        progress_bar = st.progress(0)

        with ThreadPoolExecutor() as executor:
            for idx, result in enumerate(executor.map(partial(extract_single_feature, model_choice=model_choice), image_paths)):
                filename, features = result
                if features is not None:
                    feature_dict[filename] = features
                else:
                    st.warning(f"Falha ao processar {filename}")
                progress_bar.progress((idx + 1) / total_images)

        with open(cache_file, "wb") as f:
            pickle.dump(feature_dict, f)

        st.success(f"✅ Características extraídas e salvas para o modelo {model_choice} ({len(feature_dict)} imagens).")
        return feature_dict

# Busca por imagens semelhantes
def find_similar_images(target_features, images_folder, model_choice, top_k=5):
    cached_features = cache_features(images_folder, model_choice)
    similarities = []
    for filename, features in cached_features.items():
        similarity = cosine_similarity([target_features], [features])[0][0]
        similarities.append((filename, similarity))
    similarities.sort(key=lambda x: -x[1])
    return similarities[:top_k]

# Interface Streamlit
st.title("🔍 Buscador de Imagens Semelhantes")
st.markdown("Envie uma imagem-alvo e busque por imagens semelhantes usando **ResNet50**, **CLIP** ou compare os dois modelos.")

target_image_file = st.file_uploader("📤 Envie a imagem-alvo", type=["jpg", "png", "jpeg"])
images_folder = st.text_input("📁 Caminho para a pasta de imagens", value=r"")
model_choice = st.selectbox("🧠 Escolha o modelo para busca", options=["CLIP", "ResNet50"])
top_k = st.slider("🔢 Número de correspondências", 1, 10, 5)

compare_button = st.button("🔄 Comparar CLIP vs ResNet50")
search_button = st.button("Buscar com o Modelo Selecionado")

if target_image_file and os.path.exists(images_folder):
    target_image = Image.open(target_image_file).convert("RGB")
    st.image(target_image, caption="🖼️ Imagem Alvo", use_container_width=True)

    if search_button:
        with st.spinner("🔍 Buscando imagens semelhantes..."):
            target_features = extract_features_cached(target_image_file.getvalue(), model_choice)
            matches = find_similar_images(target_features, images_folder, model_choice, top_k=top_k)

        st.subheader(f"📊 Top {top_k} Correspondências ({model_choice}):")
        for name, score in matches:
            path = os.path.join(images_folder, name)
            st.image(path, caption=f"{name} (similaridade: {score:.3f})", width=300)

        match_data = pd.DataFrame(matches, columns=["Arquivo", "Similaridade"])
        csv = match_data.to_csv(index=False)
        st.download_button(
            label="📥 Baixar resultados como CSV",
            data=csv,
            file_name=f"imagens_semelhantes_{model_choice}.csv",
            mime="text/csv"
        )

    if compare_button:
        with st.spinner("🔄 Comparando modelos..."):
            resnet_features = extract_features_cached(target_image_file.getvalue(), "ResNet50")
            clip_features = extract_features_cached(target_image_file.getvalue(), "CLIP")

            resnet_matches = find_similar_images(resnet_features, images_folder, "ResNet50", top_k=top_k)
            clip_matches = find_similar_images(clip_features, images_folder, "CLIP", top_k=top_k)

        st.subheader(f"📌 Comparação: Top {top_k} Correspondências (CLIP vs ResNet50)")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 🤖 Resultados com CLIP")
            for name, score in clip_matches:
                path = os.path.join(images_folder, name)
                st.image(path, caption=f"{name} (similaridade: {score:.3f})", width=250)

        with col2:
            st.markdown("### 🧠 Resultados com ResNet50")
            for name, score in resnet_matches:
                path = os.path.join(images_folder, name)
                st.image(path, caption=f"{name} (similaridade: {score:.3f})", width=250)