import streamlit as st

# ===== Safe Imports with Error Handling =====
try:
    import torch
    import timm
    import librosa
    import numpy as np
    import pandas as pd
    import plotly.express as px
    import os
    import gdown
except ModuleNotFoundError as e:
    st.error(f"⚠️ A required library is missing: `{e.name}`")
    st.info(
        "If you're running this on **Streamlit Cloud**, check that your `requirements.txt` "
        "contains the correct versions of `torch`, `torchaudio`, and other dependencies.\n\n"
        "For CPU-only builds, your `requirements.txt` should look like:\n"
        "```\n"
        "torch==2.3.0+cpu\n"
        "torchaudio==2.3.0+cpu\n"
        "--extra-index-url https://download.pytorch.org/whl/cpu\n"
        "```\n"
    )
    st.stop()

# ===== Download Model from Google Drive if not present =====
def download_model():
    model_path = "best_model.pth"
    if not os.path.exists(model_path):
        with st.spinner("📥 Downloading model, please wait..."):
            url = "https://drive.google.com/uc?id=1vzoMPqS_rtlTONL0PNKB96pJiu7hwph2"
            gdown.download(url, model_path, quiet=False)
    return model_path

# ===== Create Model =====
def create_model():
    model = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=264, in_chans=1)
    return model

# ===== Load Model with Partial State Dict =====
@st.cache_resource
def load_model(model_path="best_model.pth"):
    model = create_model()
    checkpoint = torch.load(model_path, map_location=torch.device("cpu"))
    state_dict = checkpoint.get("state_dict", checkpoint)
    model_state_dict = model.state_dict()
    filtered_state_dict = {k: v for k, v in state_dict.items()
                           if k in model_state_dict and model_state_dict[k].shape == v.shape}
    model.load_state_dict(filtered_state_dict, strict=False)
    model.eval()
    return model

# ===== Audio to Mel Spectrogram =====
def preprocess_audio(audio_path):
    y, sr = librosa.load(audio_path, sr=32000)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=1024, hop_length=320, n_mels=224)
    mel = librosa.power_to_db(mel, ref=np.max)
    mel = (mel - mel.mean()) / (mel.std() + 1e-6)
    mel = torch.tensor(mel).unsqueeze(0).unsqueeze(0).float()
    if mel.shape[-1] < 224:
        mel = torch.nn.functional.pad(mel, (0, 224 - mel.shape[-1]))
    mel = mel[:, :, :, :224]
    return mel

# ===== Load Taxonomy & Metadata =====
@st.cache_data
def load_metadata():
    return pd.read_csv("taxonomy.csv")

# ===== Page Config =====
st.set_page_config(page_title="EcoSonicNet", page_icon="🔊", layout="wide")

# ===== CSS Styling =====
st.markdown(
    """
    <style>
    body {
        background-color: #F0F0F0;
        color: black;
    }
    .landing {
        background: url("https://plus.unsplash.com/premium_photo-1724864863815-1469c8b74711?w=1200") no-repeat center center;
        background-size: cover;
        height: 100vh;
        display: flex;
        justify-content: center;
        align-items: center;
        flex-direction: column;
    }
    .landing-overlay {
        background-color: rgba(255,255,255,0.7);
        padding: 50px;
        border-radius: 10px;
        text-align: center;
        width: 80%;
        max-width: 800px;
    }
    .landing img {
        max-width: 400px;
        border-radius: 10px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
        margin-top: 20px;
    }
    .content {
        padding: 50px;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        border-radius: 5px;
        padding: 10px 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ===== Sidebar Navigation =====
st.sidebar.title("🌿 EcoSonicNet")
page = st.sidebar.radio("Navigate", ["Home", "Contact Us"])

# ===== Home Page =====
if page == "Home":
    # Landing section
    st.markdown(
        """
        <div class="landing">
            <div class="landing-overlay">
                <h1 style='font-size:3em;'>Discover a new world of bioacoustics...</h1>
                <img src="https://plus.unsplash.com/premium_photo-1673216099037-061c15de6a23?w=800" alt="Bird Image">
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Content section below landing
    st.markdown("<div class='content'>", unsafe_allow_html=True)

    st.title("🔊 EcoSonicNet")
    st.subheader("Bird, Amphibian, and Insect Audio Species Predictor")

    st.markdown("""
    EcoSonicNet is an AI-powered tool to classify audio recordings of **birds, amphibians, and insects**.

    - Uses **Vision Transformers (ViT)** for sound classification  
    - Supports **research, conservation, and biodiversity monitoring**

    > *“Bringing AI to the heart of nature.”*
    """)

    uploaded_file = st.file_uploader("🎧 Upload Audio (.ogg, .wav, .mp3)", type=["ogg", "wav", "mp3"])

    if uploaded_file:
        st.audio(uploaded_file, format='audio/ogg')
        with st.spinner("🔍 Analyzing..."):
            try:
                model_path = download_model()  # ensure model is downloaded
                input_tensor = preprocess_audio(uploaded_file)
                model = load_model(model_path)
                with torch.no_grad():
                    output = model(input_tensor)
                    probs = torch.softmax(output, dim=1).cpu().numpy().flatten()
                    top5_idx = probs.argsort()[-5:][::-1]
                    top5_probs = probs[top5_idx]

                taxonomy = load_metadata()
                idx2label = {i: name for i, name in enumerate(sorted(taxonomy.primary_label.unique()))}
                top5_labels = [idx2label.get(i, f"Class {i}") for i in top5_idx]

                st.subheader("🔬 Top 5 Predictions")
                for i in range(5):
                    st.markdown(f"**{top5_labels[i]}** — {top5_probs[i]*100:.2f}%")

                fig = px.bar(
                    x=top5_labels, y=top5_probs,
                    labels={'x':'Species', 'y':'Confidence'},
                    title="Prediction Confidence",
                    color=top5_probs,
                    color_continuous_scale="Viridis"
                )
                st.plotly_chart(fig, use_container_width=True)

                st.markdown("### 🧬 Taxonomy Info")
                info_df = taxonomy[taxonomy.primary_label.isin(top5_labels)]
                st.dataframe(info_df.reset_index(drop=True))

                st.markdown("### 📊 Model Accuracy")
                st.info("The model achieves **76% accuracy** on test data.")
            except Exception as e:
                st.error(f"⚠️ Something went wrong while analyzing the audio: {e}")

    st.markdown("""
    ### 📞 Contact  
    - **Email:** ecosonicnet@example.com  
    - **Phone:** +91-9876543210
    """)

    st.markdown("</div>", unsafe_allow_html=True)

# ===== Contact Us Page =====
elif page == "Contact Us":
    st.title("📞 Contact Us")
    st.markdown("""
    We'd love to hear from you!

    - **Email:** ecosonicnet@example.com  
    - **Phone:** +91-9025976078
    """)

    st.image(
        "https://images.unsplash.com/photo-1522252234503-e356532cafd5?w=800",
        use_column_width=True,
        caption="Let's connect and build for nature."
    )

# ===== Footer =====
st.markdown("---")
st.markdown(
    "<center>© 2025 EcoSonicNet. All rights reserved.</center>",
    unsafe_allow_html=True
)
