# app.py
import streamlit as st
import torch
import timm
from torchvision import transforms
from PIL import Image

# ---------------- CONFIG ----------------
MODEL_PATH = "./checkpoints/best_vit_base_patch16_224.pth"
MODEL_NAME = "vit_base_patch16_224"
IMG_SIZE = 224

# Force CPU to avoid CUDA OOM
DEVICE = torch.device("cpu")

# Class names (must match training dataset order)
CLASS_NAMES = [
    "Pepper__bell___Bacterial_spot",
    "Pepper__bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Tomato_Bacterial_spot",
    "Tomato_Early_blight",
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold",
    "Tomato_Septoria_leaf_spot",
    "Tomato_Spider_mites_Two_spotted_spider_mite",
    "Tomato__Target_Spot",
    "Tomato__Tomato_YellowLeaf__Curl_Virus",
    "Tomato__Tomato_mosaic_virus",
    "Tomato_healthy"
]
# ----------------------------------------

# Load model safely on CPU
@st.cache_resource
def load_model():
    # Load checkpoint on CPU
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

    # Create model
    model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=len(CLASS_NAMES))
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.eval()  # set to evaluation mode
    return model

model = load_model()

# Image preprocessing
def preprocess_image(image: Image.Image):
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet normalization
                             std=[0.229, 0.224, 0.225])
    ])
    image = image.convert("RGB")
    img_tensor = transform(image).unsqueeze(0)  # add batch dimension
    return img_tensor  # keep on CPU

# Prediction function
def predict(image: Image.Image):
    tensor = preprocess_image(image)
    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, dim=1)
        return CLASS_NAMES[pred.item()], conf.item()

# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="Crop Disease Detector", page_icon="🌿", layout="centered")

st.title("🌿 Crop Disease Detection using Vision Transformer")
st.write("Upload a leaf image to detect the disease type.")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Leaf Image", use_column_width=True)
    st.write("Analyzing...")

    label, confidence = predict(image)
    st.success(f"**Prediction:** {label}")
    st.info(f"**Confidence:** {confidence*100:.2f}%")

    # Optional: additional info for healthy leaves
    if "healthy" in label.lower():
        st.balloons()
        st.write("The leaf appears to be healthy!")
    else:
        st.warning("Disease detected. Please check treatment recommendations.")

st.markdown("---")
st.caption("Developed by Najiya — Vision Transformer + Streamlit Integration")