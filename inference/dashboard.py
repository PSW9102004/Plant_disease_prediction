import os
import argparse
import streamlit as st
from PIL import Image
import torch
import torch.nn.functional as F
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from models.base_cnn import BaseCNN
from data.transforms import val_transforms


def load_model(checkpoint_path, num_classes, device):
    model = BaseCNN(num_classes=num_classes, pretrained=False).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model


def predict(model, image, device):
    img = image.convert('RGB')
    input_tensor = val_transforms(img).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = F.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, dim=1)
    return pred.item(), conf.item()


def main():
    st.title("Plant Disease Detection Dashboard")

    # Sidebar inputs
    st.sidebar.header("Model Configuration")
    checkpoint_path = st.sidebar.text_input("Checkpoint Path", value="checkpoints/best_model.pth")
    data_dir = st.sidebar.text_input("Data Directory", value="data/raw/PlantDoc/PlantDoc-Dataset/train")
    num_classes = st.sidebar.number_input("Number of Classes", value=27, min_value=2)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(checkpoint_path, num_classes, device)

    st.sidebar.header("Image Selection")
    uploaded_file = st.sidebar.file_uploader("Upload an Image", type=["png", "jpg", "jpeg"])

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        pred, conf = predict(model, image, device)
        st.write(f"**Prediction:** Class {pred}   \n**Confidence:** {conf:.2f}")

    st.sidebar.header("Batch Prediction")
    dir_path = st.sidebar.text_input("Image Directory", value="")
    if dir_path and st.sidebar.button("Run Batch Prediction"):
        st.write(f"Batch predictions for directory: {dir_path}")
        files = [f for f in os.listdir(dir_path) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
        results = []
        for fname in files:
            img_path = os.path.join(dir_path, fname)
            image = Image.open(img_path)
            pred, conf = predict(model, image, device)
            results.append((fname, pred, conf))

        # Display results
        for fname, pred, conf in results:
            st.write(f"{fname}: Class {pred}, Confidence {conf:.2f}")

if __name__ == '__main__':
    main()
