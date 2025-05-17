import streamlit as st
import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
import io
from pathlib import Path
from scipy.ndimage.interpolation import zoom
from inference import create_slice_montage

from inference import (
    get_model,
    ORGAN_LABELS,
)

st.set_page_config(
    page_title="Medical Image Segmentation Demo", page_icon="🔬", layout="wide"
)


# Cache the model loading to avoid reloading on every interaction
def load_segmentation_model(checkpoint_path, num_classes=9):
    """Load the segmentation model and cache it"""

    class Args:
        def __init__(self, num_classes):
            self.model_name = "msa2net"
            self.num_classes = num_classes
            self.img_size = 224

    args = Args(num_classes)

    # Load model
    model = get_model(args)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint)
    model.eval()

    return model, args


if "isic" not in st.session_state:
    st.session_state.isic, _ = load_segmentation_model(
        "msa2net_isic.pth", num_classes=2
    )

if "ct" not in st.session_state:
    st.session_state.ct, _ = load_segmentation_model("msa2net_ct.pth", num_classes=9)


def main():
    st.title("Medical Image Segmentation Demo")

    with st.sidebar:
        st.header("Segmentation Method")

        segmentation_method = st.radio("Select", ["CT", "ISIC"], index=0)

        st.divider()

    uploaded_file = st.file_uploader(
        "Choose an image file", type=["jpg", "jpeg", "png", "bmp"]
    )

    col1, col2 = st.columns(2)

    if uploaded_file is not None:
        with col1:
            st.subheader("Original Image")
            image = Image.open(uploaded_file)
            image = image.resize((224, 224), Image.LANCZOS)
            st.image(image, caption="Uploaded Image", width=512)

            if st.button(f"Run Segmentation with {segmentation_method}"):
                with st.spinner("Processing image..."):
                    # Load the model
                    model = st.session_state[segmentation_method.lower()]

                    image = np.array(image)

                    x_transforms = transforms.Compose(
                        [
                            transforms.ToTensor(),
                            transforms.Normalize([0.5], [0.5]),
                        ]
                    )
                    input = x_transforms(image).unsqueeze(0).float()

                    with torch.no_grad():
                        outputs = model(input)
                        out = torch.argmax(
                            torch.softmax(outputs, dim=1), dim=1
                        ).squeeze(0)
                        out = out.cpu().detach().numpy()

                    # Process the image
                    out = np.expand_dims(out, axis=0)
                    image = np.expand_dims(image, axis=0)

                    with col2:
                        st.subheader(f"Segmentation Result ({segmentation_method})")
                        fig_image, fig = create_slice_montage(
                            image,
                            out,
                            segmentation_type=segmentation_method,
                            output_path="./output",
                            return_fig=True,
                        )
                        st.pyplot(fig)

    st.subheader("Class Labels")
    if segmentation_method == "ISIC":
        isic_labels_bilingual = {
            "Index": [1],
            "English": ["Abnormal"],
            "Tiếng Việt": ["Bất thường"],
        }
        st.dataframe(isic_labels_bilingual)
    else:
        organ_labels_bilingual = {
            "Index": list(ORGAN_LABELS.keys()),
            "English": list(ORGAN_LABELS.values()),
            "Tiếng Việt": [
                "Lá lách",
                "Thận phải",
                "Thận trái",
                "Túi mật",
                "Gan",
                "Dạ dày",
                "Động mạch chủ",
                "Tụy",
            ],
        }
        st.dataframe(organ_labels_bilingual)


if __name__ == "__main__":
    main()
