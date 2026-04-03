import streamlit as st
import numpy as np
import pandas as pd
import altair as alt
import cv2
import os
import tempfile

from srcs.DetectionAgent import DetectionAgent
from Transformation import ImgTransformator
from Augmentation import ImgAugmentator

# ── Constants ────────────────────────────────────────────────
DATA_DIR = "data/leaves/images"
PRETRAINED_DIR = "pretrained_agent"
CLASSES = sorted([
    d for d in os.listdir(DATA_DIR)
    if os.path.isdir(os.path.join(DATA_DIR, d))
]) if os.path.isdir(DATA_DIR) else []

TRANSFORMATIONS = [
    "gaussian_blur", "mask", "roi_objects",
    "pseudolandmarks", "spots_isolation", "background_removal",
]


# ── Helpers ──────────────────────────────────────────────────

def load_class_images(class_name, max_images=None):
    """Load original images (no augmented) for a class from DATA_DIR."""
    folder = os.path.join(DATA_DIR, class_name)
    files = sorted([
        f for f in os.listdir(folder)
        if f.upper().endswith(".JPG") and "_" not in os.path.splitext(f)[0]
    ])
    if max_images:
        files = files[:max_images]
    images = []
    for f in files:
        img = cv2.imread(os.path.join(folder, f))
        if img is not None:
            images.append((f, img))
    return images


def count_images_per_class(include_augmented=False):
    """Return {class_name: count} for all classes in DATA_DIR."""
    counts = {}
    for cls in CLASSES:
        folder = os.path.join(DATA_DIR, cls)
        files = os.listdir(folder)
        if not include_augmented:
            files = [f for f in files if "_" not in os.path.splitext(f)[0]]
        counts[cls] = len([f for f in files if f.upper().endswith(".JPG")])
    return counts


def load_agent(model_dir=PRETRAINED_DIR):
    """Load a DetectionAgent from a directory."""
    return DetectionAgent.load(model_dir)


def bgr_to_rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# ── Page config ──────────────────────────────────────────────
st.set_page_config(page_title="Leaffliction", page_icon="🍃", layout="centered")
st.title("🍃 Leaffliction")
col_caption, col_link = st.columns([4, 1])
col_caption.caption("Leaf disease classification using deep learning")
col_link.markdown(
    "[![GitHub](https://img.shields.io/badge/GitHub-repo-181717?logo=github)]"
    "(https://github.com/rsterin/Leaffliction)"
)

# ── Sidebar ──────────────────────────────────────────────────
with st.sidebar:
    st.header("Model settings")
    uploaded_model = st.file_uploader(
        "Upload custom model (optional)",
        type=["h5", "json", "pkl"],
        accept_multiple_files=True,
        help="Upload model.weights.h5, model_architecture.json, and agent.pkl. "
             "Leave empty to use the pretrained model.",
    )

# ── Tabs ─────────────────────────────────────────────────────
tab_info, tab_dist, tab_augment, tab_transform, tab_predict, tab_train = st.tabs([
    "Informations", "Distribution", "Augmentation",
    "Transformation", "Prediction", "Training",
])

# ── Informations ─────────────────────────────────────────────
with tab_info:
    st.subheader("What is this?")
    st.markdown(
        """
        This app classifies **leaf diseases** from photographs using a
        Convolutional Neural Network. It is trained on 42's subject dataset and can
        distinguish **8 classes** of Apple and Grape leaves — healthy and
        diseased.
        """
    )

    st.subheader("Classes")
    cols = st.columns(4)
    for i, cls in enumerate(CLASSES):
        parts = cls.split("_", 1)
        plant = parts[0]
        condition = parts[1].replace("_", " ") if len(parts) > 1 else ""
        cols[i % 4].markdown(f"- **{plant}** — {condition}")

    st.subheader("Pipeline")
    st.code(
        "Image → Preprocessing (transforms) → CNN → Softmax → Predicted class",
        language=None,
    )

    st.subheader("Model architecture")
    st.markdown(
        """
        - **Conv2D(16)** → BatchNorm → MaxPool → Dropout(0.1)
        - **Conv2D(32)** → BatchNorm → MaxPool → Dropout(0.2)
        - **Conv2D(64)** → BatchNorm → MaxPool → Dropout(0.25)
        - Flatten → **Dense(128)** → Dropout(0.5) → **Dense(8, softmax)**
        - Optimizer: Adam · Loss: categorical cross-entropy
        """
    )

    st.subheader("Quick start")
    st.markdown(
        """
        1. Go to **Prediction** to classify a leaf image using the pretrained model.
        2. Check **Distribution** to see dataset balance.
        3. Explore **Transformation** and **Augmentation** to preview image processing.
        4. Use **Training** to generate a CLI command for training your own model.
        """
    )

# ── Distribution ─────────────────────────────────────────────
with tab_dist:
    st.subheader("Dataset distribution")

    include_aug = st.checkbox("Include augmented images", value=False)
    counts = count_images_per_class(include_augmented=include_aug)
    df = pd.DataFrame({"Class": list(counts.keys()), "Count": list(counts.values())})

    col_pie, col_bar = st.columns(2)

    with col_pie:
        pie = alt.Chart(df).mark_arc(innerRadius=40).encode(
            theta=alt.Theta("Count:Q"),
            color=alt.Color("Class:N", legend=alt.Legend(title="Class")),
            tooltip=["Class", "Count"],
        ).properties(title="Class distribution", height=350)
        st.altair_chart(pie, width="stretch")

    with col_bar:
        bar = alt.Chart(df).mark_bar().encode(
            x=alt.X("Class:N", sort="-y", axis=alt.Axis(labelAngle=-45)),
            y=alt.Y("Count:Q"),
            color=alt.Color("Class:N", legend=None),
            tooltip=["Class", "Count"],
        ).properties(title="Images per class", height=350)
        st.altair_chart(bar, width="stretch")

    st.dataframe(df, hide_index=True, width="stretch")

# ── Prediction ───────────────────────────────────────────────
with tab_predict:
    st.subheader("Classify a leaf image")

    source_option = st.radio(
        "Image source", ["Upload image", "Sample from dataset"],
        horizontal=True,
    )

    images_to_predict = []

    if source_option == "Upload image":
        uploaded_files = st.file_uploader(
            "Upload leaf image(s)", type=["jpg", "jpeg", "png"],
            accept_multiple_files=True, key="predict_upload",
        )
        for uf in (uploaded_files or []):
            file_bytes = np.frombuffer(uf.read(), dtype=np.uint8)
            img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            if img_bgr is not None:
                images_to_predict.append((uf.name, img_bgr, None))
    else:
        sample_class = st.selectbox("Class", CLASSES, key="predict_class")
        sample_count = st.slider("Number of samples", 1, 20, 5, key="predict_count")
        class_imgs = load_class_images(sample_class, max_images=sample_count)
        images_to_predict = [(name, img, sample_class) for name, img in class_imgs]

    if st.button("Run prediction", type="primary", disabled=len(images_to_predict) == 0):
        # Load model
        agent = None
        if uploaded_model and len(uploaded_model) == 3:
            with tempfile.TemporaryDirectory() as tmpdir:
                for f in uploaded_model:
                    f.seek(0)
                    path = os.path.join(tmpdir, f.name)
                    with open(path, "wb") as out:
                        out.write(f.read())
                try:
                    agent = DetectionAgent.load(tmpdir)
                except Exception as e:
                    st.error(f"Failed to load custom model: {e}")
        if agent is None:
            agent = load_agent()

        correct, total = 0, 0
        results = []

        progress = st.progress(0, text="Predicting...")
        for i, (name, img_bgr, true_class) in enumerate(images_to_predict):
            prediction, transformed_imgs = agent.predict(img_bgr)
            is_correct = (prediction == true_class) if true_class else None
            if is_correct is not None:
                total += 1
                if is_correct:
                    correct += 1
            results.append((name, img_bgr, prediction, true_class, is_correct, transformed_imgs))
            progress.progress((i + 1) / len(images_to_predict), text=f"Predicting {name}")

        if total > 0:
            st.metric("Accuracy", f"{correct / total:.2%} ({correct}/{total})")

        for name, img_bgr, prediction, true_class, is_correct, transformed_imgs in results:
            border_color = (
                (0, 200, 0) if is_correct
                else (200, 0, 0) if is_correct is False
                else (128, 128, 128)
            )
            img_bordered = cv2.copyMakeBorder(
                img_bgr, 8, 8, 8, 8, cv2.BORDER_CONSTANT, value=border_color
            )

            label = f"**{name}** → `{prediction}`"
            if true_class:
                icon = "✅" if is_correct else "❌"
                label += f" {icon} (true: `{true_class}`)"

            with st.expander(label, expanded=True):
                n_cols = 1 + len(transformed_imgs)
                img_cols = st.columns(n_cols)
                img_cols[0].image(bgr_to_rgb(img_bordered), caption="Original", width=200)
                for j, t_img in enumerate(transformed_imgs):
                    t_name = agent.transformations[j] if j < len(agent.transformations) else f"Transform {j+1}"
                    img_cols[j + 1].image(bgr_to_rgb(t_img), caption=t_name, width=200)

# ── Transformation ───────────────────────────────────────────
with tab_transform:
    st.subheader("Image transformations preview")
    st.caption("Preview how transformations are applied to leaf images before training.")

    t_class = st.selectbox("Class", CLASSES, key="transform_class")
    t_type = st.selectbox(
        "Transformation",
        ["All"] + [t.replace("_", " ").title() for t in TRANSFORMATIONS],
        key="transform_type",
    )

    if st.button("Show transformations", key="transform_btn"):
        class_imgs = load_class_images(t_class, max_images=1)
        if not class_imgs:
            st.warning("No images found for this class.")
        else:
            name, img_bgr = class_imgs[0]
            transformer = ImgTransformator(super_background=False)

            transforms_to_show = TRANSFORMATIONS if t_type == "All" else [
                t_type.lower().replace(" ", "_")
            ]

            st.image(bgr_to_rgb(img_bgr), caption=f"Original — {name}", width=300)

            t_cols = st.columns(min(len(transforms_to_show), 3))
            for i, t in enumerate(transforms_to_show):
                try:
                    result = transformer.quick_use(img_bgr, t)
                    t_cols[i % 3].image(bgr_to_rgb(result), caption=t.replace("_", " ").title(), width=250)
                except Exception as e:
                    t_cols[i % 3].error(f"{t}: {e}")

# ── Augmentation ─────────────────────────────────────────────
with tab_augment:
    st.subheader("Image augmentation preview")
    st.caption("Preview augmentation techniques used to balance and expand the training dataset.")

    a_class = st.selectbox("Class", CLASSES, key="augment_class")
    a_type = st.selectbox(
        "Augmentation type",
        ["All", "Rotation", "Blur", "Contrast", "Scaling", "Illumination", "Projective"],
        key="augment_type",
    )

    if st.button("Show augmentations", key="augment_btn"):
        class_imgs = load_class_images(a_class, max_images=1)
        if not class_imgs:
            st.warning("No images found for this class.")
        else:
            name, img_bgr = class_imgs[0]

            img_struct = {a_class: {name: {"original": img_bgr}}}
            augmentator = ImgAugmentator(img_struct)

            aug_type = None if a_type == "All" else a_type
            augmentator.augment(augmentation=aug_type)

            aug_dict = augmentator.images_structure[a_class][name]

            st.image(bgr_to_rgb(img_bgr), caption=f"Original — {name}", width=300)

            aug_keys = [k for k in aug_dict.keys() if k != "original"]
            if aug_keys:
                a_cols = st.columns(min(len(aug_keys), 3))
                for i, key in enumerate(aug_keys):
                    a_cols[i % 3].image(
                        bgr_to_rgb(aug_dict[key]),
                        caption=key.replace("_", " ").title(),
                        width=250,
                    )
            else:
                st.info("No augmentations generated.")

# ── Training ─────────────────────────────────────────────────
with tab_train:
    st.subheader("Train your own model")
    st.caption(
        "Training requires GPU resources and cannot run in the browser. "
        "Configure your parameters below and copy the generated command to train locally."
    )

    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.markdown("#### Parameters")
        train_source = st.text_input("Data source folder", value="data/leaves/images", disabled=True)
        train_dest = st.text_input("Model save name", value="my_model")
        train_epochs = st.slider("Epochs", 1, 200, 10, key="train_epochs")
        train_transforms = st.multiselect(
            "Transformations applied during training",
            TRANSFORMATIONS,
            default=["gaussian_blur"],
        )

    with col_right:
        st.markdown("#### Generated command")

        transfo_args = " ".join(train_transforms) if train_transforms else ""
        cmd = f"python train.py --source {train_source} --destination {train_dest} --epochs {train_epochs}"
        if transfo_args:
            cmd += f" --transfo {transfo_args}"

        st.code(cmd, language="bash")

        st.markdown("#### Import an existing model")
        st.caption(
            "After training locally, upload the 3 model files to use them in the **Prediction** tab."
        )
        st.markdown(
            """
            Your model folder should contain:
            - `model.weights.h5`
            - `model_architecture.json`
            - `agent.pkl`

            Upload them in the **sidebar** under *Model settings*.
            """
        )
