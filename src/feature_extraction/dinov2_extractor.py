"""
==============================================================
DINOv2 MODEL LOADER
PRL Visibility Estimation Project
==============================================================
"""

import torch


# ============================================================
# MODEL CONFIGURATION
# ============================================================

MODEL_NAME = "dinov2_vitb14"


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(

    "cuda"

    if torch.cuda.is_available()

    else "cpu"

)


# ============================================================
# LOAD MODEL
# ============================================================

def load_dinov2():

    print("=" * 60)
    print("LOADING DINOv2 MODEL")
    print("=" * 60)

    print("\nChecking Device...")

    print("Device :", DEVICE)

    print("\nLoading Model...")

    model = torch.hub.load(

        "facebookresearch/dinov2",

        MODEL_NAME

    )

    model.to(DEVICE)

    model.eval()

    print("Model Loaded Successfully!")

    print("\nModel :", MODEL_NAME)

    return model


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    model = load_dinov2()

    print("\nTesting Model\n")

    print("\nModel Name :", MODEL_NAME)

    print("Evaluation Mode :", not model.training)

    print("Device :", DEVICE)