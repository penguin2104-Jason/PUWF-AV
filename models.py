"""Model definitions used consistently by training, evaluation, and inference."""

from monai.networks import nets


MODEL_NAMES = ("unet", "resunet", "attentionunet")
NUM_CLASSES = 5


def build_model(model_name: str):
    """Build one of the three architectures evaluated in the paper."""
    name = model_name.lower().replace("-", "").replace("_", "")

    if name == "unet":
        return nets.UNet(
            spatial_dims=2,
            in_channels=3,
            out_channels=NUM_CLASSES,
            channels=(64, 128, 256, 512, 1024),
            strides=(2, 2, 2, 2),
            num_res_units=0,
            norm="batch",
            dropout=0.0,
        )

    if name == "resunet":
        return nets.UNet(
            spatial_dims=2,
            in_channels=3,
            out_channels=NUM_CLASSES,
            channels=(32, 64, 128, 256, 512),
            strides=(2, 2, 2, 2),
            num_res_units=2,
            norm="batch",
            dropout=0.0,
        )

    if name == "attentionunet":
        return nets.AttentionUnet(
            spatial_dims=2,
            in_channels=3,
            out_channels=NUM_CLASSES,
            channels=(32, 64, 128, 256, 512),
            strides=(2, 2, 2, 2),
            dropout=0.0,
        )

    raise ValueError(f"Unknown model '{model_name}'. Choose from: {', '.join(MODEL_NAMES)}")
