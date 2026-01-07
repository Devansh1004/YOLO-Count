from typing import List, Tuple
from PIL import Image
import os
import torch
import numpy as np


def number_parameters(model):
    return sum(p.numel() for p in model.parameters())


def aspect_based_center_mask(
    hw_list: List[Tuple[int, int]], save_path: str = None
) -> torch.Tensor:
    """
    Generate a binary mask based on image aspect ratio indicating the valid area
    after padding and resizing to 80x80.

    Args:
        hw_list (List[Tuple[int, int]]): list of (height, width) tuples for images
        save_path (str, optional): path to save visualization. If None, do not save

    Returns:
        torch.Tensor: binary mask tensor with shape (batch_size, 1, 80, 80)
    """
    batch_size = len(hw_list)
    masks = torch.zeros(batch_size, 1, 80, 80)

    if isinstance(hw_list, torch.Tensor):
        if hw_list.dim() == 2 and hw_list.shape[1] == 2:
            hw_list = hw_list.tolist()  # convert to List[List[int]]
        elif hw_list.dim() == 1:
            hw_list = torch.unsqueeze(hw_list, dim=0).tolist()

    if isinstance(hw_list, tuple):
        hw_list = [hw_list]

    for i, (h, w) in enumerate(hw_list):
        if h <= 0 or w <= 0:
            continue

        # Compute aspect ratio
        aspect_ratio = float(w) / float(h)

        # Compute the effective region size after resize
        if aspect_ratio >= 1:  # wide image
            new_h = int(np.floor(80 / aspect_ratio))
            new_h_ceil = int(np.ceil(80 / aspect_ratio))
            new_w = 80
        else:  # tall image
            new_h = 80
            new_w = int(np.floor(80 * aspect_ratio))
            new_w_ceil = int(np.ceil(80 * aspect_ratio))

        # Compute padding start positions to ensure center alignment
        start_h = (80 - new_h_ceil if aspect_ratio >= 1 else 80 - new_h) // 2
        start_w = (80 - new_w_ceil if aspect_ratio < 1 else 80 - new_w) // 2

        # Fill mask (use ceil to cover all possible pixels)
        end_h = start_h + (new_h_ceil if aspect_ratio >= 1 else new_h)
        end_w = start_w + (new_w_ceil if aspect_ratio < 1 else new_w)
        masks[i, 0, start_h:end_h, start_w:end_w] = 1

    if save_path is not None:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(8, 8))
        plt.imshow(masks[0, 0].numpy(), cmap="viridis")
        plt.colorbar()
        plt.axis("off")
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0.1)
        plt.close()
        print(f"save mask to {save_path}")

    return masks


def tis(image, save_path, mode="RGB"):
    """
    Converts an image to a torch.Tensor, checks for normalization, rearranges dimensions if necessary,
    and saves the image.

    Args:
    image: An image that can be converted to a tensor. It can be a 3D or 4D array.
    save_path: Path where the image will be saved.
    mode: Color mode of the image ('RGB' or 'BGR').
    """
    if "/" in save_path:
        save_dir = "/".join(save_path.split("/")[:-1])
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

    # Force convert image to torch.Tensor
    if not isinstance(image, torch.Tensor):
        image = torch.tensor(image, dtype=torch.float32)

    if isinstance(image, torch.Tensor):
        image = image.clone().cpu()

    # Check whether image data is normalized
    if image.max() > 1:
        image = image / 255.0  # Assume image data range is 0-255

    # Check dimensions and adjust order if necessary
    if image.dim() == 4:  # Assume dimensions are [1, C, H, W]
        if image.shape[0] != 1:
            raise ValueError("Dimension 4D but shape[0] not 1")
        image = image.squeeze(0)  # Assume single image, remove batch dimension
        if image.shape[0] == 3:  # C, H, W
            pass  # correct dimension order
        elif image.shape[2] == 3:  # H, W, C
            image = image.permute(2, 0, 1)  # rearrange to C, H, W
        else:
            raise ValueError("Invalid image shape, expected 3 channels")
    elif image.dim() == 3:
        if image.shape[0] == 3:  # C, H, W
            pass  # correct dimension order
        elif image.shape[2] == 3:  # H, W, C
            image = image.permute(2, 0, 1)  # rearrange to C, H, W
        else:
            raise ValueError("Invalid image shape, expected 3 channels")
    else:
        raise ValueError("Invalid image dimensions, expected 3D or 4D tensor")

    # Handle color mode
    if mode.upper() == "BGR":
        image = image[[2, 1, 0], :, :]  # BGR to RGB

    # Convert tensor to PIL image for saving
    image = (
        (image.clamp(0, 1) * 255).byte().permute(1, 2, 0).numpy()
    )  # convert to H, W, C format
    image = Image.fromarray(image)
    image.save(save_path)


def gtis(image, save_path):
    """
    Converts an image to a torch.Tensor, checks for normalization, rearranges dimensions if necessary,
    and saves the image as a grayscale image.

    Args:
    image: An image that can be converted to a tensor. It can be a 2D, 3D, or 4D array.
    save_path: Path where the image will be saved.
    """

    if "/" in save_path:
        save_dir = "/".join(save_path.split("/")[:-1])
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

    # Force image to be converted to torch.Tensor
    if not isinstance(image, torch.Tensor):
        image = torch.tensor(image, dtype=torch.float32)

    if isinstance(image, torch.Tensor):
        image = image.clone().cpu()

    # Check if image data is normalized
    if image.max() > 1:
        image = image / 255.0  # Assume image data range is 0-255

    # Check dimensions and adjust order if necessary
    if image.dim() == 4:  # Assuming dimensions are [1, C, H, W]
        if image.shape[0] != 1:
            raise ValueError("Dimension 4D but shape[0] not 1")
        image = image.squeeze(
            0
        )  # Assume processing a single image, remove batch dimension
        if image.shape[0] == 1:  # C, H, W
            pass  # Correct dimension order
        elif image.shape[2] == 1:  # H, W, C
            image = image.permute(2, 0, 1)  # Rearrange to C, H, W
        else:
            raise ValueError("Invalid image shape, expected 1 channel")
    elif image.dim() == 3:
        if image.shape[0] == 1:  # C, H, W
            pass  # Correct dimension order
        elif image.shape[2] == 1:  # H, W, C
            image = image.permute(2, 0, 1)  # Rearrange to C, H, W
        else:
            raise ValueError("Invalid image shape, expected 1 channel")
    elif image.dim() == 2:  # Assuming dimensions are H, W
        image = image.unsqueeze(0)  # Convert to 1, H, W for consistency
    else:
        raise ValueError("Invalid image dimensions, expected 2D, 3D or 4D tensor")

    # Convert tensor to PIL image to save it
    image = (
        (image.clamp(0, 1) * 255).byte().squeeze(0).numpy()
    )  # Convert to H, W format
    image = Image.fromarray(image, mode="L")
    image.save(save_path)


def viridis(image, save_path):
    """
    Convert the input image to a viridis colormap and save it with a colorbar.
    Supports 2D, 3D (1, H, W), and 4D (1, 1, H, W) input formats.

    Args:
        image: input image, can be a numpy array or torch.Tensor
        save_path: path to save the output
    """
    import matplotlib.pyplot as plt

    # Create save directory
    if "/" in save_path:
        save_dir = "/".join(save_path.split("/")[:-1])
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

    # Convert to torch.Tensor
    if not isinstance(image, torch.Tensor):
        image = torch.tensor(image, dtype=torch.float32)

    if isinstance(image, torch.Tensor):
        image = image.clone().cpu()

    # Check if normalization is needed
    if image.max() > 1:
        image = image / 255.0

    # Handle dimensions
    if image.dim() == 4:  # [1, 1, H, W]
        if image.shape[0] != 1 or image.shape[1] != 1:
            raise ValueError("4D tensor should have shape [1, 1, H, W]")
        image = image.squeeze()  # convert to 2D
    elif image.dim() == 3:  # [1, H, W]
        if image.shape[0] != 1:
            raise ValueError("3D tensor should have shape [1, H, W]")
        image = image.squeeze(0)  # convert to 2D
    elif image.dim() != 2:  # must be 2D
        raise ValueError("Invalid image dimensions, expected 2D, 3D or 4D tensor")

    # Convert to numpy array
    image = image.numpy()

    # Create and save image
    plt.figure(figsize=(8, 8))
    im = plt.imshow(image, cmap="viridis")
    plt.colorbar(im)
    plt.axis("off")
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0.1)
    plt.close()


def auto_load(model, ckpt_path, logger=None, exempt_list=[]):
    def log_info(msg):
        if logger is not None:
            logger.info(msg)
        else:
            print(msg)

    def log_warning(msg):
        if logger is not None:
            logger.warning(msg)
        else:
            print(f"WARNING: {msg}")

    # Load checkpoint
    ckpt = torch.load(ckpt_path, map_location="cpu")

    # Check if checkpoint contains a state_dict
    state_dict = ckpt.get("state_dict", ckpt)
    is_state_dict = "state_dict" in ckpt
    log_info(f"Loading from {'state_dict' if is_state_dict else 'direct checkpoint'}")

    # Filter state_dict according to exempt_list
    filtered_state_dict = {}
    for k, v in state_dict.items():
        should_exempt = False
        for exempt_pattern in exempt_list:
            if exempt_pattern in k:
                should_exempt = True
                break
        if not should_exempt:
            filtered_state_dict[k] = v

    # Get current model parameter keys
    model_keys = set(model.state_dict().keys())
    ckpt_keys = set(filtered_state_dict.keys())

    # Compute matching/missing/unexpected keys
    matched_keys = model_keys & ckpt_keys
    missing_keys = model_keys - ckpt_keys
    unexpected_keys = ckpt_keys - model_keys

    # Load with strict=False to allow partial matches
    model.load_state_dict(filtered_state_dict, strict=False)

    # Output detailed loading information
    log_info(f"Expected params: {len(model_keys)}")
    log_info(f"Successfully loaded params: {len(matched_keys)}")
    log_info(f"Missing params: {len(missing_keys)}")
    if len(missing_keys) > 0:
        log_warning(f"Missing keys: {missing_keys}")
    log_info(f"Unexpected params: {len(unexpected_keys)}")
    if len(unexpected_keys) > 0:
        log_warning(f"Unexpected keys: {unexpected_keys}")

    return model


def wrap_hw(batch_original_hw):
    h_list = batch_original_hw[0]
    w_list = batch_original_hw[1]
    return [(h.item(), w.item()) for h, w in zip(h_list, w_list)]
