import argparse
import logging
import os
import numpy as np
import torch
from torch.nn import functional as F
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import h5py


# Define global organ label mapping
ORGAN_LABELS = {
    1: "Spleen",
    2: "Right Kidney",
    3: "Left Kidney",
    4: "Gallbladder",
    5: "Liver",
    6: "Stomach",
    7: "Aorta",
    8: "Pancreas",
}


def get_color_for_label(label_id, cmap_name="viridis"):
    """
    Get consistent color for a specific label ID

    Parameters:
    -----------
    label_id: int
        The label ID (1-8 for organs)
    cmap_name: str
        Name of the colormap to use

    Returns:
    --------
    color: tuple
        RGBA color tuple
    """
    # Use a fixed number of classes for consistent colors
    num_classes = len(ORGAN_LABELS) + 1  # +1 to account for background (0)
    cmap = plt.cm.get_cmap(cmap_name, num_classes)

    # Map each label to a consistent color
    if label_id == 0:  # Background
        return (0, 0, 0, 0)  # Transparent

    # Normalize to [0,1] range for the colormap
    color_idx = label_id / num_classes
    return cmap(color_idx)


def get_model(args):
    """
    Load the model architecture based on the model name
    """
    if args.model_name.lower() == "msa2net":
        from networks.msa2net import MSA2Net

        net = MSA2Net(num_classes=args.num_classes)
    elif args.model_name.lower() == "merit":
        from networks.merit_lib.networks import MERIT_Parallel_Small

        net = MERIT_Parallel_Small(num_classes=args.num_classes)
    # Add more model options here
    else:
        raise ValueError(f"Model {args.model_name} not supported")

    return net


def preprocess_image(image_path, img_size=224):
    """
    Preprocess the input image
    """
    # Load the image depending on the format
    if image_path.endswith(".nii.gz") or image_path.endswith(".nii"):
        # Load medical image format
        image = sitk.ReadImage(image_path)
        image_np = sitk.GetArrayFromImage(image)
        # Assuming it's a 3D image, take the middle slice for 2D processing
        if len(image_np.shape) > 2 and image_np.shape[0] > 1:
            image_np = image_np[image_np.shape[0] // 2]
        # Normalize to [0, 255]
        if image_np.max() > 0:
            image_np = (
                (image_np - image_np.min()) / (image_np.max() - image_np.min())
            ) * 255.0
        image_np = image_np.astype(np.uint8)
    else:
        # Load standard image format
        image = np.array(Image.open(image_path).convert("L"))  # Convert to grayscale

    # Apply the same transforms as in training
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
            transforms.Resize((img_size, img_size)),
        ]
    )

    # Apply transform
    image_tensor = transform(image_np).unsqueeze(0)  # Add batch dimension
    return image_tensor, image_np


def visualize_3d_segmentation(
    image,
    label=None,
    slice_idx=None,
    view="axial",
    num_slices=3,
    alpha=0.5,
    save_path=None,
    cmap="viridis",
    show_legend=True,
):
    """
    Visualize 3D medical image segmentation data

    Parameters:
    -----------
    image: numpy array
        3D image volume with shape (D, H, W)
    label: numpy array, optional
        3D segmentation mask with shape (D, H, W)
    slice_idx: int or list of int, optional
        Specific slice indices to display. If None, slices are chosen automatically
    view: str, optional
        Viewing plane: 'axial', 'sagittal', or 'coronal'
    num_slices: int, optional
        Number of slices to display if slice_idx is None
    alpha: float, optional
        Opacity of the segmentation overlay (0-1)
    save_path: str, optional
        Path to save the visualization. If None, just display
    cmap: str, optional
        Colormap for the segmentation
    show_legend: bool, optional
        Whether to show a legend with organ labels

    Returns:
    --------
    fig: matplotlib figure
        The created figure
    """
    if view == "axial":
        # Axial view: depth axis is first dimension
        total_slices = image.shape[0]
        img_aspect = image.shape[2] / image.shape[1]
        if slice_idx is None:
            slice_idx = np.linspace(0, total_slices - 1, num_slices).astype(int)

        def get_slice(vol, idx):
            return vol[idx, :, :]

    elif view == "sagittal":
        # Sagittal view: width axis becomes depth
        total_slices = image.shape[2]
        img_aspect = image.shape[0] / image.shape[1]
        if slice_idx is None:
            slice_idx = np.linspace(0, total_slices - 1, num_slices).astype(int)

        def get_slice(vol, idx):
            return vol[:, :, idx].T

    elif view == "coronal":
        # Coronal view: height axis becomes depth
        total_slices = image.shape[1]
        img_aspect = image.shape[0] / image.shape[2]
        if slice_idx is None:
            slice_idx = np.linspace(0, total_slices - 1, num_slices).astype(int)

        def get_slice(vol, idx):
            return vol[:, idx, :].T

    else:
        raise ValueError(
            f"Invalid view: {view}. Choose from 'axial', 'sagittal', or 'coronal'"
        )

    # Convert to list if single slice
    if isinstance(slice_idx, int):
        slice_idx = [slice_idx]

    # Create figure
    n_cols = min(4, len(slice_idx))
    n_rows = int(np.ceil(len(slice_idx) / n_cols))
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows * img_aspect)
    )

    # Handle single plot case
    if len(slice_idx) == 1:
        axes = np.array([axes])

    # Flatten axes for easy iteration
    if n_rows * n_cols > 1:
        axes = axes.flatten()

    # Plot each slice
    for i, idx in enumerate(slice_idx):
        if i < len(axes):
            ax = axes[i]
            img_slice = get_slice(image, idx)

            # Normalize image to [0, 1] for display
            img_slice = (img_slice - img_slice.min()) / (
                img_slice.max() - img_slice.min() + 1e-8
            )

            # Show the image slice
            ax.imshow(img_slice, cmap="gray")

            # Overlay segmentation if provided
            if label is not None:
                seg_slice = get_slice(label, idx)
                # Create mask for overlay
                mask = seg_slice > 0
                if mask.any():
                    # Create a color overlay
                    color_mask = np.zeros((*seg_slice.shape, 4))

                    # Get unique labels (excluding background 0)
                    unique_labels = np.unique(seg_slice)
                    unique_labels = unique_labels[unique_labels > 0]

                    # Apply colors to each label
                    for lab_val in unique_labels:
                        color = get_color_for_label(lab_val, cmap_name=cmap)
                        mask_j = seg_slice == lab_val
                        color_mask[mask_j, :] = (*color[:3], alpha)

                    ax.imshow(color_mask, interpolation="nearest")

            ax.set_title(f"{view.capitalize()} Slice {idx}")
            ax.axis("off")

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    # Add legend if requested and there are segmentations
    if show_legend and label is not None:
        # Get unique labels across all visualized slices
        all_labels = set()
        for idx in slice_idx:
            seg_slice = get_slice(label, idx)
            unique_labs = np.unique(seg_slice)
            all_labels.update(unique_labs)

        # Remove background
        if 0 in all_labels:
            all_labels.remove(0)

        # Only show legend if there are labels
        if all_labels:
            # Create patches for legend
            patches = []
            for lab in sorted(all_labels):
                if lab in ORGAN_LABELS:
                    color = get_color_for_label(lab, cmap_name=cmap)[:3]
                    patch = plt.Rectangle(
                        (0, 0), 1, 1, color=color, label=f"{lab}: {ORGAN_LABELS[lab]}"
                    )
                    patches.append(patch)

            # Place the legend
            fig.legend(
                handles=patches,
                loc="lower center",
                bbox_to_anchor=(0.5, 0.02),
                ncol=min(3, len(patches)),
                title="Organ Labels",
                fontsize="small",
            )

            # Add some padding at the bottom for the legend
            plt.subplots_adjust(bottom=0.15)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Visualization saved to {save_path}")

    return fig


def load_h5_data(file_path):
    """
    Load 3D image and label data from h5 file

    Parameters:
    -----------
    file_path: str
        Path to the h5 file

    Returns:
    --------
    image: numpy array
        3D image volume
    label: numpy array
        3D segmentation mask
    """
    data = h5py.File(file_path, "r")
    image = data["image"][:]
    label = data["label"][:]
    return image, label


def process_3d_volume(model, image_volume, args, slice_step=1):
    """
    Process a 3D volume slice by slice and reconstruct the 3D segmentation

    Parameters:
    -----------
    model: torch model
        The segmentation model
    image_volume: numpy array
        3D image volume with shape (D, H, W)
    args: argparse namespace
        Model and processing arguments
    slice_step: int
        Step size for processing slices (1 = all slices, 2 = every other slice, etc.)

    Returns:
    --------
    prediction_volume: numpy array
        3D segmentation volume with shape (D, H, W)
    """
    model.eval()

    # Get original dimensions
    depth, height, width = image_volume.shape

    # Create an output volume for the predictions
    prediction_volume = np.zeros_like(image_volume, dtype=np.uint8)

    # Create normalization transform
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
            transforms.Resize((args.img_size, args.img_size)),
        ]
    )

    # Process each slice
    with torch.no_grad():
        for z in range(0, depth, slice_step):
            # Get slice and normalize
            img_slice = image_volume[z]

            # Normalize slice to [0, 255]
            if img_slice.max() > 0:
                img_slice = (
                    (img_slice - img_slice.min()) / (img_slice.max() - img_slice.min())
                ) * 255.0
                img_slice = img_slice.astype(np.uint8)

            # Process slice
            img_tensor = transform(img_slice).unsqueeze(0)
            img_tensor = img_tensor.cuda()

            # Run inference
            output = model(img_tensor)

            # Handle model output
            if isinstance(output, tuple) or isinstance(output, list):
                output = output[0]

            # Resize to original dimensions
            output = F.interpolate(
                output, size=(height, width), mode="bilinear", align_corners=False
            )

            # Get prediction
            prediction = torch.argmax(torch.softmax(output, dim=1), dim=1).squeeze(0)
            prediction_volume[z] = prediction.cpu().numpy()

    return prediction_volume


def inference_single_image(model, image_tensor, args):
    """
    Run inference on a single image
    """
    model.eval()
    with torch.no_grad():
        image_tensor = image_tensor.cuda()
        output = model(image_tensor)

        # Get the prediction
        if isinstance(output, tuple) or isinstance(output, list):
            # Some models may return multiple outputs, use the main prediction
            output = output[0]

        output = F.interpolate(
            output, size=image_tensor.shape[2:], mode="bilinear", align_corners=False
        )
        prediction = torch.argmax(torch.softmax(output, dim=1), dim=1).squeeze(0)

    return prediction.cpu().numpy()


def save_output(prediction, original_image, output_path):
    """
    Save the prediction as an image
    """
    # Create a figure with the original image and the prediction
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(original_image, cmap="gray")
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(prediction, cmap="viridis")
    plt.title("Segmentation")
    plt.axis("off")

    # Add legend for the prediction
    unique_labels = np.unique(prediction)
    if np.any(unique_labels > 0):
        patches = []
        for lab in sorted(unique_labels):
            if lab > 0 and lab in ORGAN_LABELS:
                color = get_color_for_label(lab, cmap_name="viridis")[:3]
                patch = plt.Rectangle(
                    (0, 0), 1, 1, color=color, label=f"{lab}: {ORGAN_LABELS[lab]}"
                )
                patches.append(patch)

        if patches:
            plt.legend(
                handles=patches,
                loc="lower center",
                bbox_to_anchor=(0.5, -0.15),
                ncol=min(3, len(patches)),
                title="Organ Labels",
            )
            plt.subplots_adjust(bottom=0.2)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    # Also save the raw prediction as a numpy array
    np.save(output_path.replace(".png", ".npy"), prediction)

    # Save a colorized overlay
    if len(original_image.shape) == 2:
        original_rgb = cv2.cvtColor(original_image, cv2.COLOR_GRAY2RGB)
    else:
        original_rgb = original_image

    # Create a colorized overlay with consistent colors per label
    overlay = np.zeros_like(original_rgb, dtype=np.float32)

    for lab in unique_labels:
        if lab > 0:  # Skip background
            mask = prediction == lab
            if mask.any():
                color = get_color_for_label(lab, cmap_name="viridis")
                for c in range(3):
                    overlay[mask, c] = color[c] * 255

    # Blend the original image with the colored segmentation
    overlay = cv2.addWeighted(original_rgb, 0.7, overlay.astype(np.uint8), 0.3, 0)
    cv2.imwrite(output_path.replace(".png", "_overlay.png"), overlay)

    return


def create_unified_visualization(
    image, prediction=None, ground_truth=None, output_path=None, slice_indices=None
):
    """
    Create a unified visualization of 3D medical segmentation data showing
    axial views and comparison with ground truth.

    Parameters:
    -----------
    image: numpy array
        3D image volume with shape (D, H, W)
    prediction: numpy array, optional
        3D segmentation prediction with shape (D, H, W)
    ground_truth: numpy array, optional
        3D segmentation ground truth with shape (D, H, W)
    output_path: str, optional
        Path to save the unified visualization
    slice_indices: list, optional
        List of axial slice indices to visualize
    """
    # Set default slice indices if not provided
    if slice_indices is None:
        middle_slice = image.shape[0] // 2
        slice_indices = [middle_slice - 20, middle_slice, middle_slice + 20]
        # Make sure indices are within bounds
        slice_indices = [max(0, min(s, image.shape[0] - 1)) for s in slice_indices]

    # Determine the layout based on what data is available
    has_gt = ground_truth is not None
    has_pred = prediction is not None

    # Determine the number of rows and columns
    n_rows = len(slice_indices)
    n_cols = 1 + has_gt + (has_gt and has_pred)

    # Create figure
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))

    # Handle single row case
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    # For each slice
    for i, slice_idx in enumerate(slice_indices):
        col = 0

        # Original image with prediction overlay
        if has_pred:
            ax = axes[i, col]
            img_slice = image[slice_idx]
            pred_slice = prediction[slice_idx] if prediction is not None else None

            # Normalize image for display
            img_slice = (img_slice - img_slice.min()) / (
                img_slice.max() - img_slice.min() + 1e-8
            )
            ax.imshow(img_slice, cmap="gray")

            # Overlay prediction
            if pred_slice is not None:
                mask = pred_slice > 0
                if mask.any():
                    color_mask = np.zeros((*pred_slice.shape, 4))
                    unique_labels = np.unique(pred_slice)
                    unique_labels = unique_labels[unique_labels > 0]
                    cmap_func = plt.cm.get_cmap("viridis", max(len(unique_labels), 1))

                    for j, lab_val in enumerate(unique_labels):
                        color = get_color_for_label(lab_val, cmap_name="viridis")
                        mask_j = pred_slice == lab_val
                        color_mask[mask_j, :] = (*color[:3], 0.4)

                    ax.imshow(color_mask, interpolation="nearest")

            ax.set_title(f"Prediction (Slice {slice_idx})")
            ax.axis("off")
            col += 1

        # Original image with ground truth overlay
        if has_gt:
            ax = axes[i, col]
            img_slice = image[slice_idx]
            gt_slice = ground_truth[slice_idx]

            # Normalize image for display
            img_slice = (img_slice - img_slice.min()) / (
                img_slice.max() - img_slice.min() + 1e-8
            )
            ax.imshow(img_slice, cmap="gray")

            # Overlay ground truth
            mask = gt_slice > 0
            if mask.any():
                color_mask = np.zeros((*gt_slice.shape, 4))
                unique_labels = np.unique(gt_slice)
                unique_labels = unique_labels[unique_labels > 0]
                cmap_func = plt.cm.get_cmap("viridis", max(len(unique_labels), 1))

                for j, lab_val in enumerate(unique_labels):
                    color = get_color_for_label(lab_val, cmap_name="viridis")
                    mask_j = gt_slice == lab_val
                    color_mask[mask_j, :] = (*color[:3], 0.4)

                ax.imshow(color_mask, interpolation="nearest")

            ax.set_title(f"Ground Truth (Slice {slice_idx})")
            ax.axis("off")
            col += 1

        # Difference map if both prediction and ground truth are available
        if has_pred and has_gt:
            ax = axes[i, col]
            img_slice = image[slice_idx]
            pred_slice = prediction[slice_idx]
            gt_slice = ground_truth[slice_idx]
            diff = (pred_slice != gt_slice).astype(np.float32)

            # Normalize image for display
            img_slice = (img_slice - img_slice.min()) / (
                img_slice.max() - img_slice.min() + 1e-8
            )
            ax.imshow(img_slice, cmap="gray", alpha=0.7)
            ax.imshow(diff, cmap="hot", alpha=0.5)
            ax.set_title(f"Difference (Slice {slice_idx})")
            ax.axis("off")

    plt.tight_layout()

    # Add legend
    all_labels = set()

    # Collect labels from prediction
    if prediction is not None:
        for idx in slice_indices:
            unique_labs = np.unique(prediction[idx])
            all_labels.update(unique_labs)

    # Collect labels from ground truth
    if ground_truth is not None:
        for idx in slice_indices:
            unique_labs = np.unique(ground_truth[idx])
            all_labels.update(unique_labs)

    # Remove background
    if 0 in all_labels:
        all_labels.remove(0)

    # Add legend if we have labels
    if all_labels:
        patches = []
        for lab in sorted(all_labels):
            if lab in ORGAN_LABELS:
                color = get_color_for_label(lab, cmap_name="viridis")[:3]
                patch = plt.Rectangle(
                    (0, 0), 1, 1, color=color, label=f"{lab}: {ORGAN_LABELS[lab]}"
                )
                patches.append(patch)

        # Place the legend
        fig.legend(
            handles=patches,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.02),
            ncol=min(3, len(patches)),
            title="Organ Labels",
        )

        # Add padding at the bottom for the legend
        plt.subplots_adjust(bottom=0.15)

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Unified visualization saved to {output_path}")

    return fig


def create_slice_montage(
    volume,
    prediction=None,
    ground_truth=None,
    output_path=None,
    n_slices=9,
    start_slice=None,
    end_slice=None,
    slice_interval=None,
):
    """
    Create a montage visualization showing multiple slices from a 3D volume

    Parameters:
    -----------
    volume: numpy array
        3D image volume with shape (D, H, W)
    prediction: numpy array, optional
        3D segmentation prediction with shape (D, H, W)
    ground_truth: numpy array, optional
        3D segmentation ground truth with shape (D, H, W)
    output_path: str, optional
        Path to save the visualization
    n_slices: int, optional
        Number of slices to include in the montage
    start_slice: int, optional
        Starting slice index (if None, will be calculated)
    end_slice: int, optional
        Ending slice index (if None, will be calculated)
    slice_interval: int, optional
        Interval between slices (if None, will be calculated)

    Returns:
    --------
    fig: matplotlib figure
        The created figure
    """
    depth = volume.shape[0]

    # Calculate slice indices
    if start_slice is None and end_slice is None:
        # Find the range where the volume is non-zero
        non_zero_slices = []
        for z in range(depth):
            if np.any(volume[z] > 0):
                non_zero_slices.append(z)

        if non_zero_slices:
            start_slice = max(0, min(non_zero_slices))
            end_slice = min(depth - 1, max(non_zero_slices))
        else:
            # Default to middle section if no non-zero slices found
            middle = depth // 2
            start_slice = max(0, middle - depth // 4)
            end_slice = min(depth - 1, middle + depth // 4)

    # Default values
    if start_slice is None:
        start_slice = 0
    if end_slice is None:
        end_slice = depth - 1

    # Calculate slice interval
    if slice_interval is None:
        # Distribute slices evenly across the range
        slice_interval = max(1, (end_slice - start_slice) // (n_slices - 1))

    # Generate slice indices
    slice_indices = list(range(start_slice, end_slice + 1, slice_interval))[:n_slices]

    # Adjust rows and columns for the montage
    n_cols = min(4, n_slices)
    n_rows = (n_slices + n_cols - 1) // n_cols  # Ceiling division

    # Create figure
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows))

    # Handle single row/col cases
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1 or n_cols == 1:
        axes = axes.reshape(n_rows, n_cols)

    # For each slice
    for i, slice_idx in enumerate(slice_indices):
        if i < n_rows * n_cols:
            row, col = i // n_cols, i % n_cols
            ax = axes[row, col]

            # Get the slice
            img_slice = volume[slice_idx]

            # Normalize for display
            if img_slice.max() > img_slice.min():
                img_slice = (img_slice - img_slice.min()) / (
                    img_slice.max() - img_slice.min()
                )

            # Display the slice
            ax.imshow(img_slice, cmap="gray")

            # Add overlays if provided
            if prediction is not None:
                pred_slice = prediction[slice_idx]
                # Create a colored overlay
                pred_overlay = np.zeros((*pred_slice.shape, 4))

                for label in range(1, int(pred_slice.max()) + 1):
                    mask = pred_slice == label
                    if mask.any():
                        color = get_color_for_label(label)  # Use consistent coloring
                        pred_overlay[mask] = (*color[:3], 0.5)

                ax.imshow(pred_overlay)

            # Add ground truth contour if provided
            if ground_truth is not None:
                gt_slice = ground_truth[slice_idx]
                # Draw contours of ground truth segments
                for label in range(1, int(gt_slice.max()) + 1):
                    mask = gt_slice == label
                    if mask.any():
                        from skimage import measure

                        contours = measure.find_contours(mask.astype(float), 0.5)
                        for contour in contours:
                            ax.plot(contour[:, 1], contour[:, 0], "r-", linewidth=1)

            # Set title and turn off axis
            ax.set_title(f"Slice {slice_idx}")
            ax.axis("off")

    # Hide unused subplots
    for i in range(len(slice_indices), n_rows * n_cols):
        row, col = i // n_cols, i % n_cols
        axes[row, col].axis("off")

    # Collect all labels from prediction and ground truth
    all_labels = set()
    if prediction is not None:
        for idx in slice_indices:
            unique_labs = np.unique(prediction[idx])
            all_labels.update(unique_labs)

    if ground_truth is not None:
        for idx in slice_indices:
            unique_labs = np.unique(ground_truth[idx])
            all_labels.update(unique_labs)

    # Remove background
    if 0 in all_labels:
        all_labels.remove(0)

    # Add legend if we have labels
    if all_labels:
        patches = []
        for lab in sorted(all_labels):
            if lab in ORGAN_LABELS:
                color = get_color_for_label(lab)[:3]
                patch = plt.Rectangle(
                    (0, 0), 1, 1, color=color, label=f"{lab}: {ORGAN_LABELS[lab]}"
                )
                patches.append(patch)

        # Place the legend below the montage
        fig.legend(
            handles=patches,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.02),
            ncol=min(3, len(patches)),
            title="Organ Labels",
        )

        # Add padding for the legend
        plt.subplots_adjust(bottom=0.15)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Slice montage saved to {output_path}")

    return fig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="msa2net", help="Model name")
    parser.add_argument("--input_image", type=str, help="Path to the input image (2D)")
    parser.add_argument(
        "--input_volume", type=str, help="Path to the input volume (3D h5 file)"
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to the model checkpoint"
    )
    parser.add_argument(
        "--output_dir", type=str, default="./output", help="Directory to save results"
    )
    parser.add_argument(
        "--img_size", type=int, default=224, help="Image size for the model input"
    )
    parser.add_argument("--num_classes", type=int, default=9, help="Number of classes")
    parser.add_argument("--gpu", type=str, default="0", help="GPU ID to use")
    parser.add_argument(
        "--visualize", action="store_true", help="Visualize the results"
    )
    parser.add_argument(
        "--view",
        type=str,
        default="axial",
        help="Visualization view: axial, sagittal, or coronal",
    )
    parser.add_argument(
        "--slice_step",
        type=int,
        default=1,
        help="Step size for processing 3D volume slices",
    )
    args = parser.parse_args()

    # Set GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s.%(msecs)03d] %(message)s",
        datefmt="%H:%M:%S",
    )

    # Load model
    logging.info(f"Loading model: {args.model_name}")
    model = get_model(args)
    model.cuda()

    # Load checkpoint
    logging.info(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint)
    model.load_state_dict(checkpoint)

    if args.input_volume:
        # Process 3D volume
        logging.info(f"Processing 3D volume: {args.input_volume}")

        # Load the volume
        image_volume, label_volume = load_h5_data(args.input_volume)
        logging.info(f"Loaded volume with shape: {image_volume.shape}")

        # Process the volume
        prediction_volume = process_3d_volume(
            model, image_volume, args, slice_step=args.slice_step
        )

        # Save the prediction volume
        output_path = os.path.join(
            args.output_dir,
            os.path.basename(args.input_volume).split(".")[0] + "_seg.npy",
        )
        np.save(output_path, prediction_volume)
        logging.info(f"Saved 3D segmentation to: {output_path}")

        # Visualize if requested
        if args.visualize:
            # Generate visualizations for multiple slices
            middle_slice = image_volume.shape[0] // 2
            slice_indices = [middle_slice - 20, middle_slice, middle_slice + 20]

            viz_path = os.path.join(
                args.output_dir,
                os.path.basename(args.input_volume).split(".")[0] + "_viz.png",
            )
            visualize_3d_segmentation(
                image_volume,
                prediction_volume,
                slice_idx=slice_indices,
                view=args.view,
                save_path=viz_path,
            )

            # If we have ground truth, visualize the comparison
            if label_volume is not None:
                gt_viz_path = os.path.join(
                    args.output_dir,
                    os.path.basename(args.input_volume).split(".")[0] + "_gt_viz.png",
                )
                visualize_3d_segmentation(
                    image_volume,
                    label_volume,
                    slice_idx=slice_indices,
                    view=args.view,
                    save_path=gt_viz_path,
                )

                # Create a side-by-side visualization
                fig, axes = plt.subplots(1, 2, figsize=(12, 6))
                axes[0].imshow(image_volume[middle_slice], cmap="gray")
                axes[0].set_title("Original Image (Middle Slice)")
                axes[0].axis("off")

                # Create a difference map
                diff = (
                    prediction_volume[middle_slice] != label_volume[middle_slice]
                ).astype(np.float32)
                axes[1].imshow(diff, cmap="hot")
                axes[1].set_title("Difference Map (Prediction vs GT)")
                axes[1].axis("off")

                plt.tight_layout()
                diff_path = os.path.join(
                    args.output_dir,
                    os.path.basename(args.input_volume).split(".")[0] + "_diff.png",
                )
                plt.savefig(diff_path)
                plt.close()

            # Add unified visualization with only axial slices
            unified_viz_path = os.path.join(
                args.output_dir,
                os.path.basename(args.input_volume).split(".")[0] + "_unified_viz.png",
            )

            # Use only axial slices
            middle_slice = image_volume.shape[0] // 2
            slice_indices = [middle_slice - 20, middle_slice, middle_slice + 20]
            # Make sure indices are within bounds
            slice_indices = [
                max(0, min(s, image_volume.shape[0] - 1)) for s in slice_indices
            ]

            create_unified_visualization(
                image_volume,
                prediction_volume,
                label_volume,
                unified_viz_path,
                slice_indices,
            )

            # Create a montage of multiple slices
            montage_path = os.path.join(
                args.output_dir,
                os.path.basename(args.input_volume).split(".")[0] + "_montage.png",
            )

            create_slice_montage(
                image_volume, prediction_volume, label_volume, montage_path, n_slices=16
            )

    elif args.input_image:
        # Process 2D image (existing functionality)
        logging.info(f"Processing image: {args.input_image}")
        image_tensor, original_image = preprocess_image(args.input_image, args.img_size)

        # Run inference
        logging.info("Running inference")
        prediction = inference_single_image(model, image_tensor, args)

        # Save output
        output_path = os.path.join(
            args.output_dir,
            os.path.basename(args.input_image).split(".")[0] + "_seg.png",
        )
        logging.info(f"Saving output to: {output_path}")
        save_output(prediction, original_image, output_path)

    else:
        logging.error("Please provide either --input_image or --input_volume")
        return

    logging.info("Inference complete!")


if __name__ == "__main__":
    main()
