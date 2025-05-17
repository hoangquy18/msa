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
        from networks.msa2net import Msa2Net

        net = Msa2Net(num_classes=args.num_classes)
    elif args.model_name.lower() == "merit":
        from networks.merit_lib.networks import MERIT_Parallel_Small

        net = MERIT_Parallel_Small(num_classes=args.num_classes)
    # Add more model options here
    else:
        raise ValueError(f"Model {args.model_name} not supported")

    return net


def create_slice_montage(
    volume,
    prediction=None,
    ground_truth=None,
    output_path=None,
    n_slices=9,
    start_slice=None,
    end_slice=None,
    slice_interval=None,
    segmentation_type="CT",
    return_fig=False,  # Add this parameter
):
    """
    Create a montage visualization showing multiple slices from a 3D volume

    Parameters:
    -----------
    volume: numpy array
        3D image volume with shape (D, H, W) for grayscale or (D, H, W, 3) for RGB
    prediction: numpy array, optional
        3D segmentation prediction with shape (D, H, W) or (1, H, W) for single slice
    ground_truth: numpy array, optional
        3D segmentation ground truth with shape (D, H, W) or (1, H, W) for single slice
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
    segmentation_type: str, optional
        Type of segmentation ("CT" or "ISIC"), determines which labels to display

    Returns:
    --------
    img_array: numpy array
        The created image as a numpy array with shape (H, W, 4) in RGBA format
    """
    # Check if volume is RGB (4D) or grayscale (3D)
    is_rgb = len(volume.shape) == 4 and volume.shape[3] == 3

    depth = volume.shape[0]

    # Check if prediction or ground_truth is single slice with shape (1, H, W)
    single_slice_pred = prediction is not None and prediction.shape[0] == 1
    single_slice_gt = ground_truth is not None and ground_truth.shape[0] == 1

    # Calculate slice indices
    if start_slice is None and end_slice is None:
        # Find the range where the volume is non-zero
        non_zero_slices = []
        for z in range(depth):
            if is_rgb:
                if np.any(volume[z] > 0):
                    non_zero_slices.append(z)
            else:
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
    n_cols = 1
    n_rows = 1

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

            # Handle RGB or grayscale normalization and display
            if is_rgb:
                # For RGB images, handle normalization more carefully
                if img_slice.max() > 1.0:  # Assume [0-255] range if > 1
                    # Normalize each channel separately for better color representation
                    normalized_slice = np.zeros_like(img_slice, dtype=np.float32)
                    for c in range(3):
                        channel = img_slice[:, :, c]
                        if channel.max() > channel.min():
                            normalized_slice[:, :, c] = (channel - channel.min()) / (
                                channel.max() - channel.min()
                            )
                        else:
                            normalized_slice[:, :, c] = channel / 255.0
                    img_slice = normalized_slice
                # Display RGB image without colormap
                ax.imshow(img_slice)
            else:
                # Normalize grayscale for display
                if img_slice.max() > img_slice.min():
                    img_slice = (img_slice - img_slice.min()) / (
                        img_slice.max() - img_slice.min()
                    )
                ax.imshow(img_slice, cmap="gray")

            # Add prediction overlay if provided
            if prediction is not None:
                # Get the correct prediction slice
                if single_slice_pred:
                    pred_slice = prediction[0]  # For shape (1, H, W)
                else:
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
                # Get the correct ground truth slice
                if single_slice_gt:
                    gt_slice = ground_truth[0]  # For shape (1, H, W)
                else:
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
            ax.axis("off")

    # Select the appropriate labels based on segmentation type
    if segmentation_type.upper() == "ISIC":
        # For ISIC segmentation, we only have abnormal class
        LABELS_DICT = {1: "Abnormal"}
    else:
        # For CT segmentation, use the standard organ labels
        LABELS_DICT = ORGAN_LABELS

    # Collect all labels from prediction and ground truth
    all_labels = set()
    if prediction is not None:
        if single_slice_pred:
            unique_labs = np.unique(prediction[0])
            all_labels.update(unique_labs)
        else:
            for idx in slice_indices:
                unique_labs = np.unique(prediction[idx])
                all_labels.update(unique_labs)

    if ground_truth is not None:
        if single_slice_gt:
            unique_labs = np.unique(ground_truth[0])
            all_labels.update(unique_labs)
        else:
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
            if lab in LABELS_DICT:
                color = get_color_for_label(lab)[:3]
                patch = plt.Rectangle(
                    (0, 0), 1, 1, color=color, label=f"{lab}: {LABELS_DICT[lab]}"
                )
                patches.append(patch)

        # Place the legend on the right side outside the figure
        fig.legend(
            handles=patches,
            loc="center left",
            bbox_to_anchor=(1.05, 0.5),  # Position to the right of the figure
            ncol=1,  # Display in a single column
            title=(
                "Organ Labels" if segmentation_type.upper() == "CT" else "Class Labels"
            ),
        )

        # Add padding on the right for the legend
        plt.subplots_adjust(right=0.8)  # Make room for the legend on the right

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Slice montage saved to {output_path}")

    # Convert figure to image array
    canvas = fig.canvas
    canvas.draw()

    # Get the RGBA buffer from the figure
    img_array = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
    img_array = img_array.reshape(canvas.get_width_height()[::-1] + (3,))

    # Return both the image array and figure if requested
    if return_fig:
        return img_array, fig

    # Close the figure to free memory if not returning it
    plt.close(fig)

    return img_array
