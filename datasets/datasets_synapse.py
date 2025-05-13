import os
import random
import h5py
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset
import imgaug as ia
import imgaug.augmenters as iaa


def mask_to_onehot(mask, num_classes):
    """
    Converts a segmentation mask (H, W, C) to (H, W, K) where the last dim is a one
    hot encoding vector, C is usually 1 or 3, and K is the number of class.
    """
    semantic_map = []
    mask = np.expand_dims(mask, -1)
    for colour in range(num_classes):
        equality = np.equal(mask, colour)
        class_map = np.all(equality, axis=-1)
        semantic_map.append(class_map)
    semantic_map = np.stack(semantic_map, axis=-1).astype(np.int32)
    return semantic_map


def augment_seg(img_aug, img, seg, num_classes=9):
    seg = mask_to_onehot(seg, num_classes)
    aug_det = img_aug.to_deterministic()
    image_aug = aug_det.augment_image(img)

    segmap = ia.SegmentationMapOnImage(seg, nb_classes=np.max(seg) + 1, shape=img.shape)
    segmap_aug = aug_det.augment_segmentation_maps(segmap)
    segmap_aug = segmap_aug.get_arr_int()
    segmap_aug = np.argmax(segmap_aug, axis=-1).astype(np.float32)
    return image_aug, segmap_aug


def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample["image"], sample["label"]

        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        x, y = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(
                image, (self.output_size[0] / x, self.output_size[1] / y), order=3
            )  # why not 3?
            label = zoom(
                label, (self.output_size[0] / x, self.output_size[1] / y), order=0
            )
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.float32))
        sample = {"image": image, "label": label.long()}
        return sample


class Synapse_dataset(Dataset):
    def __init__(
        self,
        base_dir,
        list_dir,
        split,
        img_size,
        norm_x_transform=None,
        norm_y_transform=None,
    ):
        self.norm_x_transform = norm_x_transform
        self.norm_y_transform = norm_y_transform
        self.split = split
        self.sample_list = open(os.path.join(list_dir, self.split + ".txt")).readlines()
        self.data_dir = base_dir
        self.img_size = img_size

        self.img_aug = iaa.SomeOf(
            (0, 4),
            [
                iaa.Flipud(0.5, name="Flipud"),
                iaa.Fliplr(0.5, name="Fliplr"),
                iaa.AdditiveGaussianNoise(scale=0.005 * 255),
                iaa.GaussianBlur(sigma=(1.0)),
                iaa.LinearContrast((0.5, 1.5), per_channel=0.5),
                iaa.Affine(scale={"x": (0.5, 2), "y": (0.5, 2)}),
                iaa.Affine(rotate=(-40, 40)),
                iaa.Affine(shear=(-16, 16)),
                iaa.PiecewiseAffine(scale=(0.008, 0.03)),
                iaa.Affine(translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}),
            ],
            random_order=True,
        )

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        if self.split == "train":
            slice_name = self.sample_list[idx].strip("\n")
            data_path = os.path.join(self.data_dir, slice_name + ".npz")
            data = np.load(data_path)
            image, label = data["image"], data["label"]
            image, label = augment_seg(self.img_aug, image, label, 9)
            x, y = image.shape
            if x != self.img_size or y != self.img_size:
                image = zoom(
                    image, (self.img_size / x, self.img_size / y), order=3
                )  # why not 3?
                label = zoom(label, (self.img_size / x, self.img_size / y), order=0)

        else:
            vol_name = self.sample_list[idx].strip("\n")
            filepath = self.data_dir + "/{}.npy.h5".format(vol_name)
            data = h5py.File(filepath)
            image, label = data["image"][:], data["label"][:]

        sample = {"image": image, "label": label}
        if self.norm_x_transform is not None:
            sample["image"] = self.norm_x_transform(sample["image"].copy())
        if self.norm_y_transform is not None:
            sample["label"] = self.norm_y_transform(sample["label"].copy())
        sample["case_name"] = self.sample_list[idx].strip("\n")
        return sample


class ISIC_dataset(Dataset):
    def __init__(
        self,
        base_dir,
        list_dir,
        split,
        img_size,
        norm_x_transform=None,
        norm_y_transform=None,
    ):
        self.norm_x_transform = norm_x_transform
        self.norm_y_transform = norm_y_transform
        self.split = split
        self.data_dir = base_dir
        self.seg_dir = (
            os.path.join(base_dir, self.split + "/label")
            if os.path.exists(os.path.join(base_dir, self.split + "/label"))
            else list_dir
        )
        self.img_size = img_size
        self.sample_list = os.listdir(self.data_dir)
        # Image augmentation for training
        self.img_aug = (
            iaa.SomeOf(
                (0, 4),
                [
                    iaa.Flipud(0.5, name="Flipud"),
                    iaa.Fliplr(0.5, name="Fliplr"),
                    iaa.AdditiveGaussianNoise(scale=0.005 * 255),
                    iaa.GaussianBlur(sigma=(1.0)),
                    iaa.LinearContrast((0.5, 1.5), per_channel=0.5),
                    iaa.Affine(scale={"x": (0.5, 2), "y": (0.5, 2)}),
                    iaa.Affine(rotate=(-40, 40)),
                    iaa.Affine(shear=(-16, 16)),
                    iaa.PiecewiseAffine(scale=(0.008, 0.03)),
                    iaa.Affine(translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}),
                ],
                random_order=True,
            )
            if split == "train"
            else None
        )

        self.sample_list = os.listdir(self.data_dir + "/" + self.split + "/" + "image")
        self.image_dir = os.path.join(self.data_dir, self.split + "/image")
        self.seg_dir = os.path.join(self.data_dir, self.split + "/label")

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        slice_name = self.sample_list[idx]

        # Load RGB image
        img_extensions = [".jpg", ".jpeg", ".png"]
        img_path = None
        for ext in img_extensions:
            if os.path.exists(os.path.join(self.image_dir, slice_name)):
                img_path = os.path.join(self.image_dir, slice_name)
                break

        if img_path is None:
            raise FileNotFoundError(f"No image file found for {slice_name}")

        # Load using PIL and convert to numpy
        from PIL import Image

        image = np.array(Image.open(img_path))  # This will be RGB

        # Load segmentation mask (binary PNG)
        seg_path = os.path.join(
            self.seg_dir, slice_name.split(".")[0] + "_Segmentation.png"
        )
        if not os.path.exists(seg_path):
            raise FileNotFoundError(f"No segmentation file found at {seg_path}")

        label = np.array(Image.open(seg_path))
        if len(label.shape) > 2:  # If the mask has multiple channels, convert to binary
            label = label[:, :, 0]

        # Convert to binary mask if not already
        label = (label > 0).astype(np.float32)

        # Apply augmentations for training
        if self.split == "train" and self.img_aug is not None:
            image, label = augment_seg(self.img_aug, image, label, 2)

        # Resize if needed
        if image.shape[0] != self.img_size or image.shape[1] != self.img_size:
            image = zoom(
                image,
                (self.img_size / image.shape[0], self.img_size / image.shape[1], 1),
                order=3,
            )
            label = zoom(
                label,
                (self.img_size / label.shape[0], self.img_size / label.shape[1]),
                order=0,
            )

        # Apply transformations if provided
        sample = {"image": image, "label": label}
        if self.norm_x_transform is not None:
            sample["image"] = self.norm_x_transform(sample["image"].copy())
        if self.norm_y_transform is not None:
            sample["label"] = self.norm_y_transform(sample["label"].copy())

        sample["case_name"] = slice_name
        return sample
