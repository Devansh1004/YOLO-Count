from typing import Tuple
import os
import json
import cv2
import inflect
import numpy as np
import torch
from torch.utils.data import Dataset
from pycocotools.coco import COCO
from lvis import LVIS
from datasets import load_dataset


class FSCData(Dataset):
    """FSC147 dataset loader"""

    def __init__(
        self,
        root: str,
        split: str,
        flip: bool = True,
        img_res: int = 640,
        cell_res: int = 80,
    ):
        super().__init__()
        self.root = root
        self.split = split
        self.flip = flip
        self.img_res = img_res
        self.cell_res = cell_res

        # Read dataset splits
        with open(os.path.join(root, "FSC_147/Train_Test_Val_FSC_147.json"), "r") as f:
            self.splits = json.load(f)
        self.image_list = self.splits[split]

        self.inflect_engine = inflect.engine()

        # Read class information and convert to singular form
        self.classes = {}
        with open(os.path.join(root, "FSC_147/ImageClasses_FSC_147.txt"), "r") as f:
            for line in f:
                img_name, cls_name = line.strip().split("\t")
                # Convert plural to singular
                words = cls_name.split()
                words[-1] = self.inflect_engine.singular_noun(words[-1]) or words[-1]
                self.classes[img_name] = " ".join(words)

        self.eps = 1e-6

        # Read annotation files
        self.point_list = {}
        anno_path = os.path.join(root, "FSC_147/annotation_FSC_147_384.json")
        with open(anno_path, "r") as f:
            annotations = json.load(f)
            for img_name, anno in annotations.items():
                if "points" in anno:
                    self.point_list[img_name] = anno["points"]

    def padding_and_resize(self, img, res, keep_sum=False):
        """Pad and resize while keeping aspect ratio

        Args:
            img: numpy array (H, W, C) or (H, W)
        Returns:
            resized_img: numpy array (img_res, img_res, C) or (img_res, img_res)
        """
        h, w = img.shape[:2]
        ratio = max(h, w) / res
        new_h, new_w = int(h / ratio), int(w / ratio)

        # resize
        if len(img.shape) == 3:
            resized = cv2.resize(img, (new_w, new_h))
        else:
            resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # padding
        pad_h1, pad_h2 = (res - new_h) // 2, (res - new_h + 1) // 2
        pad_w1, pad_w2 = (res - new_w) // 2, (res - new_w + 1) // 2

        if len(img.shape) == 3:
            padded = np.pad(
                resized, ((pad_h1, pad_h2), (pad_w1, pad_w2), (0, 0)), mode="constant"
            )
        else:
            padded = np.pad(
                resized, ((pad_h1, pad_h2), (pad_w1, pad_w2)), mode="constant"
            )

        if keep_sum:
            padded_sum = np.sum(padded)
            if padded_sum > 0:
                padded = padded * np.sum(img) / padded_sum

        return padded

    def __getitem__(self, idx):
        img_name = self.image_list[idx]
        img_id = img_name.split(".")[0]

        # Load image
        img_path = os.path.join(self.root, "images_384_VarV2", img_name)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        original_hw = img.shape[:2]
        img = self.padding_and_resize(img, self.img_res)

        # Load density map
        density_path = os.path.join(
            self.root, "gt_density_map_adaptive_384_VarV2", f"{img_id}.npy"
        )
        density = np.load(density_path)
        density = self.padding_and_resize(density, self.cell_res, keep_sum=True)

        # Convert to tensor
        img = torch.from_numpy(img).float().permute(2, 0, 1) / 255.0
        density = torch.from_numpy(density).float().unsqueeze(0)

        # Random horizontal flip
        if self.flip and np.random.random() > 0.5:
            img = torch.flip(img, [2])
            density = torch.flip(density, [2])

        category = (density > self.eps).clone().detach().float()

        return {
            "image": img,
            "density_label": density,
            "category_label": category,
            "text_label": self.classes.get(img_name, "unknown"),
            "image_id": img_id,
            "count": density.sum().item(),
            "original_hw": original_hw,
            "img_path": img_path,
        }

    def __len__(self):
        return len(self.image_list)


class WeakFSCData(FSCData):
    """Weakly supervised FSC147 dataset loader"""

    def __init__(
        self,
        root: str,
        split: str,
        flip: bool = True,
        img_res: int = 640,
        cell_res: int = 80,
    ):
        super().__init__(root, split, flip, img_res, cell_res)

        # Read negative sample annotations
        self.neg_points = {}
        neg_anno_dir = os.path.join(root, "neg_annot")
        # Check if directory exists
        if os.path.exists(neg_anno_dir):
            for filename in os.listdir(neg_anno_dir):
                if filename.endswith(".json"):
                    img_id = filename.split(".")[0]
                    try:
                        with open(os.path.join(neg_anno_dir, filename), "r") as f:
                            self.neg_points[f"{img_id}.jpg"] = json.load(f)
                    except (json.JSONDecodeError, FileNotFoundError):
                        # Use an empty list if file read fails
                        self.neg_points[f"{img_id}.jpg"] = []

    def _point_to_cell(self, point_x, point_y, orig_w, orig_h) -> Tuple[int, int]:
        """Convert point coordinates to cell coordinates

        Args:
            point_x: x coordinate in the original image
            point_y: y coordinate in the original image
            orig_w: original image width
            orig_h: original image height
        Returns:
            cell_x, cell_y: coordinates in the cell_res sized grid
        """
        # Compute resized dimensions
        ratio = max(orig_h, orig_w) / self.cell_res
        new_h, new_w = int(orig_h / ratio), int(orig_w / ratio)

        # Compute padding sizes
        pad_h1 = (self.cell_res - new_h) // 2
        pad_w1 = (self.cell_res - new_w) // 2

        # Scale coordinates according to resize
        scaled_x = point_x / ratio
        scaled_y = point_y / ratio

        # Add padding offset
        cell_x = int(scaled_x + pad_w1)
        cell_y = int(scaled_y + pad_h1)

        return cell_x, cell_y

    def __getitem__(self, idx):
        img_name = self.image_list[idx]
        img_id = img_name.split(".")[0]

        # Load image
        img_path = os.path.join(self.root, "images_384_VarV2", img_name)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        original_hw = img.shape[:2]
        orig_h, orig_w = img.shape[:2]
        img = self.padding_and_resize(img, self.img_res)

        # Initialize category_label and valid_label_mask
        category_label = torch.zeros((1, self.cell_res, self.cell_res))
        valid_label_mask = torch.zeros((1, self.cell_res, self.cell_res))

        # Process positive sample points
        pos_points = self.point_list.get(img_name, [])
        for px, py in pos_points:
            cell_x, cell_y = self._point_to_cell(px, py, orig_w, orig_h)
            if 0 <= cell_x < self.cell_res and 0 <= cell_y < self.cell_res:
                category_label[0, cell_y, cell_x] = 1
                valid_label_mask[0, cell_y, cell_x] = 1

        # Process negative sample points
        neg_points = self.neg_points.get(img_name, [])
        for px, py in neg_points:
            cell_x, cell_y = self._point_to_cell(px, py, orig_w, orig_h)
            if 0 <= cell_x < self.cell_res and 0 <= cell_y < self.cell_res:
                category_label[0, cell_y, cell_x] = 0
                valid_label_mask[0, cell_y, cell_x] = 1

        # Convert image to tensor
        img = torch.from_numpy(img).float().permute(2, 0, 1) / 255.0

        # Random horizontal flip
        if self.flip and np.random.random() > 0.5:
            img = torch.flip(img, [2])
            category_label = torch.flip(category_label, [2])
            valid_label_mask = torch.flip(valid_label_mask, [2])

        return {
            "image": img,
            "category_label": category_label,
            "valid_label_mask": valid_label_mask,
            "text_label": self.classes.get(img_name, "unknown"),
            "image_id": img_id,
            "count": float(len(pos_points)),
            "original_hw": original_hw,
            "img_path": img_path,
        }


class COCOData(Dataset):
    """COCO dataset loader"""

    def __init__(
        self,
        root: str,
        split: str,
        flip: bool = True,
        img_res: int = 640,
        cell_res: int = 80,
    ):
        super().__init__()
        self.root = root
        self.split = split
        self.flip = flip
        self.img_res = img_res
        self.cell_res = cell_res

        # Initialize COCO API
        anno_file = os.path.join(root, f"annotations/instances_{split}2017.json")
        self.coco = COCO(anno_file)

        # Get all image IDs and filter out images without annotations
        all_image_ids = list(self.coco.imgs.keys())
        self.image_ids = []
        self.image_categories = {}

        for img_id in all_image_ids:
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            anns = self.coco.loadAnns(ann_ids)
            categories = set(ann["category_id"] for ann in anns)
            if categories:  # keep only images with annotations
                self.image_ids.append(img_id)
                self.image_categories[img_id] = list(categories)

        if len(self.image_ids) == 0:
            raise RuntimeError(f"No valid annotated images found in {split} dataset")

        # Category name dictionary
        self.category_names = {
            cat["id"]: cat["name"] for cat in self.coco.loadCats(self.coco.getCatIds())
        }

    def padding_and_resize(self, img, res, keep_sum=False):
        """Same padding_and_resize method as in FSCData"""
        h, w = img.shape[:2]
        ratio = max(h, w) / res
        new_h, new_w = int(h / ratio), int(w / ratio)

        # resize
        if len(img.shape) == 3:
            resized = cv2.resize(img, (new_w, new_h))
        else:
            resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # padding
        pad_h1, pad_h2 = (res - new_h) // 2, (res - new_h + 1) // 2
        pad_w1, pad_w2 = (res - new_w) // 2, (res - new_w + 1) // 2

        if len(img.shape) == 3:
            padded = np.pad(
                resized, ((pad_h1, pad_h2), (pad_w1, pad_w2), (0, 0)), mode="constant"
            )
        else:
            padded = np.pad(
                resized, ((pad_h1, pad_h2), (pad_w1, pad_w2)), mode="constant"
            )

        if keep_sum:
            padded_sum = np.sum(padded)
            if padded_sum > 0:
                padded = padded * np.sum(img) / padded_sum

        return padded

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]

        # Randomly select a category
        category_id = np.random.choice(self.image_categories[img_id])

        # Load image
        img_path = os.path.join(self.root, f"{self.split}2017", img_info["file_name"])
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        original_hw = img.shape[:2]
        h, w = img.shape[:2]

        # Get all annotations for the category
        ann_ids = self.coco.getAnnIds(imgIds=img_id, catIds=[category_id])
        anns = self.coco.loadAnns(ann_ids)

        # First pad and resize the original image to the target size
        img = self.padding_and_resize(img, self.img_res)

        # Initialize maps at cell_res (80x80) scale
        proportion_map = np.zeros((self.cell_res, self.cell_res), dtype=np.float32)
        category_map = np.zeros((self.cell_res, self.cell_res), dtype=np.float32)

        # Process each annotation that belongs to the current category
        for ann in anns:
            if ann["category_id"] == category_id:
                # Get instance mask and resize to cell_res
                instance_mask = self.coco.annToMask(ann)
                instance_mask = self.padding_and_resize(
                    instance_mask.astype(np.float32), self.cell_res
                )
                # Binarize mask
                instance_mask = (instance_mask > 0).astype(np.float32)

                # Update category map
                category_map = np.logical_or(category_map, instance_mask).astype(
                    np.float32
                )

                # Compute the number of pixels occupied by the instance
                instance_pixels = np.sum(instance_mask)
                if instance_pixels > 0:
                    # Fill 1/instance_pixels at the instance location
                    proportion_map += instance_mask / instance_pixels

        # Convert to tensor
        img = torch.from_numpy(img).float().permute(2, 0, 1) / 255.0
        proportion_map = torch.from_numpy(proportion_map).float().unsqueeze(0)
        category_map = torch.from_numpy(category_map).float().unsqueeze(0)

        # Random horizontal flip
        if self.flip and np.random.random() > 0.5:
            img = torch.flip(img, [2])
            proportion_map = torch.flip(proportion_map, [2])
            category_map = torch.flip(category_map, [2])

        category_map = (category_map > 0).clone().detach().float()

        return {
            "image": img,
            "proportion_label": proportion_map,
            "category_label": category_map,
            "text_label": self.category_names[category_id],
            "image_id": img_id,
            "count": float(
                len([ann for ann in anns if ann["category_id"] == category_id])
            ),
            "original_hw": original_hw,
        }

    def __len__(self):
        return len(self.image_ids)


class LVISData(Dataset):
    """LVIS dataset loader"""

    def __init__(
        self,
        root: str,
        split: str,
        flip: bool = True,
        img_res: int = 640,
        cell_res: int = 80,
    ):
        super().__init__()
        self.root = root
        self.split = split
        self.flip = flip
        self.img_res = img_res
        self.cell_res = cell_res

        # Use correct LVIS annotation file naming
        anno_file = os.path.join(root, f"annotations/lvis_v1_{split}.json")
        self.lvis = LVIS(anno_file)

        # Get all image IDs and filter out images without annotations
        all_image_ids = list(self.lvis.imgs.keys())
        self.image_ids = []
        self.image_categories = {}

        for img_id in all_image_ids:
            img_info = self.lvis.load_imgs([img_id])[0]
            file_name = img_info["coco_url"].split("/")[-1]
            img_path = os.path.join(self.root, f"{self.split}2017", file_name)
            if not os.path.exists(img_path):
                continue
            ann_ids = self.lvis.get_ann_ids(img_ids=[img_id])
            anns = self.lvis.load_anns(ann_ids)
            categories = set(ann["category_id"] for ann in anns)
            if categories:
                self.image_ids.append(img_id)
                self.image_categories[img_id] = list(categories)

        if len(self.image_ids) == 0:
            raise RuntimeError(f"No valid annotated images found in {split} dataset")

        # Category names dictionary
        self.category_names = {
            cat["id"]: cat["name"] for cat in self.lvis.cats.values()
        }

    def padding_and_resize(self, img, res, keep_sum=False):
        """Same padding_and_resize method as in FSCData"""
        h, w = img.shape[:2]
        ratio = max(h, w) / res
        new_h, new_w = int(h / ratio), int(w / ratio)

        # resize
        if len(img.shape) == 3:
            resized = cv2.resize(img, (new_w, new_h))
        else:
            resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # padding
        pad_h1, pad_h2 = (res - new_h) // 2, (res - new_h + 1) // 2
        pad_w1, pad_w2 = (res - new_w) // 2, (res - new_w + 1) // 2

        if len(img.shape) == 3:
            padded = np.pad(
                resized, ((pad_h1, pad_h2), (pad_w1, pad_w2), (0, 0)), mode="constant"
            )
        else:
            padded = np.pad(
                resized, ((pad_h1, pad_h2), (pad_w1, pad_w2)), mode="constant"
            )

        if keep_sum:
            padded_sum = np.sum(padded)
            if padded_sum > 0:
                padded = padded * np.sum(img) / padded_sum

        return padded

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_info = self.lvis.load_imgs([img_id])[0]

        # Update: extract filename from coco_url
        file_name = img_info["coco_url"].split("/")[-1]
        img_path = os.path.join(self.root, f"{self.split}2017", file_name)

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        original_hw = img.shape[:2]

        # Get all annotations for the category
        ann_ids = self.lvis.get_ann_ids(img_ids=[img_id])
        anns = self.lvis.load_anns(ann_ids)

        # Randomly select a category ID
        category_id = np.random.choice(self.image_categories[img_id])

        img = self.padding_and_resize(img, self.img_res)

        proportion_map = np.zeros((self.cell_res, self.cell_res), dtype=np.float32)
        category_map = np.zeros((self.cell_res, self.cell_res), dtype=np.float32)

        for ann in anns:
            if ann["category_id"] == category_id:
                instance_mask = self.lvis.ann_to_mask(ann)
                instance_mask = self.padding_and_resize(
                    instance_mask.astype(np.float32), self.cell_res
                )
                instance_mask = (instance_mask > 0).astype(np.float32)

                category_map = np.logical_or(category_map, instance_mask).astype(
                    np.float32
                )

                instance_pixels = np.sum(instance_mask)
                if instance_pixels > 0:
                    proportion_map += instance_mask / instance_pixels

        img = torch.from_numpy(img).float().permute(2, 0, 1) / 255.0
        proportion_map = torch.from_numpy(proportion_map).float().unsqueeze(0)
        category_map = torch.from_numpy(category_map).float().unsqueeze(0)

        if self.flip and np.random.random() > 0.5:
            img = torch.flip(img, [2])
            proportion_map = torch.flip(proportion_map, [2])
            category_map = torch.flip(category_map, [2])

        category_map = (category_map > 0).clone().detach().float()

        return {
            "image": img,
            "proportion_label": proportion_map,
            "category_label": category_map,
            "text_label": self.category_names[category_id],
            "image_id": img_id,
            "count": float(
                len([ann for ann in anns if ann["category_id"] == category_id])
            ),
            "original_hw": original_hw,
        }

    def __len__(self):
        return len(self.image_ids)


class Obj365Data(Dataset):
    def __init__(self, root, novel_only=True, split="validation"):
        self.hf_id = "jxu124/objects365"
        self.novel_only = novel_only
        self.split = split
        self.root = root
        self.novel_categories = []
        self.hf_dataset = load_dataset(self.hf_id, split=self.split)
        self.img_res = 640

        if self.novel_only:
            self.novel_categories = self.get_novel_categories()
            self.hf_dataset = self.filter_novel_data(self.hf_dataset)

    def get_novel_categories(self):
        novel_json_path = os.path.join(self.root, "annotations/novel_obj365.json")
        with open(novel_json_path, "r") as f:
            novel_data = json.load(f)
        novel_categories = []
        for key in novel_data.keys():
            category = key.split("'")[1]
            novel_categories.append(category.lower())
        return novel_categories

    def filter_novel_data(self, dataset):
        novel_set = set(self.novel_categories)

        def has_novel(sample):
            sample_categories = [ann["category"].lower() for ann in sample["anns_info"]]
            return any(cat in novel_set for cat in sample_categories)

        filtered_dataset = dataset.filter(has_novel)

        def keep_only_novel_anns(sample):
            sample["anns_info"] = [
                ann
                for ann in sample["anns_info"]
                if ann["category"].lower() in novel_set
            ]
            return sample

        filtered_dataset = filtered_dataset.map(keep_only_novel_anns)
        return filtered_dataset

    def padding_and_resize(self, img, res, keep_sum=False):
        h, w = img.shape[:2]
        ratio = max(h, w) / res
        new_h, new_w = int(h / ratio), int(w / ratio)

        if len(img.shape) == 3:
            resized = cv2.resize(img, (new_w, new_h))
        else:
            resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        pad_h1, pad_h2 = (res - new_h) // 2, (res - new_h + 1) // 2
        pad_w1, pad_w2 = (res - new_w) // 2, (res - new_w + 1) // 2

        if len(img.shape) == 3:
            padded = np.pad(
                resized, ((pad_h1, pad_h2), (pad_w1, pad_w2), (0, 0)), mode="constant"
            )
        else:
            padded = np.pad(
                resized, ((pad_h1, pad_h2), (pad_w1, pad_w2)), mode="constant"
            )

        if keep_sum:
            padded_sum = np.sum(padded)
            if padded_sum > 0:
                padded = padded * np.sum(img) / padded_sum

        return padded

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        hf_info = self.hf_dataset[idx]
        image_path = os.path.join(self.root, hf_info["image_path"])
        anns_info = hf_info["anns_info"]
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Image not found: {image_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.padding_and_resize(img, self.img_res)
        img = torch.from_numpy(img).float().permute(2, 0, 1) / 255.0

        # Count instances for each novel category
        category_count = {}
        for ann in anns_info:
            category = ann["category"].lower()
            if category not in category_count:
                category_count[category] = 0
            category_count[category] += 1
        # Randomly select a novel category
        category = np.random.choice(list(category_count.keys()))
        count = category_count[category]
        # Convert np.str_ to str
        category = str(category)

        return {"image": img, "text_label": category, "count": float(count)}


class OImgv7Data(Dataset):
    def __init__(
        self,
        root: str,
        split: str,
        flip: bool = False,
        novel_only: bool = True,
        img_res: int = 640,
        cell_res: int = 80,
    ):
        super().__init__()
        self.root = root
        self.split = split
        self.flip = flip
        self.img_res = img_res
        self.cell_res = cell_res
        self.novel_only = novel_only
        self.novel_categories = []

        anno_file = os.path.join(root, f"labels.json")
        self.coco = COCO(anno_file)

        all_image_ids = list(self.coco.imgs.keys())
        self.image_ids = []
        self.image_categories = {}

        for img_id in all_image_ids:
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            anns = self.coco.loadAnns(ann_ids)
            categories = set(ann["category_id"] for ann in anns)
            if categories:
                self.image_ids.append(img_id)
                self.image_categories[img_id] = list(categories)

        if len(self.image_ids) == 0:
            raise RuntimeError(f"No valid annotated images found in {split} dataset")

        self.category_names = {
            cat["id"]: cat["name"] for cat in self.coco.loadCats(self.coco.getCatIds())
        }

        if self.novel_only:
            self.novel_categories = self.get_novel_categories()
            self.filter_novel_data()

    def get_novel_categories(self):
        novel_json_path = os.path.join(self.root, "annotations/novel_oimgv7.json")
        with open(novel_json_path, "r") as f:
            novel_data = json.load(f)
        novel_categories = []
        for key in novel_data.keys():
            category = key.split("'")[1]
            novel_categories.append(category.lower())
        return novel_categories

    def filter_novel_data(self):
        novel_set = set(self.novel_categories)
        filtered_image_ids = []
        filtered_image_categories = {}

        for img_id in self.image_ids:
            has_novel = False
            novel_cats_in_img = []
            for cat_id in self.image_categories[img_id]:
                cat_name = self.category_names[cat_id].lower()
                if cat_name in novel_set:
                    has_novel = True
                    novel_cats_in_img.append(cat_id)

            if has_novel:
                filtered_image_ids.append(img_id)
                filtered_image_categories[img_id] = novel_cats_in_img

        self.image_ids = filtered_image_ids
        self.image_categories = filtered_image_categories

    def padding_and_resize(self, img, res, keep_sum=False):
        h, w = img.shape[:2]
        ratio = max(h, w) / res
        new_h, new_w = int(h / ratio), int(w / ratio)

        if len(img.shape) == 3:
            resized = cv2.resize(img, (new_w, new_h))
        else:
            resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        pad_h1, pad_h2 = (res - new_h) // 2, (res - new_h + 1) // 2
        pad_w1, pad_w2 = (res - new_w) // 2, (res - new_w + 1) // 2

        if len(img.shape) == 3:
            padded = np.pad(
                resized, ((pad_h1, pad_h2), (pad_w1, pad_w2), (0, 0)), mode="constant"
            )
        else:
            padded = np.pad(
                resized, ((pad_h1, pad_h2), (pad_w1, pad_w2)), mode="constant"
            )

        if keep_sum:
            padded_sum = np.sum(padded)
            if padded_sum > 0:
                padded = padded * np.sum(img) / padded_sum

        return padded

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]

        category_id = np.random.choice(self.image_categories[img_id])

        img_path = os.path.join(self.root, f"data", img_info["file_name"])
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.padding_and_resize(img, self.img_res)
        img = torch.from_numpy(img).float().permute(2, 0, 1) / 255.0

        ann_ids = self.coco.getAnnIds(imgIds=img_id, catIds=[category_id])
        anns = self.coco.loadAnns(ann_ids)
        count = len([ann for ann in anns if ann["category_id"] == category_id])

        return {
            "image": img,
            "text_label": self.category_names[category_id],
            "count": float(count),
        }

    def __len__(self):
        return len(self.image_ids)
