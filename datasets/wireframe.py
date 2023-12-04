import contextlib
import os
from typing import Callable, Literal, Optional

from PIL import Image
from pycocotools.coco import COCO
from torchvision.datasets import VisionDataset


class WireframeDataset(VisionDataset):
    def __init__(self, root: str, split: Literal["train", "tese"],
                 transforms: Optional[Callable] = None):
        super().__init__(root, transforms)
        self.split = split
        with contextlib.redirect_stdout(None):
            self.coco = COCO(self.annotation_file)
        self.ids = sorted(self.coco.imgs.keys())

    def _load_image(self, id: int) -> Image.Image:
        path = self.coco.loadImgs(id)[0]["file_name"]
        return Image.open(os.path.join(self.image_dir, path)).convert("RGB")

    def _load_annotation(self, id: int):
        ann = self.coco.loadAnns(self.coco.getAnnIds(id))[0]
        del ann["id"], ann["image_id"]
        ann.update({"file_name": self.coco.loadImgs(id)[0]["file_name"]})
        return ann

    def __getitem__(self, index):
        id = self.ids[index]
        img = self._load_image(id)
        ann = self._load_annotation(id)

        if self.transforms is not None:
            img, ann = self.transforms(img, ann)

        return img, ann

    def __len__(self) -> int:
        return len(self.ids)

    @property
    def image_dir(self) -> str:
        return os.path.join(self.root, "images", self.split)

    @property
    def annotation_file(self) -> str:
        return os.path.join(self.root, self.split + ".json")
