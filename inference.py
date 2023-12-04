import os
from argparse import ArgumentParser

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import transforms as T
from datasets.wireframe import WireframeDataset
from models.build import MODELS, build_model


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("dataset", choices=["wireframe", "yorkurban"])
    parser.add_argument("model", choices=MODELS)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", default="outputs/")
    args = parser.parse_args()

    model = build_model(args.model)
    model.to(args.device)
    model.eval()

    if args.model.startswith("letr"):
        transforms = T.Compose([
            T.Resize(1100),
            T.ToTensor(),
            T.Normalize([.538, .494, .453], [.257, .263, .273]),
        ])
    else:
        transforms = T.Compose([
            T.Resize([512, 512]),
            T.ToTensor(),
            # T.Normalize([.4303, .4072, .3870], [.0874, .0868, .0911]),
            T.Normalize([.430, .407, .387], [.087, .087, .091]),
        ])
    dataset = WireframeDataset(os.path.join("data", args.dataset),
                               split="test", transforms=transforms)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    output_dir = os.path.join(args.output_dir, args.dataset, args.model)
    os.makedirs(output_dir, exist_ok=True)
    for images, annos in tqdm(dataloader):
        images = images.to(args.device)
        with torch.no_grad():
            lines, scores = model(images)
        lines, scores = lines.cpu().numpy(), scores.cpu().numpy()
        file_name = annos["file_name"][0]
        np.savez(os.path.join(output_dir, file_name.replace(".jpg", "")),
                 lines=lines, scores=scores)
