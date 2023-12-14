import json
import os
import pickle
import shutil
import zipfile
from argparse import ArgumentParser

from tqdm import tqdm


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("dataset", choices=["wireframe", "yorkurban"])
    parser.add_argument("root")
    parser.add_argument("--output-dir", default="./data")
    args = parser.parse_args()

    if args.dataset == "wireframe":
        if not os.path.exists(os.path.join(args.root, "pointlines")):
            with zipfile.ZipFile(os.path.join(args.root, "pointlines.zip")) as f:
                f.extractall(args.root)
        if not os.path.exists(os.path.join(args.root, "v1.1")):
            with zipfile.ZipFile(os.path.join(args.root, "v1.1.zip")) as f:
                f.extractall(args.root)
        os.makedirs(os.path.join(args.output_dir, "wireframe"))
        os.makedirs(os.path.join(args.output_dir, "wireframe", "images"))
        for split in ("train", "test"):
            image_names = []
            with open(os.path.join(args.root, "v1.1", "{}.txt".format(split))) as f:
                for row in f:
                    row = row.strip()
                    if row != "":
                        image_names.append(row)
            image_names.sort()
            images = []
            annos = []
            for id, image_name in enumerate(tqdm(image_names, desc=split), start=1):
                pkl_path = os.path.join(args.root, "pointlines", image_name.replace(".jpg", ".pkl"))
                with open(pkl_path, "rb") as f:
                    anno = pickle.load(f)
                images.append({
                    "id": id,
                    "file_name": image_name,
                })
                lines = anno["lines"]
                for i in range(len(lines)):
                    lines[i] = list(map(int, lines[i]))
                points = anno["points"]
                for i in range(len(points)):
                    points[i] = list(map(float, points[i]))
                annos.append({
                    "id": id,
                    "image_id": id,
                    "lines": lines,
                    "points": points,
                })
            dataset = {
                "info": {
                    "description": "Wireframe Dataset (CVPR 2018)",
                    "url": "https://github.com/huangkuns/wireframe",
                    "year": "2018",
                    "date_created": "2021/05/07",
                },
                "images": images,
                "annotations": annos,
            }
            print("dumping json...")
            with open(os.path.join(args.output_dir, "wireframe", "{}.json".format(split)), "w") as f:
                json.dump(dataset, f)
            print("copying images...")
            shutil.copytree(os.path.join(args.root, "v1.1", split),
                            os.path.join(args.output_dir, "wireframe", "images", split),
                            dirs_exist_ok=True)
    else:
        raise NotImplementedError()
