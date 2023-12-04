from argparse import ArgumentParser

import torch
from PIL import Image, ImageDraw

from datasets import transforms as T
from models.build import MODELS, build_model


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("model", choices=MODELS)
    parser.add_argument("--device", default="cuda")
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
            T.Normalize([.4303, .4072, .3870], [.0874, .0868, .0911]),
        ])
    filename = "data/wireframe/images/test/00031546.jpg"
    image = Image.open(filename)
    image_tensor, _ = transforms(image, None)
    image_tensor = image_tensor[None].to(args.device)

    with torch.no_grad():
        lines, scores = model(image_tensor)
    lines, scores = lines.cpu(), scores.cpu()
    indices = torch.argsort(scores, descending=True)
    lines, scores = lines[indices], scores[indices]
    n = min((scores > 0.).sum().item(), 500)
    lines, scores = lines[:n], scores[:n]

    lines = lines * 4
    image = Image.open(filename)
    image = image.resize((512, 512))
    draw = ImageDraw.Draw(image)
    for xy in lines:
        xy = xy.tolist()
        draw.line(xy, fill=(0, 255, 0))
    image.save("a.png")
