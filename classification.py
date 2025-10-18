import os
import argparse

import timm
from torchvision import transforms
import torch
torch.hub.set_dir("./cache")
import sys
from PIL import Image

from tqdm import tqdm
sys.path.append("")

theme_available=["Abstractionism", "Artist_Sketch", "Blossom_Season", "Bricks", "Byzantine", "Cartoon",
 "Cold_Warm", "Color_Fantasy", "Comic_Etch", "Crayon", "Cubism", "Dadaism", "Dapple",
 "Defoliation", "Early_Autumn", "Expressionism", "Fauvism", "French", "Glowing_Sunset",
 "Gorgeous_Love", "Greenfield", "Impressionism", "Ink_Art", "Joy", "Liquid_Dreams",
 "Magic_Cube", "Meta_Physics", "Meteor_Shower", "Monet", "Mosaic", "Neon_Lines", "On_Fire",
 "Pastel", "Pencil_Drawing", "Picasso", "Pop_Art", "Red_Blue_Ink", "Rust", "Seed_Images",
 "Sketch", "Sponge_Dabbed", "Structuralism", "Superstring", "Surrealism", "Ukiyoe",
 "Van_Gogh", "Vibrant_Flow", "Warm_Love", "Warm_Smear", "Watercolor", "Winter"]

class_available = ["Architectures", "Bears", "Birds", "Butterfly", "Cats", "Dogs", "Fishes", "Flame", "Flowers",
                   "Frogs", "Horses", "Human", "Jellyfish", "Rabbits", "Sandwiches", "Sea", "Statues", "Towers",
                   "Trees", "Waterfalls"]

# def a function to return the index of the theme in the theme_available list given a test_theme avoid using torch.where
def get_theme_idx(test_theme):
    for idx, theme in enumerate(theme_available):
        if theme == test_theme:
            return idx

if __name__ == "__main__":

    # Parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--theme", type=str, default=None)
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--ckpt", type=str, default="style50_cls.pth")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    input_dir = os.path.join(args.input_dir, args.theme) if args.theme is not None else args.input_dir

    # Create folder if not exist
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, f"{args.theme}.pth") if args.theme is not None else os.path.join(args.output_dir, "result.pth")

    model = timm.create_model("vit_large_patch16_224.augreg_in21k", pretrained=True).to(device)
    num_classes = len(class_available)
    model.head = torch.nn.Linear(1024, num_classes).to(device)
    # load checkpoint
    model.load_state_dict(torch.load(args.ckpt, map_location=device)["model_state_dict"])
    model.eval()

    results = {}
    results["test_theme"] = args.theme if args.theme is not None else "sd"
    results["input_dir"] = args.input_dir
    results["loss"] = {class_: 0.0 for class_ in class_available}
    results["acc"] = {class_: 0.0 for class_ in class_available}
    results["pred_loss"] = {class_: 0.0 for class_ in class_available}
    results["misclassified"] = {class_: {other_class: 0 for other_class in class_available} for class_ in class_available}

    # Initialize misclassification record in the results dictionary

    image_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    
    for test_theme in tqdm(theme_available):
        for i in range(1, 6):
            for idx, object_class in enumerate(class_available):
                theme_label = idx
                img_path = os.path.join(input_dir, f"{test_theme}_{object_class}_0{i}.png")
                image = Image.open(img_path)
                target_image = image_transform(image).unsqueeze(0).to(device)
                with torch.no_grad():
                    res = model(target_image)
                    label = torch.tensor([theme_label]).to(device)
                    loss = torch.nn.functional.cross_entropy(res, label)
                    # softmax the prediction
                    res_softmax = torch.nn.functional.softmax(res, dim=1)
                    pred_loss = res_softmax[0][theme_label]
                    pred_success = (torch.argmax(res) == theme_label).sum()
                    pred_label = torch.argmax(res)

                results["loss"][object_class] += loss
                results["pred_loss"][object_class] += pred_loss
                results["acc"][object_class] += (pred_success * 1.0 / (len(theme_available) * 5))
                misclassified_as = class_available[pred_label.item()]
                results["misclassified"][object_class][misclassified_as] += 1

        if not args.dry_run:
            torch.save(results, output_path)