# predict.py

import argparse
import torch
from common import data_transforms, get_device, get_data_paths, load_cat_to_name


def get_input_args():
    parser = argparse.ArgumentParser(description="Do prediction")

    parser.add_argument(
        "data_dir",
        type=str,
        help="Directory for image folder. E.g. `flowers`",
    )

    parser.add_argument(
        "model_path",
        type=str,
        help="Model checkpoint path. E.g. `./`",
    )

    parser.add_argument(
        "-ip",
        "--image_path",
        type=str,
        default=None,
        help="Image path. If omit pick random from data_dir/test",
    )

    parser.add_argument(
        "-cp",
        "--cat_to_name_path",
        type=str,
        default="./",
        help="cat_to_name file path",
    )

    parser.add_argument(
        "-k",
        "--top_k",
        type=int,
        default=5,
        help="Top K classes",
    )

    parser.add_argument(
        "--gpu",
        type=bool,
        default=True,
        help="Use GPU for training",
    )

    return parser.parse_args()


def load_checkpoint(mpath):
    checkpoint = torch.load(str(mpath + "/checkpoint.pth"))
    model = checkpoint["model"]
    # model.features = checkpoint["features"]

    # Turn off gradient of model.features.parameters to train the classifier
    for para in model.features.parameters():
        para.requires_grad = False

    model.classifier = checkpoint["classifier"]
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.class_to_idx = checkpoint["class_to_idx"]

    return model, checkpoint["class_to_idx"]


def process_image(image_path, image_transform):
    """Scales, crops, and normalizes a PIL image for a PyTorch model,
    returns an Numpy array
    """

    pilImage = Image.open(image_path)
    # image_transform = data_transforms["validTest"]

    return image_transform(pilImage)


def predict(model, device, image, cat_to_name, topk=5):
    image = image.unsqueeze(0)

    # move to device
    image = image.to(device)
    model.to(device)
    model.eval()

    with torch.no_grad():
        output = model(image)
        ps = torch.exp(output)

    # Convert indices to classes
    idx_to_class = {val: key for key, val in model.class_to_idx.items()}
    top_ps, top_classes = ps.topk(topk, dim=1)
    top_labels = [idx_to_class[c] for c in top_classes.tolist()[0]]
    top_flowers = [cat_to_name[str(label)] for label in top_labels]

    return top_ps.tolist()[0], top_labels, top_flowers


def main():
    args = get_input_args()

    print("|- Init device", end="... ")
    device = get_device(args.gpu)
    print("done")

    print("|- Load checkpoint", end="... ")
    (model,) = load_checkpoint(args.model_path)
    model = model.to(device)
    print("done")

    print("|- Load labels", end="... ")
    labels = load_cat_to_name(args.cat_to_name_path)
    print("done")

    print("|- Load image:", end=" ")

    # Process image
    image_path = args.image_path
    if image_path == None:
        _, _, test_dir = get_data_paths(data_dir)
        random_label_idx = random.choice(os.listdir(test_dir))
        random_image_file = random.choice(
            os.listdir("{}/{}".format(test_dir, random_label_idx))
        )
        image_path = os.path.join(test_dir, random_label_idx, random_image_file)
    print("{}{}".format("(random) " if args.image_path == None else "", image_path))

    print("|- Process image", end="... ")
    image = process_image(
        image_path=image_path, image_transform=data_transforms["validTest"]
    )
    print("done")

    print("|- Do prediction", end="... ")
    top_ps, top_labels, top_flowers = predict(model, device, image, labels, args.top_k)

    print("   Result:")
    print("   |- Top probabilities: ", top_ps)
    print("   |- Top labels: ", top_labels)
    print("   \\- Top flower names: ", top_flowers)

    return 0


if __name__ == "__main__":
    main()
