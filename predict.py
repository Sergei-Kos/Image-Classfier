import argparse
import torch
from PIL import Image
import json
from torchvision import models
import numpy as np


def get_input_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--arch', type=str, help='CNN Model Architecture')
    parser.add_argument('--path', type=str, help='Path to the test picture')
    parser.add_argument('--categories', type=str, help='File with the class names')
    parser.add_argument('--topk', type=int, help='Amount of top classes')
    parser.add_argument('--gpu', type=bool, help='GPU usage')

    return parser.parse_args()


def input_validation(args):
    if args.arch:
        if args.arch == 'vgg' or args.arch == 'densenet':
            checkpoint_path = 'checkpoint_' + args.arch + ".pth"
            if args.arch == 'vgg':
                model = models.vgg16(pretrained=True)
            elif args.arch == 'densenet':
                model = models.densenet121(pretrained=True)
        else:
            print('Invalid model choice. Using checkpoint of the default model: densenet')
            checkpoint_path = 'checkpoint_densenet.pth'
            model = models.densenet121(pretrained=True)
    else:
        print('Using checkpoint of the default model: densenet')
        checkpoint_path = 'checkpoint_densenet.pth'
        model = models.densenet121(pretrained=True)

    if args.path:
        path = args.path
    else:
        path = "flowers/test/15/image_06351.jpg"
        print("Default picture path: {}".format(path))

    if args.categories:
        categories = args.categories
    else:
        categories = "cat_to_name.json"
        print("Default categories path: {}".format(categories))

    if args.topk:
        topk = args.topk
    else:
        topk = 5
        print("Default amount of top classes: {}".format(topk))

    if args.gpu:
        if args.gpu:
            if torch.cuda.is_available():
                device = torch.device("cuda")
            else:
                device = torch.device("cpu")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")
        print("Using default cpu")

    return checkpoint_path, model, path, categories, topk, device


def get_checkpoint(path, model):
    checkpoint = torch.load(path)

    for param in model.parameters():
        param.requires_grad = False

    model.classifier = checkpoint['classifier']
    model.class_to_idx = checkpoint['class_to_index_connection']
    model.load_state_dict(checkpoint['state_dict'])

    return model


def process_image(imagepath):
    im = Image.open(imagepath)

    width, height = im.size

    shortest_side = 256
    if width > height:
        ratio = width / height
        height = shortest_side
        width = shortest_side * ratio
    elif height > width:
        ratio = height / width
        width = shortest_side
        height = shortest_side * ratio
    else:
        height = shortest_side
        width = shortest_side

    im.thumbnail((width, height), Image.ANTIALIAS)

    x0 = (width - 224) / 2
    y0 = (height - 224) / 2
    x1 = x0 + 224
    y1 = y0 + 224

    im = im.crop((x0, y0, x1, y1))

    image_to_np_array = np.array(im) / 255

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    im = (image_to_np_array - mean) / std

    final_image = im.transpose((2, 0, 1))

    return final_image


def main():
    in_arg = get_input_args()
    print(in_arg.arch, in_arg.path, in_arg.categories, in_arg.topk, in_arg.gpu)
    checkpoint_path, model, path, categories, topk, device = input_validation(in_arg)

    with open(categories, 'r') as f:
        cat_to_name = json.load(f)

    model = get_checkpoint(checkpoint_path, model)

    model.eval()
    model.to(device)

    array_image = process_image(path)

    tensor = torch.from_numpy(array_image).float().unsqueeze(0)
    tensor = tensor.to(device)

    log_predictions = model.forward(tensor)
    predictions = torch.exp(log_predictions)
    top_probs, top_classes = predictions.topk(topk)

    top_probs = np.array(top_probs.detach())[0]
    top_classes = np.array(top_classes.detach())[0]

    idx_to_class = {value: key for key, value in model.class_to_idx.items()}
    top_classes = [idx_to_class[label] for label in top_classes]

    print("Top probabilities: {} ".format(top_probs))
    print("Top classes: {} ".format(top_classes))


if __name__ == "__main__": main()
