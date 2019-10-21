import argparse
from time import time
import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
import numpy as np


def get_input_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--arch', type=str, help='CNN Model Architecture')
    parser.add_argument('--rate', type=float, help='Learning rate')
    parser.add_argument('--hiddenUnits', type=str, help='Hidden units of the model')
    parser.add_argument('--epochs', type=int, help='Learning epochs')
    parser.add_argument('--gpu', type=bool, help='GPU usage')

    return parser.parse_args()


def input_validation(args):
    input_parameters = 1024
    if args.arch:
        if args.arch == 'vgg':
            model = models.vgg16(pretrained=True)
            input_parameters = 25088
        elif args.arch == 'densenet':
            model = models.densenet121(pretrained=True)
        else:
            print('Invalid model choice. Using default model: densenet')
            model = models.densenet121(pretrained=True)
    else:
        arch = 'densenet121'
        model = models.densenet121(pretrained=True)

    if args.rate:
        rate = args.rate
    else:
        rate = 0.001
        print("Default rate: {}".format(rate))

    if args.hiddenUnits:
        hidden_units_split = args.hiddenUnits.split(',')
        if len(hidden_units_split) < 3:
            hidden_units_array = [int(unit) for unit in hidden_units_split]
        else:
            print("More than 2 values. Taking only first two")
            hidden_units_array = hidden_units_split[:2]
    else:
        hidden_units_array = [512, 256]
        print("Default amout of hidden_units: {}".format(hidden_units_array))

    if args.epochs:
        epochs = args.epochs
    else:
        epochs = 5
        print("Default amout of epochs: {}".format(epochs))

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

    return model, input_parameters, rate, hidden_units_array, epochs, device


def create_model(model, input_parameters, hidden_units_array, class_to_idx):
    for param in model.parameters():
        param.requires_grad = False

    if len(hidden_units_array) == 1:
        model.classifier = nn.Sequential(nn.Linear(input_parameters, hidden_units_array[0]),
                                         nn.ReLU(),
                                         nn.Dropout(0.2),
                                         nn.Linear(hidden_units_array[0], 102),
                                         nn.LogSoftmax(dim=1))
    else:
        model.classifier = nn.Sequential(nn.Linear(input_parameters, hidden_units_array[0]),
                                         nn.ReLU(),
                                         nn.Dropout(0.2),
                                         nn.Linear(hidden_units_array[0], hidden_units_array[1]),
                                         nn.ReLU(),
                                         nn.Dropout(0.2),
                                         nn.Linear(hidden_units_array[1], 102),
                                         nn.LogSoftmax(dim=1))

    model.class_to_idx = class_to_idx
    return model


def validation(model, device, criterion, data_loader):
    checking_loss = 0
    accuracy = 0
    model.eval()

    # Turn off gradients for validation
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            log_ps = model.forward(inputs)
            checking_loss += criterion(log_ps, labels)

            ps = torch.exp(log_ps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor))

    return checking_loss / len(data_loader), accuracy / len(data_loader)


def main():
    start_time = time()

    in_arg = get_input_args()
    print(in_arg.arch, in_arg.rate, in_arg.hiddenUnits, in_arg.epochs, in_arg.gpu)
    model, input_parameters, rate, hidden_units_array, epochs, device = input_validation(in_arg)

    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

    test_and_validation_transforms = transforms.Compose([transforms.Resize(255),
                                                         transforms.CenterCrop(224),
                                                         transforms.ToTensor(),
                                                         transforms.Normalize([0.485, 0.456, 0.406],
                                                                              [0.229, 0.224, 0.225])])

    data_dir = 'flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    validation_data = datasets.ImageFolder(valid_dir, transform=test_and_validation_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_and_validation_transforms)

    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    validationloader = torch.utils.data.DataLoader(validation_data, batch_size=64)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=64)

    model = create_model(model, input_parameters, hidden_units_array, train_data.class_to_idx)

    criterion = nn.NLLLoss()
    learning_rate = rate
    optimizer = optim.Adam(model.classifier.parameters(), rate)

    model.to(device)

    for e in range(epochs):
        running_loss = 0
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            log_ps = model.forward(inputs)
            loss = criterion(log_ps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        else:
            validation_loss, validation_accuracy = validation(model, device, criterion, validationloader)

            print("Epoch: {}/{}.. ".format(e + 1, epochs),
                  "Training Loss: {:.3f}.. ".format(running_loss / len(trainloader)),
                  "Validation Loss: {:.3f}.. ".format(validation_loss),
                  "Validation Accuracy: {:.3f}".format(validation_accuracy))

            running_loss = 0
            model.train()

    checkpoint = {'input_size': input_parameters,
                  'hidden_layers': hidden_units_array,
                  'output_size': 102,
                  'classifier': model.classifier,
                  'learning_rate': rate,
                  'criterion': criterion,
                  'epochs': epochs,
                  'class_to_index_connection': model.class_to_idx,
                  'optimizer': optimizer.state_dict(),
                  'state_dict': model.state_dict()}

    torch.save(checkpoint, 'checkpoint_' + in_arg.arch + ".pth")

    end_time = time()
    tot_time = end_time - start_time

    print("\n** Total Elapsed Runtime:",
          str(int((tot_time / 3600))) + ":" + str(int((tot_time % 3600) / 60)) + ":"
          + str(int((tot_time % 3600) % 60)))


if __name__ == "__main__": main()
