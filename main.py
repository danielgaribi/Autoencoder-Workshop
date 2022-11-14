import numpy as np
import torch
import torchvision
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from datetime import datetime
from PIL import Image
import os
import matplotlib.pyplot as plt
import pickle
import argparse


from data import load_data

device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

print(device, " will be used.\n")

def setArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_of_epochs", default=100)
    parser.add_argument("--batch_size", default=20)
    parser.add_argument("--lr", default=0.0001)
    args = parser.parse_args()
    return args

class AutoEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()

        convOutDim = 64 * 16 * 16
        latentDim = 256

        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(3, 8, 3, stride=1, padding=1),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm2d(8),
            torch.nn.Conv2d(8, 16, 3, stride=2, padding=1),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm2d(16),
            torch.nn.Conv2d(16, 16, 3, stride=1, padding=1),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm2d(16),
            torch.nn.Conv2d(16, 32, 3, stride=2, padding=1),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm2d(32),
            torch.nn.Conv2d(32, 32, 3, stride=2, padding=1),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm2d(32),
            torch.nn.Conv2d(32, 64, 3, stride=2, padding=1),
            torch.nn.Flatten(),
            torch.nn.Linear(convOutDim, latentDim)
        )

        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(latentDim, convOutDim),
            torch.nn.Unflatten(1, (64, 16, 16)),
            torch.nn.ConvTranspose2d(64, 32, 3, stride=1, padding=1),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm2d(32),
            torch.nn.ConvTranspose2d(32, 32, 4, stride=2, padding=1),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm2d(32),
            torch.nn.ConvTranspose2d(32, 16, 3, stride=1, padding=1),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm2d(16),
            torch.nn.ConvTranspose2d(16, 16, 4, stride=2, padding=1),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm2d(16),
            torch.nn.ConvTranspose2d(16, 8, 4, stride=2, padding=1),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm2d(8),
            torch.nn.ConvTranspose2d(8, 3, 4, stride=2, padding=1)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

def main(args):
    num_of_epochs = args.num_of_epochs
    batch_size = args.batch_size
    lr = args.lr
    lr_cut_loss = [0.1, 0.075, 0.05]
    lr_cut_factor = 10

    uniq_name = f"{datetime.now().strftime('%m-%d-%Y-%H-%M-%S')}_epoch-{num_of_epochs}_batch_size-{batch_size}_lr-{lr}"

    dataset_path = './dataset/'
    data_loader, val_loader = load_data(dataset_path, batch_size)

    ae = AutoEncoder().to(device)
    criterion1 = torch.nn.L1Loss()
    criterion2 = torch.nn.MSELoss()
    optimizerEncoder = torch.optim.Adam(ae.encoder.parameters(), lr=lr)
    optimizerDecoder = torch.optim.Adam(ae.decoder.parameters(), lr=lr)

    trainning_losses = []
    validation_losses = []
    minValidationLoss = 1

    for epoch in range(num_of_epochs):
        print(f"epoch {epoch}")
        print(f"training...")
        trainning_loss = []
        for images_batch in tqdm(data_loader):
            images_batch = images_batch.to(device)
            optimizerEncoder.zero_grad()
            optimizerDecoder.zero_grad()
            result = ae(images_batch)
            batch_loss = 0.5 * (criterion1(images_batch, result) + criterion2(images_batch, result))
            batch_loss.backward()
            optimizerDecoder.step()
            optimizerEncoder.step()
            trainning_loss.append(torch.mean(batch_loss).cpu().item())

        validation_loss = []
        print(f"Validation...")
        is_first = True
        with torch.no_grad():
            for images_batch in tqdm(val_loader):
                images_batch = images_batch.to(device)
                result = ae(images_batch)
                batch_loss = 0.5 * (criterion1(images_batch, result) + criterion2(images_batch, result))
                validation_loss.append(torch.mean(batch_loss).cpu().item())
                if is_first:
                    is_first = False
                    save_image(images_batch[0], f"epoch_{epoch}_origin", uniq_name)
                    save_image(result[0], f"epoch_{epoch}_rec", uniq_name)

        print(f"epoch {epoch}: trainning_loss={np.mean(trainning_loss)}, validation_loss={np.mean(validation_loss)}")

        trainning_losses.append(np.mean(trainning_loss))
        validation_losses.append(np.mean(validation_loss))

        if np.mean(validation_loss) < minValidationLoss:
            minValidationLoss = np.mean(validation_loss)
            torch.save(ae.state_dict(), os.path.join("out", uniq_name, f"{uniq_name}.pt"))

        if len(lr_cut_loss) != 0 and lr_cut_loss[0] > np.mean(validation_loss):
            print(f"cut lr by {lr_cut_factor}")
            lr = lr / lr_cut_factor
            lr_cut_loss.pop(0)
            optimizerEncoder = torch.optim.Adam(ae.encoder.parameters(), lr=lr)
            optimizerDecoder = torch.optim.Adam(ae.decoder.parameters(), lr=lr)

    save_results(trainning_losses, validation_losses, uniq_name)

def save_results(trainning_losses, validation_losses, uniq_name):
    plt.plot(trainning_losses, label="training loss")
    plt.plot(validation_losses, label="validation loss")
    plt.legend()
    plt.savefig(os.path.join("out", uniq_name, f"{uniq_name}_plot.png"))

    dict = {
        "trainning_losses": trainning_losses,
        "validation_losses": validation_losses
    }
    pickle_file_name = os.path.join("out", uniq_name, f"{uniq_name}_data.pkl")
    with open(pickle_file_name, 'wb') as f:
        pickle.dump(dict, f)


def save_image(x, name, uniq_name):
    img = custom_to_pil(x)
    folder_path = os.path.join("out", uniq_name)
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)

    imgpath = os.path.join(folder_path, f"{name}.png")
    img.save(imgpath)

def custom_to_pil(x):
    x = x.detach().cpu()
    x = torch.clamp(x, -1., 1.)
    x = (x + 1.) / 2.
    x = x.permute(1, 2, 0).numpy()
    x = (255 * x).astype(np.uint8)
    x = Image.fromarray(x)
    if not x.mode == "RGB":
        x = x.convert("RGB")
    return x

args = setArguments()
main(args)