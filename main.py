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


from data import load_data

device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

print(device, " will be used.\n")

num_of_epochs = 50
batch_size = 10
lr = 0.0001

uniq_name = f"{datetime.now().strftime('%m-%d-%Y-%H-%M-%S')}_epoch-{num_of_epochs}_batch_size-{batch_size}_lr-{lr}"

class AutoEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()

        nof_conv_layers = 6
        c = [3, 4, 8, 8, 16, 16, 1]
        kernelSizes = [3] * nof_conv_layers
        stride = [1, 2, 1, 2, 2, 2]
        padding = [1] * nof_conv_layers
        convOutDim = 16 * 16 * 1
        latentDim = 256

        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(3, 4, 3, stride=1, padding=1),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm2d(4),
            torch.nn.Conv2d(4, 8, 3, stride=2, padding=1),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm2d(8),
            torch.nn.Conv2d(8, 8, 3, stride=1, padding=1),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm2d(8),
            torch.nn.Conv2d(8, 16, 3, stride=2, padding=1),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm2d(16),
            torch.nn.Conv2d(16, 16, 3, stride=2, padding=1),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm2d(16),
            torch.nn.Conv2d(16, 1, 3, stride=2, padding=1),
            torch.nn.Flatten(),
            torch.nn.Linear(convOutDim, latentDim)
        )

        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(latentDim, convOutDim),
            torch.nn.Unflatten(1, (1, 16, 16)),
            torch.nn.ConvTranspose2d(1, 4, 3, stride=1, padding=1),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm2d(4),
            torch.nn.ConvTranspose2d(4, 8, 4, stride=2, padding=1),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm2d(8),
            torch.nn.ConvTranspose2d(8, 16, 3, stride=1, padding=1),
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

        # Elayers = []
        # for i in range(nof_conv_layers):
        #     Elayers.append(torch.nn.Conv2d(c[i], c[i + 1], kernelSizes[i], stride=stride[i], padding=padding[i]))
        #     Elayers.append(torch.nn.LeakyReLU())
        #     Elayers.append(torch.nn.BatchNorm2d(c[i + 1]))
        #
        # Elayers.append(torch.nn.Flatten())
        # # Elayers.append(torch.nn.Linear(convOutDim ,latentDim))
        #
        # self.encoder = torch.nn.Sequential(*Elayers)
        #
        # Dlayers = []
        # # Dlayers.append(torch.nn.Linear(latentDim, convOutDim))
        # Dlayers.append(torch.nn.Unflatten(1, (1, 16, 16)))
        #
        # for i in reversed(range(nof_conv_layers)):
        #     Dlayers.append(torch.nn.ConvTranspose2d(c[i + 1], c[i], kernelSizes[i], stride=stride[i], padding=0))
        #     Dlayers.append(torch.nn.LeakyReLU())
        #     Dlayers.append(torch.nn.BatchNorm2d(c[i]))
        #
        # self.decoder = torch.nn.Sequential(*Dlayers)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


def main():
    dataset_path = './dataset/'
    data_loader, val_loader = load_data(dataset_path, batch_size)

    ae = AutoEncoder().to(device)
    criterion = torch.nn.L1Loss()
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
            batch_loss = criterion(images_batch, result)
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
                batch_loss = criterion(images_batch, result)
                validation_loss.append(torch.mean(batch_loss).cpu().item())
                if is_first:
                    is_first = False
                    save_image(images_batch[0], f"epoch_{epoch}_origin")
                    save_image(result[0], f"epoch_{epoch}_rec")

        print(f"epoch {epoch}: trainning_loss={np.mean(trainning_loss)}, validation_loss={np.mean(validation_loss)}")

        trainning_losses.append(np.mean(trainning_loss))
        validation_losses.append(np.mean(validation_loss))

        if np.mean(validation_loss) < minValidationLoss:
            minValidationLoss = np.mean(validation_loss)
            torch.save(ae.state_dict(), os.path.join("out", uniq_name, f"{uniq_name}.pt"))

    save_results(trainning_losses, validation_losses)

def save_results(trainning_losses, validation_losses):
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


def save_image(x, name):
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

main()