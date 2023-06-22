import torch
from torchvision import transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Lambda, ToPILImage, CenterCrop, Resize
from torchvision import transforms
from torch.optim import Adam
from datasets import load_dataset
import numpy as np

from diffusion.DDMFunctions import DDMFunctions #Githubで変更

class Train():
    def __init__(self, model, image_size, channels, timesteps, dataset_name, schedule_type, device):
        self.image_size = image_size
        self.channels = channels
        self.timesteps = timesteps
        self.device = device
        self.model = model

        self.dm = DDMFunctions(timesteps, schedule_type)
        self.dataset = load_dataset(dataset_name)
        self.optimizer = Adam(self.model.parameters(), lr=1e-3)
        self.results_folder = self.make_results_folder()

        # define image transformations (e.g. using torchvision)
        self.transform = Compose([
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Lambda(lambda t: (t * 2) - 1)
        ])


    def make_results_folder(self):
        from pathlib import Path
        results_folder = Path("./results")
        results_folder.mkdir(exist_ok = True)
        return results_folder
            
    def transforms(self, examples):
        examples["pixel_values"] = [self.transform(image.convert("L")) for image in examples["image"]]
        del examples["image"]

        return examples

        return examples

    def train(self, epochs, save_and_sample_every=1000, batch_size=128):
        transformed_dataset = self.dataset.with_transform(self.transforms).remove_columns("label")

        # create dataloader
        dataloader = DataLoader(transformed_dataset["train"], batch_size=batch_size, shuffle=True)

        for epoch in range(epochs):
            for step, batch in enumerate(dataloader):
                self.optimizer.zero_grad()

                batch_size = batch["pixel_values"].shape[0]
                batch = batch["pixel_values"].to(self.device)

                # Algorithm 1 line 3: sample t uniformally for every example in the batch
                t = torch.randint(0, self.timesteps, (batch_size,), device=self.device).long()

                loss = self.dm.p_losses(self.model, batch, t, loss_type="huber")

                if step % 100 == 0:
                    print("Loss:", loss.item())

                loss.backward()
                self.optimizer.step()

                # save generated images
                if step != 0 and step % save_and_sample_every == 0:
                    milestone = step // save_and_sample_every
                    nrow = self.timesteps // save_and_sample_every
                    batches = self.dm.num_to_groups(4, batch_size)
                    # all_images_list = list(map(lambda n: self.dm.sample(self.model, self.image_size, batch_size=n, channels=self.channels), batches))
                    all_images_list = self.dm.sample(self.model, self.image_size, batch_size=1, channels=self.channels)
                    all_images_list_tensor = [torch.tensor(arr) for arr in np.array(all_images_list)]
                    all_images = torch.cat(all_images_list_tensor, dim=0)
                    all_images = (all_images + 1) * 0.5
                    save_image(all_images, str(self.results_folder / f'sample-{milestone}.png'), nrow = nrow)
