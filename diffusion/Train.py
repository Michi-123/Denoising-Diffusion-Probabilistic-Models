from torchvision import transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Lambda, ToPILImage, CenterCrop, Resize
from torchvision import transforms
from torch.optim import Adam
from datasets import load_dataset

from diffusion.DMFunctions import DMFunctions #Githubで変更

class Train():
    def __init__(self, model, image_size, timesteps, dataset_name, results_folder, device):
        self.model = model
        self.results_folder = results_folder
        self.image_size = image_size
        self.optimizer = Adam(self.model.parameters(), lr=1e-3)
        self.dataset = load_dataset(dataset_name)
        self.dm = DMFunctions(timesteps)
        self.device = device

        # define image transformations (e.g. using torchvision)
        self.transform = Compose([
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Lambda(lambda t: (t * 2) - 1)
        ])
    
    def transforms(self, examples):
        examples["pixel_values"] = [self.transform(image.convert("L")) for image in examples["image"]]
        del examples["image"]

        return examples

        return examples

    def train(self, epochs, save_and_sample_every=10, batch_size=128):
        transformed_dataset = self.dataset.with_transform(self.transforms).remove_columns("label")

        # create dataloader
        dataloader = DataLoader(transformed_dataset["train"], batch_size=batch_size, shuffle=True)

        for epoch in range(epochs):
            for step, batch in enumerate(dataloader):
                self.optimizer.zero_grad()

                batch_size = batch["pixel_values"].shape[0]
                batch = batch["pixel_values"].to(self.device)

                # Algorithm 1 line 3: sample t uniformally for every example in the batch
                t = torch.randint(0, timesteps, (batch_size,), device=self.device).long()

                loss = dm.p_losses(self.model, batch, t, loss_type="huber")

                if step % 100 == 0:
                    print("Loss:", loss.item())

                loss.backward()
                self.optimizer.step()

                # save generated images
                if step != 0 and step % save_and_sample_every == 0:
                    milestone = step // save_and_sample_every
                    batches = dm.num_to_groups(4, batch_size)
                    all_images_list = list(map(lambda n: dm.sample(self.model, batch_size=n, channels=channels), batches))
                    all_images = torch.cat(all_images_list, dim=0)
                    all_images = (all_images + 1) * 0.5
                    save_image(all_images, str(self.results_folder / f'sample-{milestone}.png'), nrow = 6)


    # def transforms(self, examples):
        # 画像を指定のサイズに切り取って、値域を0-255から -1.0 - +1.0 に変換
        # transform = Compose([
        #     Resize(self.image_size),
        #     CenterCrop(self.image_size),
        #     ToTensor(),
        #     Lambda(lambda t: (t * 2) - 1),
        # ])
