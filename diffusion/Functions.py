""" Functions.py """
#@title Functions

import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import torch
import torch.nn.functional as F

class Functions():
    def __init__(self, timesteps):    

        self.timesteps = timesteps #300 # 1000

        self.betas = torch.linspace(0.0001, 0.02, timesteps)
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)

    def extract(self, a, t, x_shape):
        batch_size = t.shape[0]
        out = a.gather(-1, t.cpu())
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


    def reverse_transform(self, im):
        im = im.squeeze().numpy().transpose(1, 2, 0)
        im = (im + 1.0) / 2 * 255
        im = im.astype(np.uint8)
        return im


    # forward diffusion (using the nice property)
    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = self.extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = self.extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise


    # @title 画像の表示の定義
    def plot(self, x_noisy):
        noisy_image = self.reverse_transform(x_noisy)
        text = "Step:" + str(t)
        plt.text(5, 20, text, fontdict=None, bbox=dict(facecolor='white', alpha=1))
        plt.imshow(noisy_image)
        plt.show()

    def p_losses(self, denoise_model, x_start, t, noise=None, loss_type="l1"):
        if noise is None:
            noise = torch.randn_like(x_start)

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        predicted_noise = denoise_model(x_noisy, t)

        if loss_type == 'l1':
            loss = F.l1_loss(noise, predicted_noise)
        elif loss_type == 'l2':
            loss = F.mse_loss(noise, predicted_noise)
        elif loss_type == "huber":
            loss = F.smooth_l1_loss(noise, predicted_noise)
        else:
            raise NotImplementedError()

        return loss


    @torch.no_grad()
    def p_sample(self, model, x, t, t_index):
        betas_t = self.extract(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = self.extract(
            self.sqrt_one_minus_alphas_cumprod, t, x.shape
        )
        sqrt_recip_alphas_t = self.extract(self.sqrt_recip_alphas, t, x.shape)

        # Equation 11 in the paper
        # Use our model (noise predictor) to predict the mean
        model_mean = sqrt_recip_alphas_t * (
            x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t
        )

        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = self.extract(self.posterior_variance, t, x.shape)
            noise = torch.randn_like(x)
            # Algorithm 2 line 4:
            return model_mean + torch.sqrt(posterior_variance_t) * noise


    # Algorithm 2 but save all images:
    @torch.no_grad()
    def p_sample_loop(self, model, shape):
        device = next(model.parameters()).device

        b = shape[0]
        # start from pure noise (for each example in the batch)
        img = torch.randn(shape, device=device)
        imgs = []

        for i in tqdm(reversed(range(0, self.timesteps)), desc='sampling loop time step', total=self.timesteps):
            img = self.p_sample(model, img, torch.full((b,), i, device=device, dtype=torch.long), i)
            imgs.append(img.cpu().numpy())
        return imgs


    @torch.no_grad()
    def sample(self, model, image_size, batch_size=16, channels=3):
        return self.p_sample_loop(model, shape=(batch_size, channels, image_size, image_size))


    def num_to_groups(self, num, divisor):
        groups = num // divisor
        remainder = num % divisor
        arr = [divisor] * groups
        if remainder > 0:
            arr.append(remainder)
        return arr


    def exists(self, x):
        return x is not None

    def default(self, val, d):
        if self.exists(val):
            return val
        return d() if self.isfunction(d) else d


    def num_to_groups(self, num, divisor):
        groups = num // divisor
        remainder = num % divisor
        arr = [divisor] * groups
        if remainder > 0:
            arr.append(remainder)
        return arr

    # @title サンプリングの定義
    def q_sample(self, x_start, t, mode=None):
        """ mode:確認モードの種類 """

        t = torch.tensor([t])
        noise = torch.randn_like(x_start)
        #
        sqrt_alphas_cumprod_t = self.extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = self.extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )
        if mode == 1:
            # 検証（元画像の強さ）
            q = sqrt_alphas_cumprod_t * x_start
        elif mode == 2:
            # 検証（ノイズの強さ）
            q = sqrt_one_minus_alphas_cumprod_t * noise
        else:
            q = sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

            noise_list.append(noise)

        return q


    # @title 画像の表示の定義
    def plot(self, x_noisy):
        noisy_image = self.reverse_transform(x_noisy)
        text = "Step:" + str(t)
        plt.text(5, 20, text, fontdict=None, bbox=dict(facecolor='white', alpha=1))
        plt.imshow(noisy_image)
        plt.show()

    # 画像を指定のサイズに切り取って、値域を0-255から -1.0 - +1.0 に変換
    def transform(self, image_size):
        return Compose([
            Resize(image_size),
            CenterCrop(image_size),
            ToTensor(),
            Lambda(lambda t: (t * 2) - 1),
        ])

