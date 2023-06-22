""" Functions.py """
#@title Functions

# transform_

import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import torch
import torch.nn.functional as F
from torchvision.transforms import Compose, ToTensor, Lambda, ToPILImage, CenterCrop, Resize
from diffusion.Scheduler import Scheduler

class DMFunctions():
    def __init__(self, timesteps, schedule_type):
        
        scheduler = Scheduler(timesteps)
        # βスケジュールの設定
        if schedule_type == 'linear':
            # デフォルトは線形スケジュール
            # self.betas = torch.linspace(0.0001, 0.02, timesteps)
            self.betas = scheduler.linear_beta_schedule(timesteps)
            
        elif schedule_type == 'cosine':
            self.betas = scheduler.cosine_beta_schedule(timesteps)
            
        else:
            pass
        
        self.timesteps = timesteps #300 # 1000

        # alphasの定義
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0) # 累積積
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)

        # 拡散プロセス q(x_t | x_{t-1})等の計算
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod) # 累積積の平方根
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod) # 累積積の平方根の補数

        # 事後確率 q(x_{t-1} | x_t, x_0)の計算
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)

        self.noise_list = []

    def extract(self, a, t, x_shape):
        batch_size = t.shape[0]
        out = a.gather(-1, t.cpu())
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


    def reverse_transform(self, im):
        im = im.squeeze().numpy().transpose(1, 2, 0)
        im = (im + 1.0) / 2 * 255
        im = im.astype(np.uint8)
        return im


    # @title 画像の表示の定義
    def plot(self, x_noisy, timestep=None):
        noisy_image = self.reverse_transform(x_noisy)
        if timestep:
            text = "Step:" + str(timestep)
            plt.text(0, 0, text, fontdict=None, bbox=dict(facecolor='white', alpha=1))
        plt.axis('off')
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
    def q_sample(self, x_start, t, mode=None, noise=None):
        """ mode:確認モードの種類 """
        if noise is None:
            noise = torch.randn_like(x_start)

        # 元画像の強さ
        sqrt_alphas_cumprod_t = self.extract(self.sqrt_alphas_cumprod, t, x_start.shape)

        # ノイズの強さ
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
            # 崩壊画像
            q = sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

            self.noise_list.append(noise)

        return q

    def test(self):
        print('Test2')

    # # 画像を指定のサイズに切り取って、値域を0-255から -1.0 - +1.0 に変換
    # def transform_(self, image_size):
    #     return Compose([
    #         Resize(image_size),
    #         CenterCrop(image_size),
    #         ToTensor(),
    #         Lambda(lambda t: (t * 2) - 1),
    #     ])


    def transforms(self, examples):
       examples["pixel_values"] = [transform(image.convert("L")) for image in examples["image"]]
       del examples["image"]
    
       return examples
