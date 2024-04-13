import sys
# sys.path.append(".")
from PIL import Image

# also disable grad to save memory
import torch




import yaml
import torch
import torchvision.transforms.functional as TF
from omegaconf import OmegaConf
from models.vae.taming.models.vqgan import VQModel, GumbelVQ
import PIL


import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import cv2
import os

local_rank = os.getenv("LOCAL_RANK", default=-1)

class VQVAE:

	def __init__(self, config_path, ckpt_path, is_gumbel=False):
		
		self.is_gumbel = is_gumbel
			
		config16384 = self.load_config(config_path)
		self.model = self.load_vqgan(config16384, ckpt_path=ckpt_path)
		self.model = self.model.to(torch.device(f"cuda:{local_rank}"))
		


	def preprocess(self, img, target_image_size=256, map_dalle=True):
		s = min(img.size)
		
		if s < target_image_size:
			raise ValueError(f'min dim for image {s} < {target_image_size}')
			
		r = target_image_size / s
		s = (round(r * img.size[1]), round(r * img.size[0]))
		img = TF.resize(img, s, interpolation=PIL.Image.LANCZOS)
		img = TF.center_crop(img, output_size=2 * [target_image_size])
		img = torch.unsqueeze(T.ToTensor()(img), 0)
		return img

	def load_config(self, config_path, display=False):
		config = OmegaConf.load(config_path)
		if display:
			print(yaml.dump(OmegaConf.to_container(config)))
		return config

	def load_vqgan(self, config, ckpt_path=None, is_gumbel=False):
		if self.is_gumbel:
			model = GumbelVQ(**config.model.params)
		else:
			model = VQModel(**config.model.params)
		if ckpt_path is not None:
			sd = torch.load(ckpt_path)["state_dict"]
			missing, unexpected = model.load_state_dict(sd, strict=False)
		return model.eval()

	def preprocess_vqgan(self, x):
		x = 2.*x - 1.
		return x

	def custom_to_pil(self, x):
		x = x.detach().cpu()
		x = torch.clamp(x, -1., 1.)
		x = (x + 1.)/2.
		x = x.permute(1,2,0).numpy()
		x = (255*x).astype(np.uint8)
		#to rgb
		x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
	#   x = Image.fromarray(x)
	#   if not x.mode == "RGB":
	#     x = x.convert("RGB")
		return x

	# def reconstruct_with_vqgan(x):
	# 	# could also use model(x) for reconstruction but use explicit encoding and decoding here
	# 	z, _, [_, _, indices] =  self.model.encode(x)
	# 	# tp long
	# 	return 
	# 	indices = indices.unsqueeze(0)
	# 	print(f"VQGAN --- {model.__class__.__name__}: latent shape: {z.shape[2:]}")
	# 	xrec = model.decode_code(indices)
	# 	return xrec

	def decode_indices(self, indices):
		imgs = self.model.decode_code(indices)
		return imgs

	def encode_imgs(self, x, size=256):
		# x_vqgan = preprocess(img, target_image_size=size, map_dalle=False)
		
		b = x.shape[0]
		x = self.preprocess_vqgan(x)
  
		
	
		
		z, _, [_, _, indices] =  self.model.encode(x)
		indices = indices.chunk(b)
		indices = torch.stack(indices, dim=0)
		return indices



# model = VQGAN("ckpt/model.yaml", "ckpt/last.ckpt")

# img = Image.open("images/000000128658.jpg")
# img = np.array(img)

# img = cv2.resize(img, (256, 256))
# img = T.ToTensor()(img)

# x = torch.stack([img, img, img], dim=0)

# indices = model.encode_imgs(x)
# imgs = model.decode_indices(indices)

# draw_img = model.custom_to_pil(imgs[0])
# cv2.imwrite("test.jpg", draw_img)

# print(indices[0].shape)
# print(imgs[0].shape)