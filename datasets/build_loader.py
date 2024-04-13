import torch
from .coco import CoCo
from .transforms import get_transform
from torchvision.datasets import ImageFolder
import webdataset as wds

def my_collate_fn(samples):
    images, texts = zip(*samples)
    images = torch.stack(images)
    return images, texts




def build_loader(cfg):
	if cfg.dataset.name == "coco":
		train_ds = CoCo(cfg, dataType='train2017', annType='captions', is_train=True)
		
		if cfg.dataset.params.train_test_split:
			train_size = int(cfg.dataset.params.train_test_split * len(train_ds))
			val_size = len(train_ds) - train_size
			train_ds, val_ds = torch.utils.data.random_split(train_ds, [train_size, val_size])

		else:
			val_ds = CoCo(cfg, dataType='val2017', annType='captions', is_train=False)

	if cfg.dataset.name == "imagenet":
		train_ds = ImageFolder(root=cfg.dataset.params.train_path, transform=get_transform(cfg))
		if cfg.dataset.params.train_test_split:
			train_size = int(cfg.dataset.params.train_test_split * len(train_ds))
			val_size = len(train_ds) - train_size
			train_ds, val_ds = torch.utils.data.random_split(train_ds, [train_size, val_size])
		else:
			assert False, "Train test split is required for imagenet dataset"
   
	if cfg.dataset.name == "webdataset":
		dataset = wds.DataPipeline(
			wds.SimpleShardList(cfg.dataset.params.train_path),

			# at this point we have an iterator over all the shards
			wds.shuffle(100),

			# add wds.split_by_node here if you are using multiple nodes
			wds.split_by_worker,

			# at this point, we have an iterator over the shards assigned to each worker
			wds.tarfile_to_samples(),

			# this shuffles the samples in memory
			wds.shuffle(1000),

			# this decodes the images and json
			wds.decode("pil"),
			wds.to_tuple("jpg", "txt"),
			wds.map_tuple(get_transform(cfg), None),
			wds.shuffle(1000),
			# wds.batched(cfg.dataset.params.batch_size)
		)
  
		train_dl = wds.WebLoader(dataset, batch_size=cfg.dataset.params.batch_size)
  
		return train_dl, train_dl






	train_dl = torch.utils.data.DataLoader(train_ds,
											batch_size=cfg.dataset.params.batch_size, 
											shuffle=cfg.dataset.params.shuffle, 
											num_workers=cfg.dataset.params.num_workers)  
 

	val_dl = torch.utils.data.DataLoader(val_ds,
											batch_size=cfg.dataset.params.batch_size, 
											shuffle=cfg.dataset.params.shuffle, 
											num_workers=cfg.dataset.params.num_workers) 
 

	return (train_dl, val_dl)


			
			
 