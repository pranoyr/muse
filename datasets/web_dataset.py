import matplotlib.pyplot as plt
import torch.utils.data
import torch.nn
from random import randrange
import os
import cv2
os.environ["WDS_VERBOSE_CACHE"] = "1"
from torchvision import transforms as T 

import webdataset as wds


transform = T.Compose([
    # to pil
    T.ToPILImage(),
    T.Resize((256, 256)),
    T.ToTensor()
])



url = "http://192.168.2.8:8000/mscoco/{00000}.tar"
# dataset = wds.WebDataset(url).shuffle(1000).decode("rgb").to_tuple("jpg","txt")


# pipe_line = [wds.WebDataset(url).shuffle(1000).decode("rgb").to_tuple("jpg","txt")]



dataset = (wds.WebDataset(url)
           .shuffle(1000)
           .decode("rgb")
           .to_tuple("jpg", "txt")
           .map_tuple(transform, None))

train_dl = wds.WebLoader(dataset, batch_size=2, num_workers=0)


for batch in train_dl:
    img, txt = batch
    # img = img.permute(1,2,0)  * 255
    # img = img.numpy()
    # cv2.imwrite("image.jpg", img)

    # cv2.imshow("image", img)
    
    print(img.shape, txt)
    # cv2.waitKey(0)
    
    
   

# print(dataset)


# for image in dataset:
#     print(image.shape)
#     # plt.imshow(image)
#     # plt.show()
#     # break