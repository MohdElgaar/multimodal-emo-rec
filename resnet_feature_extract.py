from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import torchvision
from torchvision import models, transforms
import torch
import numpy as np
from PIL import Image
import cv2, os, sys

device = torch.device('cuda')
resnet = models.resnext101_32x8d(pretrained=True).to(device)
resnet.fc = torch.nn.Identity()

t = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]),
    ])

def process(path):
    img_path = os.path.join(out_dir,path + '_imgs')
    out_path = os.path.join(out_dir, "%s_img2.npz"%path)
    if os.path.isfile(out_path):
        return
    num_frames = len(os.listdir(img_path))
    imgs = []
    for idx in range(num_frames):
        im = Image.open(os.path.join(img_path, 'frame%03d.jpg'%idx)).resize((224,224))
        im = cv2.normalize(cv2.cvtColor(np.array(im), cv2.COLOR_BGR2RGB), 0, 1, dtype=cv2.CV_32F)
        im = t(im)
        imgs.append(im)
    bs = 25
    outs = []
    for i in range(0,len(imgs),bs):
        batch = imgs[i:i+bs]
        batch = torch.stack(batch).to(device)
        with torch.no_grad():
            out = resnet(batch)
        out = out.cpu().numpy()
        outs.append(out)
    if len(outs) > 0:
        out = np.concatenate(outs)
    else:
        out = np.zeros((1,512)).astype(np.float32)
    np.savez(out_path, out)

folder = sys.argv[1]
root_dir = "/usr/cs/public/mohd/2020-1/%s"%folder
out_dir = "/usr/cs/public/mohd/data/%s"%folder
os.makedirs(out_dir, exist_ok=True)
os.makedirs(os.path.join(out_dir, folder), exist_ok = True)

if len(sys.argv) > 2:
    files = [line.strip() for line in open('/usr/cs/public/mohd/%s_data.txt'%folder)][int(sys.argv[2]):]
else:
    files = [line.strip() for line in open('/usr/cs/public/mohd/%s_data.txt'%folder)]
n = len(files)
with ThreadPoolExecutor(5) as threads:
    for fn in tqdm(threads.map(process,files), total = n):
        pass
# for fn in tqdm(files, total = n):
#     process(fn)
# process(files[0])
