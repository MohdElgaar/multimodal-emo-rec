from concurrent.futures import ThreadPoolExecutor
import os, sys
from PIL import Image
from tqdm import tqdm
from facenet_pytorch import InceptionResnetV1, fixed_image_standardization
import torch
import numpy as np
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
resnet = InceptionResnetV1(pretrained='vggface2').to(device).eval()

def process(fn):
    img_dir = os.path.join(out_dir,folder,'%s_imgs'%fn)
    out_file = os.path.join(out_dir,folder,'%s_img.npz'%fn)
    num_frames = len(os.listdir(img_dir))
    if num_frames > 0:
        imgs = []
        for frame in range(num_frames):
            img = Image.open(os.path.join(img_dir,"frame{:03d}.jpg".format(frame))).resize((160,160))
            imgs.append(np.asarray(img).transpose(2,0,1).astype('float32'))
        imgs = np.stack(imgs)
        imgs = fixed_image_standardization(imgs)
        batch_size = 50
        outs = []
        for i in range(0,num_frames,batch_size):
            segment = torch.tensor(imgs[i:i+batch_size]).to(device)
            with torch.no_grad():
                embeddings = resnet(segment).cpu().numpy()
            outs.append(embeddings)
        embeddings = np.concatenate(outs)
    else:
        print("[num_frames == 0] %s"%fn)
        embeddings = np.zeros((1,512)).astype('float32')

    np.savez(out_file,embeddings)


FNULL = open(os.devnull, 'w')
root_dir = "/usr/cs/public/mohd/2020-1/"
out_dir = "/usr/cs/public/mohd/data"
folder = sys.argv[1]
files = [line.strip() for line in open('/usr/cs/public/mohd/%s_data.txt'%folder)]
files.sort()

# with ThreadPoolExecutor(15) as threads:
#     for fn in tqdm(threads.map(process,files), total = len(files)):
#         pass
for fn in tqdm(files):
    process(fn)
    exit()
