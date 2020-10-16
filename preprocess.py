from concurrent.futures import ThreadPoolExecutor
import subprocess
from PIL import Image
import cv2
from glob import glob
from tqdm import tqdm
import os, shutil, sys
from video_feature_extract import crop_folder
from time import time
import numpy as np

def process_audio(f, folder):
    fname = f.split('/')[-1]
    fdir = os.path.join(out_dir, 'tmp', fname.replace('mp4', 'wav'))
    command = "ffmpeg -y -i %s -ac 2 -f wav %s"%(f, fdir)
    r = subprocess.call(command, shell=True, stdout=FNULL, stderr=subprocess.STDOUT)

    txt_file = fdir.replace('wav', 'csv')
    command = "/usr/cs/grad/doc/melgaar/opensmile-2.3.0/bin/linux_x64_standalone_static/SMILExtract -C /usr/cs/grad/doc/melgaar/opensmile-2.3.0/config/emobase.conf -I %s -O %s"%(fdir, txt_file)
    r = subprocess.call(command, shell=True, stdout=FNULL, stderr=subprocess.STDOUT)

    with open(txt_file, 'r') as f:
        s = f.readlines()[-1].strip()
    s = s.split(',')[1:-1]
    s = ",".join(s)
    with open(txt_file.replace('tmp', folder), 'w') as f:
        f.write(s)

    os.remove(txt_file)
    os.remove(fdir)

def process_vid(fn):
    out_file = os.path.join(out_dir,folder,fn + '_img_new.npz')
    # if os.path.isfile(outfile):
    #     return
    vid_dir = os.path.join(out_dir, folder, fn + '_imgs')
    os.makedirs(vid_dir, exist_ok=True)
    path = os.path.join(root_dir, folder, fn + '.mp4')
    vidcap = cv2.VideoCapture(path)
    success,image = vidcap.read()
    count = 0
    imgs = []
    while success:
      # cv2.imwrite(os.path.join(vid_dir,"frame{:03d}.jpg".format(count)), image)
      imgs.append(image)
      success,image = vidcap.read()
      count += 1
    vidcap.release()
    images, embeddings = crop_folder(imgs, out_file)
    for i,image in enumerate(images):
        image = Image.fromarray(image)
        image.save(os.path.join(vid_dir,"frame{:03d}.jpg".format(i)))
    np.savez(out_file, embeddings)
    

def process(fn):
    process_vid(fn)
        # process_audio(f, folder)
    # elif f.endswith('npz'):
        # shutil.copyfile(f, os.path.join(out_dir, folder, f.split('/')[-1]))

FNULL = open(os.devnull, 'w')
root_dir = "/usr/cs/public/2020-1/"
out_dir = "/usr/cs/public/data"
os.makedirs(out_dir, exist_ok=True)
os.makedirs(os.path.join(out_dir,'tmp'), exist_ok=True)

folder = sys.argv[1]
os.makedirs(os.path.join(out_dir, folder), exist_ok = True)
if len(sys.argv) > 2:
    files = [line.strip() for line in open('/usr/cs/public/%s_data.txt'%folder)][int(sys.argv[2]):]
else:
    files = [line.strip() for line in open('/usr/cs/public/%s_data.txt'%folder)]
n = len(files)
with ThreadPoolExecutor(10) as threads:
    for fn in tqdm(threads.map(process,files), total = n):
        pass
# for fn in tqdm(files, total = n):
#     process(fn)
# process(files[0])
