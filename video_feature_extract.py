from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import os, shutil
import numpy as np
import cv2
import time
from PIL import Image
from glob import glob

from mtcnn import FaceCropperAll
# from torchvision.models import inception_v3
# from torchvision import transforms
# from torch.nn import Identity
import torch
from facenet_pytorch import InceptionResnetV1, fixed_image_standardization


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
# cnn = inception_v3(pretrained=True).to(device)
# cnn.fc = Identity()
# cnn.eval()
# t = transforms.Compose([transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]), ])

resnet = InceptionResnetV1(pretrained='vggface2').to(device).eval()
torch.no_grad()


# load models
facecropper = FaceCropperAll(type='mtcnn', resizeFactor=1.0)

def test_input(parentPath, interval=10):

    frame_seq = os.listdir(parentPath)
    frame_seq.sort()

    input_list = []

    max_second = 10
    cnt = 0

    N = len(frame_seq)
    # if(len(frame_seq) < max_second * interval):
    #     N = len(frame_seq)
    # else:
    #     N = max_second * interval

    for i in range(0, N, interval):
        frame_seqAbsPath = os.path.join(parentPath, frame_seq[i])
        try:
            img = cv2.imread(frame_seqAbsPath)
            input_list.append(img)
        except Exception as ex:
            print("Error:", ex)

    return input_list


def testFaceCropper(input_list):
    global facecropper
    res, cnt = facecropper.detectMulti(input_list)
    if(res is None):
        print('TEST FAIL FACE CROPPER')
        return None
    return res, cnt


def feature_extract(out_file,imgs):

    interval = 1  # Use 1 frame for 1 second (1 frame/sec)
    # img = test_input(frame_path, interval)
    # face cropper
    resCropped, cnt = testFaceCropper(imgs)

    # not detected
    # if cnt == 0:
    #     print("[cnt == 0] %s"%out_file)

    # VGGface feature
    return resCropped

data_dir = '/usr/cs/public/data/train'


from time import time
def crop_folder(imgs,out_file):
    resCropped = feature_extract(out_file,imgs)
    # os.makedirs(out_dir,exist_ok=True)
    batch_size = 60
    outs = []
    for i in range(0,len(resCropped),batch_size):
        segment = []
        for i in range(i, min(len(resCropped), i+batch_size)):
            im = Image.fromarray(resCropped[i]).resize((160,160))
            segment.append(np.asarray(im).transpose(2,0,1).astype(np.float32))
        segment = torch.tensor(np.stack(segment)).to(device)
        segment = fixed_image_standardization(segment)
        with torch.no_grad():
            out = resnet(segment).detach().cpu().numpy()
        outs.append(out)
    if len(outs) > 0:
        outs = np.concatenate(outs)
        return resCropped, outs
    else:
        return resCropped, np.zeros((1,512)).astype(np.float32)
    #     # np.save(os.path.join(out_dir, "frame%03d.npy"%i), out)
    # imgs = np.concatenate(outs)
    # for img in out:
    #     img = np.expand_dims(img,0)
    #     np.save(os.path.join(out_dir, "frame%03d.npy"%i), img)
    # shutil.rmtree(path)
    # output = np.stack(resCropped)
    # np.save(out_file, output)

if __name__ == '__main__':
    files = [x.rstrip() for x in open('/usr/cs/public/mohd/train_data.txt', 'r')]
# files = files[8000:]
# with ThreadPoolExecutor(2) as threads:
    for folder in tqdm(files, total = len(files)):
            process(folder)
