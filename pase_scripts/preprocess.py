import os, sys
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from pase.models.frontend import wf_builder
import torch
from glob import glob
import librosa
import numpy as np

def load_wav(path):
      return librosa.core.load(path, sr=16000)[0]

pase = wf_builder('cfg/frontend/PASE+.cfg').eval()
pase.load_pretrained('FE_e199.ckpt', load_last=True)
pase = pase.cuda()

def process(fn):
    fn = os.path.join(root_dir, fn + '.wav')
    out_fn = os.path.join(out_dir, "%s_pase.npy"%os.path.basename(fn)[:-4])
    if os.path.isfile(out_fn):
        return
    wav = load_wav(fn).reshape(1,1,-1)
    n = wav.shape[-1]
    max_size = 200000
    outs = []
    offset = 0
    while offset < n:
        segment = wav[:,:,offset:offset+max_size]
        if segment.shape[2] < 1000:
            break
        segment = torch.tensor(segment).cuda()
        with torch.no_grad():
            out = pase(segment)
        out = out.squeeze().cpu().numpy().transpose()
        outs.append(out)
        offset += max_size
    out = np.concatenate(outs,0)
    np.save(out_fn, out)

# root_dir = '/usr/cs/public/mohd/MOSEI/Audio/Full/WAV_16000'
def main(folder):
    # files = glob('/usr/cs/public/mohd/data/%s/*.wav'%folder)
    files = [line.strip() for line in open('/usr/cs/public/mohd/%s_data.txt'%folder)]
    print(folder, len(files))
    # with ThreadPoolExecutor() as threads:
    #     for fn in tqdm(threads.map(process,files), total = len(files)):
    #         pass
    for fn in tqdm(files):
        process(fn)

if __name__ == '__main__':
    folder = sys.argv[1]
    root_dir = '/usr/cs/public/mohd/data/%s'%folder
    out_dir = '/usr/cs/public/mohd/data/%s'%folder
    if folder == 'all':
        for folder in ['train', 'val', 'test1']:
            main(folder)
    else:
        main(folder)
