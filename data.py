from glob import glob
from time import time
from math import ceil
from tqdm import tqdm
import os
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import torch

emotions = ['neu', 'hap', 'ang', 'fea', 'dis', 'sur', 'sad']
emo_map = {k:i for i,k in enumerate(emotions)}

class Collates():
    audio_segment = 20
    img_segment = 10
    step = 30
    
    def collate_fn(self, batch):
      audios, texts, imgs, ys_audio, ys_img, ys_int, spks, gens, ages = [list(x) for x in zip(*batch)]
    
      audio_len = []
      for i, audio in enumerate(audios):
        start_id = np.random.randint(0,max(1,audio.shape[0] - self.step*self.audio_segment + 1))
        audios[i] = audio[start_id:start_id + self.step*self.audio_segment:self.step]
        
      for i, img in enumerate(imgs):
        start_id = np.random.randint(0, max(img.shape[0] - self.step*self.img_segment + 1, 1))
        imgs[i] = img[start_id:start_id + self.step*self.img_segment:self.step]

      text_lens = [text.shape[0] for text in texts]
      max_text_len = max(text_lens)
      for i,text in enumerate(texts):
        n = text_lens[i]
        texts[i] = np.pad(text, ((0,max_text_len-n),(0,0)), mode='wrap')

      img_lens = [img.shape[0] for img in imgs]
      max_img_len = max(img_lens)
      for i,img in enumerate(imgs):
        n = img_lens[i]
        imgs[i] = np.pad(img, ((0,max_img_len-n),(0,0)), mode='wrap')

      audio_lens = [audio.shape[0] for audio in audios]
      max_audio_len = max(audio_lens)
      for i,audio in enumerate(audios):
        n = audio_lens[i]
        audios[i] = np.pad(audio, ((0,max_audio_len-n),(0,0)), mode='wrap')

      res = np.stack(audios), audio_lens, \
        np.stack(texts), text_lens, \
        np.stack(imgs), img_lens, \
        np.array(ys_audio), np.array(ys_img), np.array(ys_int), \
        np.array(spks), np.array(gens), np.array(ages).astype('float32')
      return res

    def val_collate_fn(self, batch):
      entry = batch[0]
      audio = np.expand_dims(entry[0],0)
      text = np.expand_dims(entry[1],0)
      img = np.expand_dims(entry[2],0)
      ys_audio = np.array([entry[3]])
      ys_img = np.array([entry[4]])
      ys_int = np.array([entry[5]])
      spk = np.array([entry[6]])
      gen = np.array([entry[7]])
      age = np.array([entry[8]]).astype('float32')

      audios, imgs = [], []
      img_lens, audio_lens = [], []
      audio_n = audio.shape[1]
      img_n = img.shape[1]

      i = 0
      while True:
        audio_offset = i*self.step//5
        img_offset = i*self.step//5

        if audio_offset >= audio_n and img_offset >= img_n:
            break

        if audio_offset < audio_n:
          audios.append(audio[:, audio_offset:audio_offset + self.step*self.audio_segment:self.step, :])
        else:
          start_id = np.random.randint(0, max(1,audio_n - self.step*self.audio_segment + 1))
          audios.append(audio[:, start_id:start_id+self.step*self.audio_segment:self.step, :])

        if img_offset < img_n:
            imgs.append(img[:, img_offset:img_offset + self.step*self.img_segment:self.step, :])
        else:
          start_id = np.random.randint(0, max(img_n - self.step*self.img_segment + 1, 1))
          imgs.append(img[:, start_id:start_id+self.step*self.img_segment:self.step, :])
        img_lens.append(imgs[-1].shape[1])
        audio_lens.append(audios[-1].shape[1])
        i += 1

      text_lens = [text.shape[1]]

      res = audios, audio_lens, text, text_lens, imgs, img_lens, ys_audio, ys_img, ys_int, spk, gen, age
      return res

    def test_collate_fn(self, batch):
      entry = batch[0]
      audio = np.expand_dims(entry[0],0)
      text = np.expand_dims(entry[1],0)
      img = np.expand_dims(entry[2],0)

      audios, imgs = [], []
      img_lens, audio_lens = [], []
      audio_n = audio.shape[1]
      img_n = img.shape[1]
      audio_segments = ceil(audio_n/self.audio_segment)
      img_segments = ceil(img_n/self.img_segment)

      i = 0
      while True:
        audio_offset = i*self.audio_segment//2
        img_offset = i*self.img_segment//2

        if audio_offset >= audio_n and img_offset >= img_n:
            break

        if audio_offset < audio_n:
          audios.append(audio[:, audio_offset:audio_offset + self.audio_segment, :])
        else:
          start_id = np.random.randint(0, audio_n - self.audio_segment + 1)
          audios.append(audio[:, start_id:start_id+self.audio_segment, :])

        if img_offset < img_n:
            imgs.append(img[:, img_offset:img_offset + self.img_segment, :])
        else:
          start_id = np.random.randint(0, max(img_n - self.img_segment + 1, 1))
          imgs.append(img[:, start_id:start_id+self.img_segment, :])
        img_lens.append(imgs[-1].shape[1])
        audio_lens.append(audios[-1].shape[1])
        i += 1

      text_lens = [text.shape[1]]

      res = audios, audio_lens, text, text_lens, imgs, img_lens
      return res

class MyDataset(Dataset):
  def __init__(self, data_path, data_txt, mode='train', size=-1):
    super().__init__()
    files = [x.rstrip() for x in open(data_txt)]
    self.files = files
    self.path = data_path
    self.mode = mode
    if size == -1:
        self.size = len(files)
    else:
        self.size = min(len(files),size)
    self.update_data()

  def update_data(self):
      np.random.shuffle(self.files)
      self.texts, self.audios, self.imgs = [], [], []
      for fn in tqdm(self.files[:self.size]):
          self.texts.append(np.load(os.path.join(self.path, "%s.npz"%fn[:5]))['word_embed'])
          self.audios.append(np.load(os.path.join(self.path, "%s_pase.npy"%fn)))
          self.imgs.append(np.load(os.path.join(self.path, "%s_img.npz"%fn))['arr_0'].astype('float32'))

  def __getitem__(self, idx):
    audio_data = self.audios[idx]
    text_data = self.texts[idx]
    img_data = self.imgs[idx]
#     audio_data = np.ones((1,1))
#     img_data = np.ones((1,1))
    
    if self.mode == 'test':
        return audio_data, text_data, img_data

    fn = self.files[idx]
    fn_split = fn.split('-')
    y_int = emo_map[fn_split[-3]]
    y_img = emo_map[fn_split[-2]]
    y_audio = emo_map[fn_split[-1]]
    
    spk = int(fn_split[2])
    gen= 0 if fn_split[3] == 'm' else 1
    age = int(fn_split[2])

    return audio_data, text_data, img_data, y_audio, y_img, y_int, spk, gen, age

  def __len__(self):
    return self.size
