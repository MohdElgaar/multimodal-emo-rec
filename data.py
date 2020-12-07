from glob import glob
from time import time
from math import ceil
from tqdm import tqdm
import os
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import torch

emotions = ['hap', 'sad', 'fea', 'sur', 'ang', 'dis', 'neu']

emo_map = {k:i for i,k in enumerate(emotions)}

class Collates():
    audio_segment = 30
    audio_step = 1
    img_segment = 30
    text_segment = 30
    text_step = 1
    img_step = 1
    fixed_size = False
    
    def collate_fn(self, batch):
      audios, texts, imgs, ys_int = [list(x) for x in zip(*batch)]
    
#       audio_mean = np.array([audio.mean(0) for audio in audios])
#       img_mean = np.array([img.mean(0) for img in imgs])
    
      audio_len = []
      for i, audio in enumerate(audios):
#         start_id = np.random.randint(0,max(1,audio.shape[0] - self.audio_step*self.audio_segment + 1))
        start_id = np.random.randint(0,audio.shape[0])
        audios[i] = audio[start_id:start_id + self.audio_step*self.audio_segment:self.audio_step]
        
      for i, img in enumerate(imgs):
#         start_id = np.random.randint(0, max(img.shape[0] - self.img_step*self.img_segment + 1, 1))
        start_id = np.random.randint(0, img.shape[0])
        imgs[i] = img[start_id:start_id + self.img_step*self.img_segment:self.img_step]
        
      for i, text in enumerate(texts):
#         start_id = np.random.randint(0, max(img.shape[0] - self.img_step*self.img_segment + 1, 1))
        start_id = np.random.randint(0, text.shape[0])
        texts[i] = text[start_id:start_id + self.text_step*self.text_segment:self.text_step]

#       text_lens = [text.shape[0] for text in texts]
#       max_text_len = max(text_lens)
#       for i,text in enumerate(texts):
#         n = text_lens[i]
#         texts[i] = np.pad(text, ((0,max_text_len-n),(0,0)), mode='wrap')

      img_lens = [img.shape[0] for img in imgs]
      if self.fixed_size:
        max_img_len = self.img_segment
      else:
        max_img_len = max(img_lens)
      for i,img in enumerate(imgs):
        n = img_lens[i]
        imgs[i] = np.pad(img, ((0,max_img_len-n),(0,0)), mode='wrap')

      audio_lens = [audio.shape[0] for audio in audios]
      if self.fixed_size:
        max_audio_len = self.audio_segment
      else:
        max_audio_len = max(audio_lens)
      for i,audio in enumerate(audios):
        n = audio_lens[i]
        audios[i] = np.pad(audio, ((0,max_audio_len-n),(0,0)), mode='wrap')
        
      text_lens = [text.shape[0] for text in texts]
      if self.fixed_size:
        max_text_len = self.text_segment
      else:
        max_text_len = max(text_lens)
      for i,text in enumerate(texts):
        n = text_lens[i]
        texts[i] = np.pad(text, ((0,max_text_len-n),(0,0)), mode='wrap')

        
#       audio_mean = np.repeat(np.expand_dims(audio_mean, 1), max_audio_len, 1)
#       img_mean = np.repeat(np.expand_dims(img_mean, 1), max_img_len, 1)
        
      audios = np.stack(audios)
#       audios = np.concatenate([audios, audio_mean], -1)
      imgs = np.stack(imgs)
#       imgs = np.concatenate([imgs, img_mean], -1)
      texts = np.stack(texts)

        
      res = audios, audio_lens, \
        texts, text_lens, \
        imgs, img_lens, np.array(ys_int).astype('float32')
      return res

    def val_collate_fn(self, batch):
      entry = batch[0]
      audio = entry[0]
      text = entry[1]
      img = entry[2]
      if len(entry) == 4:
        ys_int = np.array([entry[3]]).astype('float32')
      else:
        ys_int = None

      audios, imgs, texts = [], [], []
      img_lens, audio_lens, text_lens = [], [], []
      audio_n = audio.shape[0]
      img_n = img.shape[0]
      text_n = text.shape[0]
        
#       audio_mean = audio.mean(0)
#       img_mean = img.mean(0)

      if self.audio_step > 1:
        audio_off_step = self.audio_step
      else:
        audio_off_step = audio_n//10
      if self.img_step > 1:
        img_off_step = self.img_step
      else:
        img_off_step = img_n//10
      if self.text_step > 1:
        text_off_step = self.text_step
      else:
        text_off_step = text_n//10

      i = 0
      while True:
        audio_offset = i*audio_off_step
        img_offset = i*img_off_step
        text_offset = i*text_off_step

        
        if audio_offset >= audio_n and img_offset >= img_n and text_offset >= text_n:
            break

        if audio_offset < audio_n:
          audio_segment = audio[audio_offset:audio_offset + self.audio_step*self.audio_segment:self.audio_step, :]
        else:
#           start_id = np.random.randint(0, max(1,audio_n - self.audio_step*self.audio_segment + 1))
          start_id = np.random.randint(0, audio_n)
          audio_segment = audio[start_id:start_id+self.audio_step*self.audio_segment:self.audio_step, :]

        if img_offset < img_n:
            img_segment = img[img_offset:img_offset + self.img_step*self.img_segment:self.img_step, :]
        else:
#           start_id = np.random.randint(0, max(img_n - self.img_step*self.img_segment + 1, 1))
          start_id = np.random.randint(0, img_n)
          img_segment = img[start_id:start_id+self.img_step*self.img_segment:self.img_step, :]
        
        if text_offset < text_n:
            text_segment = text[text_offset:text_offset + self.text_step*self.text_segment:self.text_step, :]
        else:
#           start_id = np.random.randint(0, max(text_n - self.text_step*self.text_segment + 1, 1))
          start_id = np.random.randint(0, text_n)
          text_segment = text[start_id:start_id+self.text_step*self.text_segment:self.text_step, :]
        
#         audio_segment = np.concatenate([audio, np.repeat(np.expand_dims(audio_mean, 0), audio.shape[0], 0)], -1)
#         img_segment = np.concatenate([img, np.repeat(np.expand_dims(img_mean, 0), img.shape[0], 0)], -1)
    
        img_lens.append(img_segment.shape[0])
        audio_lens.append(audio_segment.shape[0])
        text_lens.append(text_segment.shape[0])
        
        
        if self.fixed_size:
          max_img_len = self.img_segment
          n = img_lens[-1]
          img_segment = np.pad(img_segment, ((0,max_img_len-n),(0,0)), mode='wrap')

        if self.fixed_size:
          max_audio_len = self.audio_segment
          n = audio_lens[-1]
          audio_segment = np.pad(audio_segment, ((0,max_audio_len-n),(0,0)), mode='wrap')
        
        audios.append(audio_segment)
        imgs.append(img_segment)
        texts.append(text_segment)

        i += 1



      res = audios, audio_lens, texts, text_lens, imgs, img_lens, ys_int
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
#     self.update_data()

  def update_data(self):
      np.random.shuffle(self.files)
      self.texts, self.audios, self.imgs = {}, {}, {}
      files = set([x.split()[0] for x in self.files[:self.size]])
      for fn in tqdm(files):
          self.texts[fn] = np.load(os.path.join(self.path, "%s_long.npz"%fn))['word_embed']
          self.audios[fn] = np.load(os.path.join(self.path, "%s_pase.npy"%fn))
          self.imgs[fn] = np.load(os.path.join(self.path, "%s_img.npz"%fn))['arr_0'].astype('float32')

  def __getitem__(self, idx):
    entry = self.files[idx]
#     fn, labels, start, end = entry.split()
#     start, end = max(0.,float(start)), float(end)
    fn, labels = entry.split()
    
    audio_data = self.audios[fn]
#     audio_n = audio_data.shape[0]
#     audio_start, audio_end = int(audio_n * start), int(audio_n * end)
#     audio_data = audio_data[audio_start:max(audio_start+1, audio_end)]
    
    text_data = self.texts[fn]
#     text_n = text_data.shape[0]
#     text_start, text_end = int(text_n * start), int(text_n * end)
#     text_data = text_data[text_start:max(text_start+1, text_end)]
    
    img_data = self.imgs[fn]
#     img_n = img_data.shape[0]
#     img_start, img_end = int(img_n * start), int(img_n * end)
#     img_data = img_data[img_start:max(img_start+1, img_end)]
    
    
    if self.mode == 'test':
        return audio_data, text_data, img_data

    emo_labels = labels.split('-')[2:]
    y_int = list(map(float, emo_labels))
#     y_int = emo_map[fn_split[-1]]
    
    return audio_data, text_data, img_data, y_int

  def __len__(self):
    return self.size
