from glob import glob
from time import time
from math import ceil
from tqdm import tqdm
import os
import numpy as np
from torch.utils.data import Dataset

emotions = ['neu', 'hap', 'ang', 'fea', 'dis', 'sur', 'sad']
emo_map = {k:i for i,k in enumerate(emotions)}
AUDIO_SEGMENT = 20
IMG_SEGMENT = 16

def collate_fn(batch):
  audios, texts, imgs, ys_audio, ys_img, ys_int = [list(x) for x in zip(*batch)]

  text_lens = [text.shape[0] for text in texts]
  max_text_len = max(text_lens)
  for i,text in enumerate(texts):
    n = text_lens[i]
    texts[i] = np.pad(text, ((0,max_text_len-n),(0,0)), mode='constant')

  res = np.stack(audios).transpose(0,2,1), np.stack(texts), text_lens, np.stack(imgs), np.array(ys_audio), np.array(ys_img), np.array(ys_int)
  return res

def val_collate_fn(batch):
  entry = batch[0]
  audio = np.expand_dims(entry[0],0).transpose(0,2,1)
  text = np.expand_dims(entry[1],0)
  img = np.expand_dims(entry[2],0)
  ys_audio = np.array([entry[3]])
  ys_img = np.array([entry[4]])
  ys_int = np.array([entry[5]])

  audios, imgs = [], []
  audio_n = audio.shape[1]
  img_n = img.shape[1]
  audio_segments = ceil(audio_n/AUDIO_SEGMENT)
  img_segments = ceil(img_n/IMG_SEGMENT)

  for i in range(max(audio_segments, img_segments)):
    audio_offset = i*AUDIO_SEGMENT
    img_offset = i*IMG_SEGMENT
    if audio_offset < audio_n:
      audios.append(audio[:, audio_offset:audio_offset + AUDIO_SEGMENT, :])
    else:
      start_id = np.random.randint(0, audio_n - AUDIO_SEGMENT + 1)
      audios.append(audio[:, start_id:start_id+AUDIO_SEGMENT, :])
    if img_offset < img_n:
        imgs.append(img[:, img_offset:img_offset + IMG_SEGMENT, :])
    else:
      start_id = np.random.randint(0, img_n - IMG_SEGMENT + 1)
      imgs.append(img[:, start_id:start_id+IMG_SEGMENT, :])


  text_lens = [text.shape[0]]

  res = audios, text, text_lens, imgs, ys_audio, ys_img, ys_int
  return res


class MyDataset(Dataset):
  def __init__(self, data_path, data_txt, mode='train', size=10000):
    super().__init__()
    files = [x.rstrip() for x in open(data_txt)][:2000]
    self.files = files
    self.path = data_path
    self.mode = mode
    self.size = min(len(files),size)
    self.update_data()

  def update_data(self):
      np.random.shuffle(self.files)
      t0 = time()
      texts, audios, imgs = [], [], []
      for fn in tqdm(self.files[:self.size]):
          texts.append(np.load(os.path.join(self.path, "%s.npz"%fn[:5]))['word_embed'])
          audios.append(np.load(os.path.join(self.path, "%s_mel.npy"%fn)).astype('float32'))
          imgs.append(np.load(os.path.join(self.path, "%s_img_new.npz"%fn))['arr_0'])
      self.texts = texts
      self.audios = audios
      self.imgs = imgs
#       self.texts = [np.load(os.path.join(self.path, "%s.npz"%fn[:5]))['word_embed']
#               for fn in self.files[:self.size]]
#       self.audios = [np.load(os.path.join(self.path, "%s_mel.npy"%fn)).astype('float32')
#               for fn in self.files[:self.size]]
#       self.imgs = [np.load(os.path.join(self.path, "%s_img.npz"%fn))['arr_0']
#               for fn in self.files[:self.size]]

  def __getitem__(self, idx):
    audio_data = self.audios[idx]
    text_data = self.texts[idx]
    img_data = self.imgs[idx]
    if self.mode == 'train':
        start_id = np.random.randint(0,audio_data.shape[1] - AUDIO_SEGMENT + 1)
        audio_data = audio_data[:,start_id:start_id+AUDIO_SEGMENT]

        start_id = np.random.randint(0,img_data.shape[0] - IMG_SEGMENT + 1)
        img_data = img_data[start_id:start_id + IMG_SEGMENT, :]

    fn = self.files[idx]
    fn_split = fn.split('-')
    y_int = emo_map[fn_split[-3]]
    y_img = emo_map[fn_split[-2]]
    y_audio = emo_map[fn_split[-1]]

    return audio_data, text_data, img_data, y_audio, y_img, y_int

  def __getitem__old(self, idx):
    fn = self.files[idx]
    audio_path = os.path.join(self.path, "%s_mel.npy"%fn)
    audio_data = np.load(audio_path).astype('float32')
    if self.mode == 'train':
        start_id = np.random.randint(0,audio_data.shape[1] - AUDIO_SEGMENT + 1)
        audio_data = audio_data[:,start_id:start_id+AUDIO_SEGMENT]

    text_path = os.path.join(self.path, "%s.npz"%fn[:5])
    text_data = np.load(text_path)['word_embed']

    img_path = os.path.join(self.path, "%s_img.npz"%fn)
    imgs = np.load(img_path)['arr_0']
    num_frames = imgs.shape[0]
    if self.mode == 'train':
        start_id = np.random.randint(0,num_frames - IMG_SEGMENT + 1)
        img_data = imgs[start_id:start_id + IMG_SEGMENT, :]
    else:
        img_data = imgs

    fn_split = fn.split('-')
    y_int = emo_map[fn_split[-3]]
    y_img = emo_map[fn_split[-2]]
    y_audio = emo_map[fn_split[-1]]

    return audio_data, text_data, img_data, y_audio, y_img, y_int

  def __len__(self):
    return self.size
