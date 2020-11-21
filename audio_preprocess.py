from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import subprocess
import librosa
import librosa.filters
import math
import os, sys
import numpy as np
import scipy
from rVAD_fast import get_vad

class Hparams():
  cleaners='english_cleaners'
  # Audio:
  num_mels=80
  num_freq=1025
  sample_rate=16000
  frame_length_ms=25
  frame_shift_ms=10
  preemphasis=0.97
  min_level_db=-100
  ref_level_db=20

  # Model:
  outputs_per_step=5
  embed_depth=256
  prenet_depths=[256, 128]
  encoder_depth=256
  postnet_depth=256
  attention_depth=256
  decoder_depth=256

  # Training:
  batch_size=32
  adam_beta1=0.9
  adam_beta2=0.999
  initial_learning_rate=0.002
  decay_learning_rate=True
  use_cmudict=False  # Use CMUDict during training to learn pronunciation of ARPAbet phonemes

  # Eval:
  max_iters=200
  griffin_lim_iters=60
  power=1.5              # Power to raise magnitudes to prior to Griffin-Lim

hparams = Hparams()

def load_wav(path):
  return librosa.core.load(path, sr=hparams.sample_rate)[0]


def save_wav(wav, path):
  wav *= 32767 / max(0.01, np.max(np.abs(wav)))
  scipy.io.wavfile.write(path, hparams.sample_rate, wav.astype(np.int16))


def preemphasis(x):
  return scipy.signal.lfilter([1, -hparams.preemphasis], [1], x)


def inv_preemphasis(x):
  return scipy.signal.lfilter([1], [1, -hparams.preemphasis], x)


def spectrogram(y):
  D = _stft(preemphasis(y))
  S = _amp_to_db(np.abs(D)) - hparams.ref_level_db
  return _normalize(S)


def inv_spectrogram(spectrogram):
  '''Converts spectrogram to waveform using librosa'''
  S = _db_to_amp(_denormalize(spectrogram) + hparams.ref_level_db)  # Convert back to linear
  return inv_preemphasis(_griffin_lim(S ** hparams.power))          # Reconstruct phase


def melspectrogram(y):
  D = _stft(preemphasis(y))
  S = _amp_to_db(_linear_to_mel(np.abs(D))) - hparams.ref_level_db
  return _normalize(S)


def find_endpoint(wav, threshold_db=-40, min_silence_sec=0.8):
  window_length = int(hparams.sample_rate * min_silence_sec)
  hop_length = int(window_length / 4)
  threshold = _db_to_amp(threshold_db)
  for x in range(hop_length, len(wav) - window_length, hop_length):
    if np.max(wav[x:x+window_length]) < threshold:
      return x + hop_length
  return len(wav)


def _griffin_lim(S):
  '''librosa implementation of Griffin-Lim
  Based on https://github.com/librosa/librosa/issues/434
  '''
  angles = np.exp(2j * np.pi * np.random.rand(*S.shape))
  S_complex = np.abs(S).astype(np.complex)
  y = _istft(S_complex * angles)
  for i in range(hparams.griffin_lim_iters):
    angles = np.exp(1j * np.angle(_stft(y)))
    y = _istft(S_complex * angles)
  return y


def _stft(y):
  n_fft, hop_length, win_length = _stft_parameters()
  return librosa.stft(y=y, n_fft=n_fft, hop_length=hop_length, win_length=win_length)


def _istft(y):
  _, hop_length, win_length = _stft_parameters()
  return librosa.istft(y, hop_length=hop_length, win_length=win_length)


def _stft_parameters():
  n_fft = (hparams.num_freq - 1) * 2
  hop_length = int(hparams.frame_shift_ms / 1000 * hparams.sample_rate)
  win_length = int(hparams.frame_length_ms / 1000 * hparams.sample_rate)
  return n_fft, hop_length, win_length


# Conversions:

_mel_basis = None

def _linear_to_mel(spectrogram):
  global _mel_basis
  if _mel_basis is None:
    _mel_basis = _build_mel_basis()
  return np.dot(_mel_basis, spectrogram)

def _build_mel_basis():
  n_fft = (hparams.num_freq - 1) * 2
  return librosa.filters.mel(hparams.sample_rate, n_fft, n_mels=hparams.num_mels)

def _amp_to_db(x):
  return 20 * np.log10(np.maximum(1e-5, x))

def _db_to_amp(x):
  return np.power(10.0, x * 0.05)

def _normalize(S):
  return np.clip((S - hparams.min_level_db) / -hparams.min_level_db, 0, 1)

def _denormalize(S):
  return (np.clip(S, 0, 1) * -hparams.min_level_db) + hparams.min_level_db


def process2(fn):
    src = os.path.join(data_dir,"%s.mp4"%fn)
    dst = os.path.join(tmp_dir,"%s.wav"%fn)
    out_path = os.path.join(out_dir,'%s_mel.npy'%fn)
    vad = os.path.join(out_dir,'%s_vad.txt'%fn)
    out_wav = os.path.join(out_dir,'%s.wav'%fn)
    if os.path.isfile(out_wav):
        return
    command = "ffmpeg -y -i %s -ac 2 -f wav %s"%(src, dst)
    r = subprocess.call(command, shell=True, stdout=FNULL, stderr=subprocess.STDOUT)

    wav = load_wav(dst)
    # with open(vad, 'r') as f:
    #     start, end = map(int, f.read().strip().split(', '))

    # mel = melspectrogram(wav)
    vad = get_vad(dst).tolist()
    if 1 in vad:
        start = vad.index(1)
        end = len(vad) - 1 - vad[::-1].index(1)
    else:
        start = 0
        end = -1

    # np.save(out_path,mel[:,start:end])
    # os.remove(dst)

    if start == 0 and end == -1:
        pass
    else:
        start = 25 + 10*start
        end = 25 + 10*end
    start = int(16000 * start/1000)
    end = int(16000 * end/1000)
    save_wav(wav[start:end], out_wav)
    # with open(out_path.replace('mel','vad'), 'w') as f:
    #     f.write("%d, %d"%(start, end))

def process(fn):
    src = os.path.join(data_dir,"%s.mp4"%fn)
    dst = os.path.join(tmp_dir,"%s.wav"%fn)
    out_path = os.path.join(out_dir,'%s_mel.npy'%fn)
    # if os.path.isfile(out_path):
    #     return
    command = "ffmpeg -y -i %s -ac 2 -f wav %s"%(src, dst)
    r = subprocess.call(command, shell=True, stdout=FNULL, stderr=subprocess.STDOUT)

    wav = load_wav(dst)

    mel = melspectrogram(wav)
    vad = get_vad(dst).tolist()
    if 1 in vad:
        start = vad.index(1)
        end = len(vad) - 1 - vad[::-1].index(1)
    else:
        start = 0
        end = -1

    np.save(out_path,mel[:,start:end])
    os.remove(dst)

    if start == 0 and end == -1:
        pass
    else:
        start = 25 + 10*start
        end = 25 + 10*end
    with open(out_path.replace('mel','vad'), 'w') as f:
        f.write("%d, %d"%(start, end))

if __name__ == '__main__':
    folder = sys.argv[1]
    files = [x.rstrip() for x in open('/usr/cs/public/mohd/%s_data.txt'%folder)]
    files.sort()
    data_dir = '/usr/cs/public/mohd/2020-1/%s'%folder
    tmp_dir = '/usr/cs/public/mohd/data/tmp'
    out_dir = '/usr/cs/public/mohd/data/%s'%folder
    FNULL = open(os.devnull, 'w')
    for fn in tqdm(files,total=len(files)):
        process2(fn)
    # with ThreadPoolExecutor(5) as threads:
    #     for _ in tqdm(threads.map(process, files), total = len(files)):
    #         pass
    # process2(files[0])
