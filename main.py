import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from time import time
from datetime import datetime
from model import MyModel
from data import MyDataset, collate_fn, val_collate_fn
import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--name')
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--reg', type=float, default=1e-3)
parser.add_argument('--resume')
parser.add_argument('--attn', type=str, default="")

args = parser.parse_args()


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')

train_path = '/usr/cs/public/data/train'
train_txt = '/usr/cs/public/train_data.txt'
train_dataset = MyDataset(train_path, train_txt, size=15000)
train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True,
        collate_fn = collate_fn, num_workers=0)

val_path = '/usr/cs/public/data/val'
val_txt = '/usr/cs/public/val_data.txt'
val_dataset = MyDataset(val_path, val_txt, mode='val')
val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=True,
        collate_fn = val_collate_fn, num_workers=0)

if args.resume is not None:
    name = args.resume
else:
    cur_datetime = datetime.now().strftime("%m-%d-%H:%M:%S")
    if args.name:
        name = "%s-%s"%(cur_datetime,args.name)
    else:
        name = cur_datetime

model = MyModel(name, lr=args.lr, attn=args.attn, reg=args.reg).to(device)


if args.resume is not None:
  ckpt_path = os.path.join('/usr/cs/grad/doc/melgaar/ckpt', "%s.pt"%args.resume)
  state = torch.load(ckpt_path)
  model.load_state_dict(state['model_state_dict'])
  model.optimizer.load_state_dict(state['optimizer_state_dict'])
  model.lr_decay.load_state_dict(state['scheduler'])
  model.iter = state['iter']
  writer = SummaryWriter(logdir='runs/' + name, purge_step = model.iter)
  print("Loaded model")
else:
  writer = SummaryWriter(logdir='runs/' + name)

def combine_avg(preds):
    return np.expand_dims(np.mean(np.concatenate(preds),0),0)

def combine_voting(preds):
    preds = np.argmax(np.concatenate(preds), 1)
    ind = np.argmax(np.bincount(preds))
    return ind

train_iter = iter(train_dataloader)
mean = lambda l: sum(l)/len(l)
data_counter = 0
while True:
    t0 = time()
    batch = next(train_iter, None)
    if batch is None:
        train_iter = iter(train_dataloader)
        batch = next(train_iter, None)
    batch = [torch.tensor(x).to(device) for x in batch]
    t1 = time()
    res = model.train_step(batch, writer)
    t2 = time()
    if model.iter % 50 == 0:
        state = {'model_state_dict': model.state_dict(),
                'optimizer_state_dict': model.optimizer.state_dict(),
                'iter': model.iter,
                'scheduler': model.lr_decay.state_dict()}
        torch.save(state, os.path.join('/usr/cs/grad/doc/melgaar/ckpt', "%s.pt"%model.name))
    if model.iter % 50 == 0:
        total_loss, audio_acc, img_acc, int_acc = res
        print('[Train] Iter {}: {:.2f}, {:.3f}, {:.3f}, {:.3f}'.format(model.iter, total_loss, audio_acc, img_acc, int_acc))
    if model.iter % 50 == 0:
        val_iter = iter(val_dataloader)
        model.eval()
        val_batch = 128
        results = []
        for i in range(val_batch):
            batch = next(val_iter, None)
            if batch is None:
                val_iter = iter(val_dataloader)
                batch = next(val_iter, None)
            audios, text, text_lens, imgs, ys_audio, ys_img, ys_int = batch
            text, text_lens, ys_int = [torch.tensor(x).to(device) for x in (text, text_lens, ys_int)]
            preds = []
            for audio, img in zip(audios, imgs):
                audio = torch.tensor(audio).to(device)
                img = torch.tensor(img).to(device)
                batch = audio, text, text_lens, img, ys_audio, ys_img, ys_int
                with torch.no_grad():
                    pred = model.val_step(batch)
                preds.append(pred)
            avg = combine_avg(preds)
            pred_class = np.argmax(avg)
            avg = torch.tensor(avg).to(device)
            with torch.no_grad():
                loss = model.loss(avg, ys_int).detach().item()
            correctness = pred_class == ys_int
            if correctness:
                results.append((loss, 1))
            else:
                results.append((loss, 0))
        results = [mean([res[i] for res in results]) for i in range(len(results[0]))]
        # total_loss, audio_acc, img_acc, int_acc = results
        total_loss, int_acc = results
        writer.add_scalar('val/loss/total_loss', total_loss, model.iter)
        # writer.add_scalar('val/acc/audio_acc',audio_acc, model.iter)
        # writer.add_scalar('val/acc/img_acc',img_acc, model.iter)
        writer.add_scalar('val/acc/int_acc',int_acc, model.iter)
        # print('Val Iter {}: {:.2f}, {:.3f}, {:.3f}, {:.3f}'.format(model.iter, total_loss, audio_acc, img_acc, int_acc))
        print('[Val] Iter {}: {:.2f}, {:.3f}'.format(model.iter, total_loss, int_acc))
        model.train()
    if model.iter % 50 == 0:
        print("[Time] Iter {} Data time: {} Model time: {}".format(model.iter, t1-t0,t2-t1,))
    data_counter += 1
    if data_counter % 2000 == 0:
        train_dataset.update_data()
    t0 = time()
