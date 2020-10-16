import torch
from torch.nn.utils import clip_grad_norm_
from torch.nn.utils.rnn import pack_padded_sequence
import torch.nn.init as weight_init
from torch import nn, optim
import torch.nn.functional as F
from scheduler import OneCycleLR

IMG_DIM = 1024
IMG_H_DIM = 64
IMG_LAYERS=2
AUDIO_H_DIM = 64
AUDIO_LAYERS=2
TEXT_H_DIM = 64
CLIP = 5


class MyModel(nn.Module):
  def __init__(self, name, epoch_size, lr=1e-3, attn="", reg = 1e-4, text_h = 64, img_h = 64, audio_h = 64, img_dim = 512,
              audio_layers = 1, img_layers = 1, classify_img_h = 32, classify_audio_h = 32, classify_text_h = 32,
               classify_int_h = 64):
    super().__init__()

    self.name = name
    self.attn = attn
    self.iter = 0
#     self.img_pre = nn.Sequential(nn.Linear(2048,img_dim), nn.Dropout())
    self.img_rnn = nn.LSTM(img_dim,img_h,batch_first=True, num_layers=img_layers)

    self.text_rnn = nn.LSTM(200,text_h,batch_first=True, bidirectional=True)

    self.audio_rnn = nn.LSTM(80,audio_h,batch_first=True, num_layers=audio_layers)

    self.drop = nn.Dropout()

    self.fc1_audio = nn.Sequential(
        nn.Linear(audio_h*audio_layers, classify_audio_h),
        nn.Dropout(),
        nn.ReLU(),
    )
    self.fc2_audio = nn.Sequential(
        nn.Linear(classify_audio_h, 16),
        nn.Dropout(),
        nn.ReLU(),
    )
    
    self.classify_audio = nn.Linear(classify_audio_h,7)
    
    self.fc1_img = nn.Sequential(
        nn.Linear(img_h*img_layers,classify_img_h),
        nn.Dropout(),
        nn.ReLU(),
    )
    self.fc2_img = nn.Sequential(
        nn.Linear(classify_img_h,64),
        nn.Dropout(),
        nn.ReLU(),
    )
    
    self.classify_img = nn.Linear(classify_img_h,7)

    self.fc1_text = nn.Sequential(
        nn.Linear(2*text_h,classify_text_h),
        nn.Dropout(),
        nn.ReLU(),
    )
    
    self.fc2_text = nn.Sequential(
        nn.Linear(classify_text_h,16),
        nn.Dropout(),
        nn.ReLU(),
    )
    
    self.classify_text = nn.Linear(classify_text_h,7)
    
    self.classify_integrated = nn.Sequential(
        nn.Linear(16+16+64, classify_int_h),
        nn.Dropout(),
        nn.ReLU(),
        nn.Linear(classify_int_h,7)
    )

#     if self.attn == 'weighted':
#         self.attn1_audio = nn.Linear(AUDIO_H_DIM*AUDIO_LAYERS + 80, 20)
#         self.attn2_audio = nn.Linear(AUDIO_H_DIM*AUDIO_LAYERS + 80, AUDIO_H_DIM*AUDIO_LAYERS)
#         self.attn1_img = nn.Linear(IMG_H_DIM*IMG_LAYERS + IMG_DIM, 16)
#         self.attn2_img = nn.Linear(IMG_H_DIM*IMG_LAYERS + IMG_DIM, IMG_H_DIM*IMG_LAYERS)

    self.apply(self.init_weights)

    self.loss = nn.CrossEntropyLoss()
    self.optimizer = optim.AdamW(self.parameters(), lr = lr, weight_decay=reg)
#     self.optimizer = optim.SGD(self.parameters(), lr = lr, weight_decay=reg, momentum=0.9, nesterov=True)
    self.lr_decay = optim.lr_scheduler.StepLR(self.optimizer, 50000, 0.1)
#     self.lr_decay = OneCycleLR(self.optimizer, max_lr = 10*lr, total_steps = 80*epoch_size)

  def init_weights(self, m):
    if isinstance(m, nn.Linear) or isinstance(m, torch.nn.Linear):
      weight_init.kaiming_normal_(m.weight, 0)
      # m.weight.data.uniform_(-0.02, 0.02)
      if m.bias is not None:
        m.bias.data.fill_(0)
    elif isinstance(m, nn.LSTM):
      for w in m.parameters():
            if w.dim()>1:
                weight_init.orthogonal_(w)

  def attention_net(self, lstm_output, final_state):
      hidden = final_state.squeeze(0)
      attn_weights = torch.bmm(lstm_output, hidden.unsqueeze(2)).squeeze(2)
      soft_attn_weights = F.softmax(attn_weights, 1)
      new_hidden_state = torch.bmm(lstm_output.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)
  
      return new_hidden_state

  def train_step(self, batch, writer):
    audio, text, text_lens, imgs, y_audio, y_imgs, y_int = batch

    output, (h, c) = self.audio_rnn(audio)
    if self.attn == 'self_h':
        h_audio = self.attention_net(output, h)
    elif self.attn == 'self_c':
        h_audio = self.attention_net(output, c)
    elif self.attn == 'weighted':
        h = torch.cat([x for x in h], -1)
        attn_weights = F.softmax(self.attn1_audio(torch.cat((audio, h), 0)), dim=1)
        pass
    else:
        h_audio = torch.cat([x for x in h], -1)
    h_audio = self.drop(h_audio)
    h_audio = self.fc1_audio(h_audio)
    out_audio = self.classify_audio(h_audio)

    text = pack_padded_sequence(text,text_lens, batch_first=True, enforce_sorted=False)
    _, (h, _) = self.text_rnn(text)
    h_text = torch.cat([x for x in h], -1)
    h_text = self.drop(h_text)
    h_text = self.fc1_text(h_text)
    out_text = self.classify_text(h_text)

#     imgs = self.img_pre(imgs)
    output, (h, c) = self.img_rnn(imgs)
    if self.attn == 'self_h':
        h_imgs = self.attention_net(output, h)
    elif self.attn == 'self_c':
        h_imgs = self.attention_net(output, c)
    elif self.attn == 'weighted':
        pass
    else:
        h_imgs = torch.cat([x for x in h], -1)
    h_imgs = self.drop(h_imgs)
    h_imgs = self.fc1_img(h_imgs)
    out_imgs = self.classify_img(h_imgs)

    h_audio = self.fc2_audio(h_audio)
    h_text = self.fc2_text(h_text)
    h_imgs = self.fc2_img(h_imgs)
    h_all = torch.cat([h_audio, h_text, h_imgs], -1)

    out_int = self.classify_integrated(h_all)

    loss_audio = self.loss(out_audio, y_audio)
    loss_imgs = self.loss(out_imgs, y_imgs)
    loss_text = self.loss(out_text, y_int)
    loss_int = self.loss(out_int, y_int)

    # total_loss = 0.25*loss_audio + 0.25*loss_imgs + 0.5*loss_int
    loss = (loss_audio + loss_imgs + loss_int + loss_text) / 4
#     loss = (loss_audio + loss_imgs + loss_text) / 3
#     loss = loss_imgs

    audio_acc = (torch.argmax(out_audio,1) == y_audio).detach().cpu().numpy().mean()
    img_acc = (torch.argmax(out_imgs,1) == y_imgs).detach().cpu().numpy().mean()
    text_acc = (torch.argmax(out_text,1) == y_int).detach().cpu().numpy().mean()
    int_acc = (torch.argmax(out_int,1) == y_int).detach().cpu().numpy().mean()
    writer.add_scalar('train/loss/audio_loss', loss_audio, self.iter)
    writer.add_scalar('train/loss/img_loss', loss_imgs, self.iter)
    writer.add_scalar('train/loss/text_loss', loss_text, self.iter)
    writer.add_scalar('train/loss/int_loss', loss_int, self.iter)
    writer.add_scalar('train/acc/audio_acc',audio_acc, self.iter)
    writer.add_scalar('train/acc/img_acc',img_acc, self.iter)
    writer.add_scalar('train/acc/text_acc',text_acc, self.iter)
    writer.add_scalar('train/acc/int_acc',int_acc, self.iter)

    self.optimizer.zero_grad()
    loss.backward()
    clip_grad_norm_(self.parameters(), CLIP)
    self.optimizer.step()
    self.lr_decay.step()
    self.iter += 1

#     return total_loss.detach().cpu().item(), audio_acc, img_acc, int_acc

  def val_step(self, batch):
    audio, text, text_lens, imgs, y_audio, y_imgs, y_int = batch

    _, (h, _) = self.audio_rnn(audio)

    if self.attn == 'self_h':
        h_audio = self.attention_net(output, h)
    elif self.attn == 'self_c':
        h_audio = self.attention_net(output, c)
    elif self.attn == 'weighted':
        pass
    else:
        h_audio = torch.cat([x for x in h], -1)
     
    h_audio = self.fc1_audio(h_audio)
    out_audio = self.classify_audio(h_audio)

    text = pack_padded_sequence(text,text_lens, batch_first=True, enforce_sorted=False)
    _, (h, _) = self.text_rnn(text)
    h_text = torch.cat([x for x in h], -1)
    h_text = self.fc1_text(h_text)
    out_text = self.classify_text(h_text)

#     imgs = self.img_pre(imgs)
    _, (h, c) = self.img_rnn(imgs)
    if self.attn == 'self_h':
        h_imgs = self.attention_net(output, h)
    elif self.attn == 'self_c':
        h_imgs = self.attention_net(output, c)
    elif self.attn == 'weighted':
        pass
    else:
        h_imgs = torch.cat([x for x in h], -1)
    
    h_imgs = self.fc1_img(h_imgs) 
    out_imgs = self.classify_img(h_imgs)

    h_audio = self.fc2_audio(h_audio)
    h_text = self.fc2_text(h_text)
    h_imgs = self.fc2_img(h_imgs)
    h_all = torch.cat([h_audio, h_text, h_imgs], -1)

    out_int = self.classify_integrated(h_all)

    # loss_audio = self.loss(out_audio, y_audio)
    # loss_imgs = self.loss(out_imgs, y_imgs)
    # loss_int = self.loss(out_int, y_int)

    # total_loss = 0.25*loss_audio + 0.25*loss_imgs + 0.5*loss_int
    # total_loss = (loss_audio + loss_imgs + loss_int) / 3

    # audio_acc = (torch.argmax(out_audio,1) == y_audio).detach().cpu().numpy().mean()
    # img_acc = (torch.argmax(out_imgs,1) == y_imgs).detach().cpu().numpy().mean()
    # int_acc = (torch.argmax(out_int,1) == y_int).detach().cpu().numpy().mean()
    return out_audio.detach().cpu().numpy(), out_imgs.detach().cpu().numpy(), \
           out_text.detach().cpu().numpy(), out_int.detach().cpu().numpy()
