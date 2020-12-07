import torch
from torch.nn.utils import clip_grad_norm_
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.init as weight_init
from torch import nn, optim
import torch.nn.functional as F
from torch_future import OneCycleLR, Nadam, SGDW
from torch.autograd import Variable
import math

CLIP = 5

class GELU1(nn.Module):
    """
    Paper Section 3.4, last paragraph notice that BERT used the GELU instead of RELU
    """

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    
class GELU2(nn.Module):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    
    def forward(self, x):
        return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))
    
class RnnLinear(nn.Module):
    def __init__(self, in_dim, rnn_dim, activation, bidirectional=False, num_layers = 1, lin_drop=0.2, in_drop=0):
        super().__init__()
        self.rnn = nn.LSTM(in_dim, rnn_dim, bidirectional=bidirectional,
                           num_layers = num_layers, batch_first=True)
        self.num_layers = num_layers
        self.rnn_dim = rnn_dim
        self.bidirectional = bidirectional
        self.in_drop = in_drop
#         self.bn = nn.BatchNorm1d(in_dim)
#         self.lin = nn.Sequential(
#             nn.BatchNorm1d(2*rnn_dim),
#             nn.Linear(2*rnn_dim, 2*rnn_dim),
#             nn.Dropout(lin_drop),
#             activation(),
#         )
        
    def forward(self, x, lens = None):
        bs = x.shape[0]
        x = F.layer_norm(x, x.shape[1:])
#         x = self.bn(x.transpose(1,2)).transpose(1,2)
        x = F.dropout(x, self.in_drop)
        if lens is not None:
            x = pack_padded_sequence(x, lens, batch_first=True, enforce_sorted=False)
        output, (h, c) = self.rnn(x)
        h = h.view(self.num_layers, (1+self.bidirectional), bs, self.rnn_dim)[-1]
        h = torch.cat([x for x in h], -1)
#         h = self.lin(h)
        return h

class MyModel(nn.Module):
  def __init__(self, name, hparams):
    super().__init__()

    self.name = name
    self.hparams = hparams
    self.iter = 0
    self.reset_state()
    num_classes = 6
    
    if hparams.activation == 'gelu1':
        activation = GELU1
    elif hparams.activation == 'gelu2':
        activation = GELU2
    elif hparams.activation == 'relu':
        activation = nn.ReLU
    elif hparams.activation == 'leakyrelu':
        activation = nn.LeakyReLU
    
#     self.img_cnn = nn.Sequential(
#         nn.Conv2d(1, 8, 5),
#         nn.ReLU(),
#         nn.MaxPool2d(2),
#         nn.BatchNorm2d(8),
#         nn.Conv2d(8, 16, 5),
#         nn.ReLU(),
#         nn.MaxPool2d(2),
#         nn.BatchNorm2d(16),
#         nn.Conv2d(16, 32, 5),
#         nn.ReLU(),
#         nn.MaxPool2d(2),
#     )
    self.dropout = nn.Dropout(hparams.post_rnn_dropout)
#     self.rnn_img = nn.LSTM(hparams.img_dim,
#                            self.hparams.img_h,
#                            num_layers = self.hparams.img_layers,
#                            batch_first = True, bidirectional = True, dropout=hparams.img_rnn_dropout)
    self.rnn_text = RnnLinear(hparams.text_dim, hparams.text_h, activation, num_layers = hparams.text_layers,
                              bidirectional=hparams.text_bidirectional)
    
    self.rnn_img = RnnLinear(hparams.img_dim, hparams.img_h, activation, num_layers = hparams.img_layers,
                             bidirectional=hparams.img_bidirectional, in_drop = self.hparams.in_drop_img)
    self.rnn_audio = RnnLinear(hparams.audio_dim, hparams.audio_h, activation, num_layers = hparams.audio_layers,
                               bidirectional=hparams.audio_bidirectional)
#     self.rnn_img = NewRNN(hparams.img_dim, hparams.img_h)
#     self.rnn_audio = NewRNN(hparams.audio_dim, hparams.audio_h) 
#     self.rnn_text = nn.LSTM(200, self.hparams.text_h, batch_first=True, bidirectional=True,
#                             num_layers = self.hparams.text_layers, dropout=hparams.text_rnn_dropout)

#     self.rnn_audio = nn.LSTM(80, self.hparams.audio_h, batch_first = True, num_layers = self.hparams.audio_layers,
#                             bidirectional=True)
    
    self.av = nn.Sequential(
        nn.BatchNorm1d((1+hparams.audio_bidirectional)*hparams.audio_h + (1+hparams.img_bidirectional)*hparams.img_h),
        nn.Linear((1+hparams.audio_bidirectional)*hparams.audio_h + (1+hparams.img_bidirectional)*hparams.img_h, hparams.fc1_av),
        nn.Dropout(hparams.fc_dropout),
        activation(),
#         nn.BatchNorm1d(hparams.fc1_dim),
#         nn.Linear(hparams.fc1_dim, hparams.fc1_dim),
#         nn.Dropout(hparams.fc_dropout),
#         activation()
    )
    self.at = nn.Sequential(
        nn.BatchNorm1d((1+hparams.audio_bidirectional)*hparams.audio_h + (1+hparams.text_bidirectional)*hparams.text_h),
        nn.Linear((1+hparams.audio_bidirectional)*hparams.audio_h + (1+hparams.text_bidirectional)*hparams.text_h, hparams.fc1_at),
        nn.Dropout(hparams.fc_dropout),
        activation(),
#         nn.BatchNorm1d(hparams.fc1_dim),
#         nn.Linear(hparams.fc1_dim, hparams.fc1_dim),
#         nn.Dropout(hparams.fc_dropout),
#         activation()
    )
    self.vt = nn.Sequential(
        nn.BatchNorm1d((1+hparams.img_bidirectional)*hparams.img_h + (1+hparams.text_bidirectional)*hparams.text_h),
        nn.Linear((1+hparams.img_bidirectional)*hparams.img_h + (1+hparams.text_bidirectional)*hparams.text_h, hparams.fc1_vt),
        nn.Dropout(hparams.fc_dropout),
        activation(),
#         nn.BatchNorm1d(hparams.fc1_dim),
#         nn.Linear(hparams.fc1_dim, hparams.fc1_dim),
#         nn.Dropout(hparams.fc_dropout),
#         activation()
    )
    self.avt = nn.Sequential(
        nn.BatchNorm1d(hparams.fc1_av + hparams.fc1_at + hparams.fc1_vt),
        nn.Linear(hparams.fc1_av + hparams.fc1_at + hparams.fc1_vt, hparams.fc2_dim),
        nn.Dropout(hparams.fc_dropout),
        activation(),
#         nn.BatchNorm1d(hparams.fc2_dim),
#         nn.Linear(hparams.fc2_dim, hparams.fc2_dim),
#         nn.Dropout(hparams.fc_dropout),
#         activation()
    )
    
    self.classify_integrated = nn.Linear(hparams.fc2_dim, num_classes)
    
#     self.audio_spk = nn.Linear(2*hparams.audio_h, hparams.spk_dim)
#     self.img_spk = nn.Linear(2*hparams.img_h, hparams.spk_dim)

#     self.av_spk = nn.Linear((1+hparams.audio_bidirectional)*hparams.audio_h + (1+hparams.img_bidirectional)*hparams.img_h, hparams.spk_dim)
#     self.av_spk = nn.Linear(hparams.audio_h + hparams.img_h, hparams.spk_dim)
    
#     self.at_spk = nn.Linear(hparams.fc1_dim, hparams.spk_dim)
#     self.vt_spk = nn.Linear(hparams.fc1_dim, hparams.spk_dim)
#     self.avt_spk = nn.Linear(hparams.fc2_dim, hparams.spk_dim)
    
#     self.classify_spk = nn.Linear(hparams.spk_dim, hparams.num_spks)
#     self.classify_gen = nn.Linear(hparams.spk_dim, 2)
#     self.classify_age = nn.Linear(hparams.spk_dim, 1)
        

    self.apply(self.init_weights)

    w = torch.tensor([0.7, 2., 4., 3., 2.5, 3.])
#     w = torch.tensor([0.270, 0.930, 2.095, 2.130, 1.618, 2.480])
#     w = torch.tensor([1, 1., 2., 2., 1., 2.])
#     self.loss = nn.CrossEntropyLoss(reduction='none')
    self.loss = nn.BCEWithLogitsLoss(reduction='none', pos_weight = w)
#     self.loss = nn.BCEWithLogitsLoss(reduction='none')
    
    if hparams.setting == 'aux':
        self.classify_audio = nn.Linear((1+hparams.audio_bidirectional)*hparams.audio_h, num_classes)
        self.classify_img = nn.Linear((1+hparams.img_bidirectional)*hparams.img_h, num_classes)
        self.classify_text = nn.Linear((1+hparams.text_bidirectional)*hparams.text_h, num_classes)
        self.classify_av = nn.Linear(hparams.fc1_av, num_classes)
        self.classify_at = nn.Linear(hparams.fc1_at, num_classes)
        self.classify_vt = nn.Linear(hparams.fc1_vt, num_classes)

        
  def init_weights(self, m):
    if isinstance(m, nn.Linear) or isinstance(m, torch.nn.Linear):
      weight_init.kaiming_normal_(m.weight, 0)
      if m.bias is not None:
        m.bias.data.fill_(0)
    elif isinstance(m, nn.LSTM):
      for w in m.parameters():
            if w.dim()>1:
                weight_init.orthogonal_(w)
                
  def reset_state(self):
    self.stage = self.hparams.init_stage
    self.earlystop_audio = False
    self.earlystop_img = False
    self.earlystop_text = False
    self.earlystop_av = False
    self.earlystop_at = False
    self.earlystop_vt = False
    self.earlystop_spk = False
    
  def attention_net(self, lstm_output, final_state):
      hidden = final_state
      attn_weights = torch.bmm(lstm_output, hidden.unsqueeze(2)).squeeze(2)
      soft_attn_weights = F.softmax(attn_weights, 1)
      new_hidden_state = torch.bmm(lstm_output.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)
      return new_hidden_state

  def forward(self, batch):
    audio, audio_lens, text, text_lens, imgs, img_lens = batch
    bs = audio.shape[0]
#     text *= 0
#     audio *= 0
#     imgs *= 0
    
#     audio = self.ln_audio(audio)
#     output, (h_audio, c) = self.rnn_audio(audio)
#     h_audio = h_audio.view(self.hparams.audio_layers, 2, bs, self.hparams.audio_h)[-1]
#     h_audio = torch.cat([x for x in h_audio], -1)

#     audio = audio.transpose(1,2)
    
    h_audio = self.rnn_audio(audio, audio_lens)
    h_audio = self.dropout(h_audio)

#     text = self.ln_text(text)
#     text = pack_padded_sequence(text,text_lens, batch_first=True, enforce_sorted=False)
#     output, (h_text, c) = self.rnn_text(text)
#     h_text = h_text.view(self.hparams.text_layers, 2, bs, self.hparams.text_h)[-1]
#     h_text = torch.cat([x for x in h_text], -1)
#     h_text = self.rnn_text(text, text_lens)
    h_text = self.rnn_text(text, text_lens)
    h_text = self.dropout(h_text)
    
#     imgs = self.ln_img(imgs)
#     imgs = pack_padded_sequence(imgs, img_lens, batch_first=True, enforce_sorted=False)
#     output, (h_imgs, c) = self.rnn_img(imgs)
#     h_imgs = h_imgs.view(self.hparams.img_layers, 2, bs, self.hparams.img_h)[-1]
#     h_imgs = torch.cat([x for x in h_imgs], -1)

#     imgs = imgs.transpose(1,2)
    
#     h_imgs = self.img_cnn(imgs.unsqueeze(1))
#     h_imgs = h_imgs.squeeze().transpose(1,2)
    h_imgs = self.rnn_img(imgs, img_lens)
#     if self.training:
#         h_imgs += torch.normal(torch.zeros_like(h_imgs), torch.full_like(h_imgs, 0.1))
    h_imgs = self.dropout(h_imgs)

#     h_imgs = h_imgs.squeeze()
#     h_audio = h_audio.squeeze()
    
    h_av = torch.cat([h_audio, h_imgs], -1)
    av = self.av(h_av)
#     spk = self.av_spk(h_av)
#     outs_spk = self.classify_spk(spk)
#     outs_gen = self.classify_gen(spk)
#     outs_age = self.classify_age(spk)
    
    if self.hparams.setting == 'aux' and self.stage == 1:
        out_audio = self.classify_audio(h_audio)
        out_imgs = self.classify_img(h_imgs)
        out_text = self.classify_text(h_text)
        
        return out_audio, out_imgs, out_text

    h_at = torch.cat([h_audio, h_text], -1)
    h_vt = torch.cat([h_imgs, h_text], -1)
    
    at = self.at(h_at)
    vt = self.vt(h_vt)
    
    if self.hparams.setting == 'aux' and self.stage == 2:
        out_av = self.classify_av(av)
        out_at = self.classify_at(at)
        out_vt = self.classify_vt(vt)
        
        return out_av, out_at, out_vt
    
    
    h_avt = torch.cat([av, at, vt], -1)
    avt = self.avt(h_avt)

    out_int = self.classify_integrated(avt)
    return out_int
    
class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class MyModel2(nn.Module):
    def __init__(self, heads = 8, layers = 6, hid = 200, drop = 0.2):
        super().__init__()
        self.iter = 0
        self.pe = PositionalEncoding(200, max_len=100)
        self.cls = Variable(torch.zeros(1, 200), requires_grad=True).cuda()
        encoder_layer = nn.TransformerEncoderLayer(d_model=200, nhead=heads, dim_feedforward = hid, dropout=drop)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
#         self.rnn = nn.LSTM(200,100, bidirectional=True)
        self.classify = nn.Linear(200, num_classes)
        
        
    def forward(self, x, lens):
        bs, seqlen, hdim = x.shape
        cls = self.cls.view(1,1,-1).repeat(bs, 1, 1)
        x = torch.cat([cls, x], 1)
        x = self.pe(x)
        x = x.transpose(0,1)
        res = self.transformer_encoder(x)
        res = res[0]
#         res = pack_padded_sequence(res, lens, enforce_sorted=False)
#         _, (h, _) = self.rnn(res)
#         res = torch.cat([x for x in h], -1)
        res = self.classify(res)
        return res
    
class MyModel3(nn.Module):
    def __init__(self):
        super().__init__()
        self.iter = 0
        self.rnn = nn.LSTM(200, 128, bidirectional=True, batch_first = True)
        self.classify = nn.Linear(2*128, num_classes)
        
    def forward(self, x, lens):
        bs = x.shape[0]
        x = pack_padded_sequence(x, lens, batch_first=True, enforce_sorted=False)
        out, (h,c) = self.rnn(x)
        h = torch.cat([x for x in h], -1)
        out = self.classify(h)
        return out
