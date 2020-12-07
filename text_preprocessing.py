import os, sys
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

device = torch.device('cuda')
tokenizer = AutoTokenizer.from_pretrained('distilroberta-base')
model = AutoModel.from_pretrained('distilroberta-base').eval().to(device)
max_len = 512

def process(fn, root_dir, out_dir):
    in_file1 = os.path.join(root_dir, "%s.transcript"%fn)
    in_file2 = os.path.join(root_dir, "%s.plaintext"%fn)
    in_file3 = os.path.join(root_dir, "%s.en.vtt"%fn)
    in_file4 = os.path.join(root_dir, "%s-user.en.vtt"%fn)
    in_file5 = os.path.join(root_dir, "%s.txt"%fn)
    if os.path.isfile(in_file1):
        with open(in_file1) as f:
            text = f.read()
    elif os.path.isfile(in_file5):
        with open(in_file5) as f:
            text = f.read()
    elif os.path.isfile(in_file2):
        with open(in_file2) as f:
            text = f.read()
    elif os.path.isfile(in_file3):
        text = []
        with open(in_file3) as f:
            on = False
            for line in f:
                if on:
                    text.append(line)
                    on = False
                elif line[0].isnumeric():
                    on = True
        text = "\n".join(text)
    elif os.path.isfile(in_file4):
        text = []
        with open(in_file4) as f:
            on = False
            for line in f:
                if on:
                    text.append(line)
                    on = False
                elif line[0].isnumeric():
                    on = True
        text = "\n".join(text)
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt").to(device)
        outs = []
        length = inputs['input_ids'].shape[1]
        for i in range(0,length,max_len):
            segment = {'input_ids': inputs['input_ids'][:,i:i+max_len],
                    'attention_mask': inputs['attention_mask'][:,i:i+max_len]
                }
            outputs = model(**segment)[0].squeeze(0).cpu().numpy()
            outs.append(outputs)
        outputs = np.concatenate(outs, 0)
    out_path = os.path.join(out_dir, "%s_long.npz"%fn)
    np.savez(out_path, word_embed=outputs)

def main(folder):
    files = [x.rstrip() for x in open('/usr/cs/public/mohd/%s_data.txt'%folder)]
    files.sort()
    root_dir = '/usr/cs/public/mohd/MOSEI/Transcript/Full/Combined'
    out_dir = '/usr/cs/public/mohd/data/%s'%folder
    FNULL = open(os.devnull, 'w')
    for fn in tqdm(files,total=len(files), desc=folder):
        process(fn, root_dir, out_dir)
    # with ThreadPoolExecutor(5) as threads:
    #     for _ in tqdm(threads.map(process, files), total = len(files)):
    #         pass

if __name__ == '__main__':
    folder = sys.argv[1]
    if folder == 'all':
        for fold in ['train', 'val', 'test1']:
            main(fold)
    else:
        main(folder)
