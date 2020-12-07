import pandas as pd
import numpy as np
import sys

emos = ['hap', 'sad', 'fea', 'sur', 'ang', 'dis', 'neu']

df1 = pd.read_csv('MOSEI/Labels/5000_batch_raw.csv')
df2 = pd.read_csv('MOSEI/Labels/pom_extra_sqa_mono_results_mod.csv')
df3 = pd.read_csv('MOSEI/Labels/Batch_2980374_batch_results.csv')

folder = sys.argv[1]
data = [x.strip() for x in open('%s_data.txt'%folder)]
df1_ids = set(df1['Input.VIDEO_ID'])
df2_ids = set(df2['Input.VIDEO_ID'])
df3_ids = set(df3['Input.VIDEO_ID'])

debug = True
if not debug:
    out_f = open('%s_data_labeled.txt'%folder, 'w')
cum = np.array([0] * 6)
for fn in data:
    if fn in df1_ids:
        entry = df1.loc[df1['Input.VIDEO_ID'] == fn]
    elif fn in df2_ids:
        entry = df2.loc[df2['Input.VIDEO_ID'] == fn]
    elif fn in df3_ids:
        entry = df3.loc[df3['Input.VIDEO_ID'] == fn]

    # print('Video_ID:',fn)
    # print('Number of ratings:', len(entry))
    if entry.ndim == 2:
        entry = entry.mean(0)
    emo_scores = [entry['Answer.happiness'], entry['Answer.sadness'], entry['Answer.fear'], entry['Answer.surprise'], entry['Answer.anger'], entry['Answer.disgust']]
    # print('Scores:',["%.2f"%x for x in emo_scores])
    emo_scores = [1 if x > 0 else 0 for x in emo_scores]
    cum += emo_scores
    top_emo = np.argmax(emo_scores)
    if emo_scores[top_emo] == 0:
        top_emo = 6
    sent = entry['Answer.sentiment'] 
    sent_2 = 'pos' if sent >= 0 else 'neg'
    sent = abs(sent)
    if sent > 2:
        sent_7 = 3
    elif sent <= 2 and sent > 1:
        sent_7 = 2
    elif sent <= 1 and sent > 0:
        sent_7 = 1
    else:
        sent_7 = 0
    new_fn = "{} {}-{}-{}\n".format(fn, sent_7, sent_2, "-".join(map(str,emo_scores)))
    if not debug:
        out_f.write(new_fn)
print("Pos:", cum)
print("Weights:", [(2801-x)/x for x in cum])
if not debug:
    out_f.close()
