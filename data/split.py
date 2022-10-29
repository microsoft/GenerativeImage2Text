import pandas as pd
import random

random.seed(42)

samples = pd.read_csv('samples.csv')
clips = samples['clip_name'].unique()

split = (2, 2)

test_val_clips = random.sample(clips.tolist(), k=sum(split))
test_clips = test_val_clips[:split[0]]
val_clips = test_val_clips[split[0]:]

val_df = samples[samples['clip_name'].isin(val_clips)]
test_df = samples[samples['clip_name'].isin(test_clips)]
train_df = samples[~samples['clip_name'].isin(test_val_clips)]

train_df.to_csv('train.csv', index=False)
val_df.to_csv('val.csv', index=False)
test_df.to_csv('test.csv', index=False)