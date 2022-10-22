import pandas as pd
import random

random.seed(42)

samples = pd.read_csv('samples.csv')
clips = samples['clip_name'].unique()

test, val = random.sample(clips.tolist(), k=2)

val_df = samples[samples['clip_name'] == val]
test_df = samples[samples['clip_name'] == test]
train_df = samples[~(samples.index.isin(val_df.index) | samples.index.isin(test_df.index))]

train_df.to_csv('train.csv', index=False)
val_df.to_csv('val.csv', index=False)
test_df.to_csv('test.csv', index=False)