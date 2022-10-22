import itertools
import os
import csv
import re
import pandas as pd

first_5_frames_only = {}
normalize_word_map = {}
def normalize_words(str):
    words = str.split()
    normalized = []
    for word in words:
        if '_' not in word:
            normalized.append(word)
            continue
        parts = word.split('_')
        if parts[-1].isnumeric():
            parts.pop()
        normalized.extend(parts)
        normalize_word_map[word] = ' '.join(parts)
    return ' '.join(normalized)


descriptions = []
description_pattern = re.compile(r'(\w+)-motif_(\d+).csv')
for root, _, files in os.walk(os.path.join('description', 'SURF')):
    for file in files:
        match = re.match(description_pattern, file)
        if not match:
            print('invalid file', file)
            continue
        clip_name, motif_id = match.groups()
        motif_id = int(motif_id)
        with open(os.path.join(root, file), newline='') as csv_file:
            reader = csv.reader(csv_file)
            for i, row in enumerate(reader):
                if i % 2: # skip list of frames
                    description = row[0]
                    if not description:
                        continue
                    if ';' in description:
                        description = description.split(';')[0]
                        first_5_frames_only[(clip_name, motif_id, annotation_id)] = 0
                    annotation_id = i // 2
                    descriptions.append((clip_name, motif_id, annotation_id, normalize_words(description)))
descriptions_df = pd.DataFrame(descriptions, columns=['clip_name', 'motif_id', 'annotation_id', 'description'])

frames = []
frame_pattern = re.compile(r'(\w+)-(\d+)-(\d+)-(\d+)')
for file_name in os.listdir('frame'):
    root, ext = os.path.splitext(file_name)
    if ext == '.csv':
        with open(os.path.join('frame', file_name), newline='') as csv_file:
            reader = csv.reader(csv_file)
            for frame_name in itertools.chain.from_iterable(reader):
                if not frame_name:
                    continue
                match = re.match(frame_pattern, frame_name)
                assert match
                clip_name, motif_id, annotation_id, frame = match.groups()
                motif_id, annotation_id, frame = int(motif_id), int(annotation_id), int(frame) 
                if (clip_name, motif_id, annotation_id) in first_5_frames_only:
                    if first_5_frames_only[(clip_name, motif_id, annotation_id)] >= 5:
                        continue
                    first_5_frames_only[(clip_name, motif_id, annotation_id)]  += 1
                sample_id = f'{clip_name}_{motif_id}_{annotation_id}_{frame}'
                image_name = f'{clip_name}_{frame}.png'
                frames.append((sample_id, clip_name, motif_id, annotation_id, frame, image_name))

frames_df = pd.DataFrame(frames, columns=['sample_id', 'clip_name', 'motif_id', 'annotation_id', 'frame', 'image_name'])
sampled_frames_df = frames_df.groupby(['clip_name', 'motif_id', 'annotation_id']).sample(n=2, replace=True, random_state=42)


merged = sampled_frames_df.merge(descriptions_df, how='inner', on=['clip_name', 'motif_id', 'annotation_id'])
merged.to_csv('samples.csv', index=False)
for original, normalized in normalize_word_map.items():
    print(f'{original}\t->\t{normalized}')
print(merged)