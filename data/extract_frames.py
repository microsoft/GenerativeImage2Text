import pandas as pd
import cv2
import os


samples = pd.read_csv('samples.csv')

for clip_name, df in samples.groupby('clip_name'):
    frames = set(df['frame'])
    vidcap = cv2.VideoCapture(os.path.join('frame', clip_name + '.mpg'))
    frame_count = 0
    while True:
        success,image = vidcap.read()
        if not success:
            break
        if frame_count in frames:
            cv2.imwrite(os.path.join('images', f'{clip_name}_{frame_count}.png'), image)
        frame_count += 1
 