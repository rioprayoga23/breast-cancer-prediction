import os
import cv2
import pandas as pd

def bounding_box_img(filepath):
    img = cv2.imread(filepath)
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    contours, hierarchy = cv2.findContours(gray,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)[-2:]
    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
    return (x,y,w,h)

def extract_label(filepath):
    if 'benign' in filepath:
        return 'benign'
    else:
        return 'malignant'

masks = sorted(os.listdir('./static/assets/masks/'))

diz = {'filename':[],'xmin':[],'ymin':[],'width':[],'height':[],'label':[]}

for f in masks:
    x,y,w,h = bounding_box_img('./static/assets/masks/'+f)
    diz['filename'].append(f.replace('_mask',''))
    diz['xmin'].append(x)
    diz['ymin'].append(y)
    diz['width'].append(w)
    diz['height'].append(h)
    diz['label'].append(extract_label(f))    

df = pd.DataFrame(diz)
print(df.head())
df.to_csv('annotations.csv',index=False)