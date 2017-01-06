from os.path import basename
import sys, os, imtools, pickle

load_model = 'Human.pkl'    # limit 24 character

# 1. Make Directory
dirname = 'Temp'

if not os.path.isdir(dirname):
    os.mkdir(dirname)


# 2. Save Screen Shot to Directory
filename = 'video/stone.avi'
filter = 'scene'
format = 'jpg'
rate = 5
vout = 'dummy'
prefix = 'img_'
ratio = 200

cmmd = str('vlc ' + filename + ' --video-filter=' + filter + ' --scene-format=' + format + ' --rate=' + str(rate) + '--vout=' + vout + ' --scene-prefix=' + prefix +
			' --scene-ratio=' + str(ratio) + '--scene-path=' + dirname)
os.system(cmmd)


# 3. Extract Features

# 4. Run Classifier against extracted features
