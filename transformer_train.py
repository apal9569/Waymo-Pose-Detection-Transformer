#@title Load data from GCP Bucket

from typing import Optional, Any, Mapping, Tuple
import warnings
# Disable annoying warnings from PyArrow using under the hood.
warnings.simplefilter(action='ignore', category=FutureWarning)
import gcsfs
from model import cnnLayer
import dask.dataframe as dd
import matplotlib.pyplot as plt
import numpy as np
import plotly  # Used by visu3d
import tensorflow as tf
import visu3d
import os
from waymo_open_dataset import v2
from waymo_open_dataset.utils import camera_segmentation_utils
import os
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import math
import numpy as np
from tqdm import tqdm

from transformer_model import Transformer
location = '/home/016080064/waymo-open-dataset/'
object_asset_utils = v2.object_asset_utils
os.chdir('../../../')
#@markdown Specify data split
data_split = 'training' #@param ["validation", "training"]
# Path to the directory with all components of perception dataset.
dataset_dir = f'data/cmpe249-fa23/id-waymo-200/{data_split}'
# Path to the directory with all components of the perception object asset dataset.
asset_dir = dataset_dir

KeypointType= v2.perception.keypoints.KeypointType

waymo_to_coco_keypoints = {
        # KeypointType.UNSPECIFIED : 0,
        KeypointType.FOREHEAD : 13,
        KeypointType.NOSE : 0,
        KeypointType.LEFT_SHOULDER : 1,
        KeypointType.LEFT_ELBOW : 2,
        KeypointType.LEFT_WRIST : 3,
        KeypointType.LEFT_HIP : 4,
        KeypointType.LEFT_KNEE : 5,
        KeypointType.LEFT_ANKLE : 6,
        KeypointType.RIGHT_SHOULDER : 7,
        KeypointType.RIGHT_ELBOW : 8,
        KeypointType.RIGHT_WRIST : 9,
        KeypointType.RIGHT_HIP : 10,
        KeypointType.RIGHT_KNEE : 11,
        KeypointType.RIGHT_ANKLE : 12
        # KeypointType.KEYPOINT_TYPE_HEAD_CENTER : 20
    }

def read(dataset_dir: str, tag: str, context_name: str = '9547911055204230158_1567_950_1587_950.parquet') -> dd.DataFrame:
  """Creates a Dask DataFrame for the component specified by its tag."""
  paths = tf.io.gfile.glob(f'{dataset_dir}/{tag}/{context_name}')
  # print(f'{dataset_dir}/{tag}/{context_name}')
  # print(paths, f'{dataset_dir}/{tag}/{context_name}')
  return dd.read_parquet(paths)



def read_codec(dataset_dir: str, tag: str) -> Any:
  """Reads the codec file for 3D ray decompression."""
  codec_path = tf.io.gfile.glob(f'{dataset_dir}/{tag}/codec_config.json')[0]
  with tf.io.gfile.GFile(codec_path, 'r') as f:
    codec_config = v2.ObjectAssetRayCodecConfig.from_json(f.read())
  return v2.ObjectAssetRayCodec(codec_config)

D = 64
H = 4
Nx = 3
D_ff = 128
maxL = 64
dropout = 0.1

transformer = Transformer(D, H, Nx, D_ff, maxL, dropout)
opt = optim.Adagrad(transformer.parameters(), lr=0.0001)

final_arr = []
images = []
files = os.listdir(dataset_dir + "/camera_image")
files = files[:50]
# print(files)
training = 150
c = 0
for steps in tqdm(range(training)):

    for file in files:
        # print(file)
        camera = read(asset_dir, 'camera_image', file)
        hkp = read(asset_dir, 'camera_hkp', file)
        cam_hkp = v2.merge(camera, hkp)
        c = 0
        try:
            batch = int(len(cam_hkp)/32)
            remainder = int(len(cam_hkp)%32)
        except:
            continue

        for i, (_, r) in enumerate(cam_hkp.iterrows()): 
            camera = v2.CameraImageComponent.from_dict(r)
            hkp = v2.CameraHumanKeypointsComponent.from_dict(r)
            binary_data = bytes(camera.image)
        
            with open(location + str(i) + ".jpg", "wb") as f:
                f.write(binary_data)
            img = plt.imread(location + str(i) + ".jpg")
            if img.shape != (1280, 1920 ,3):
                os.remove(location + str(i) + ".jpg")
                continue
                
            img = torch.from_numpy(img)
            images.append(img.permute(2,0,1))
            
            x = np.zeros(14)
            y = np.zeros(14)

            for index, val in enumerate(hkp.camera_keypoints.type):
                
                x[waymo_to_coco_keypoints.get(KeypointType(val))] = hkp.camera_keypoints.keypoint_2d.location_px.x[index]/img.shape[0]
                y[waymo_to_coco_keypoints.get(KeypointType(val))] = hkp.camera_keypoints.keypoint_2d.location_px.y[index]/img.shape[1]

            
            final_arr.append(torch.stack((torch.from_numpy(x),torch.from_numpy(y)), dim=1))
            os.remove(location + str(i) + ".jpg")
            c+=1
            
            if (batch and c==32) or (batch == 0 and c == (remainder)):
                out = transformer(torch.stack(images).to(torch.float))
                out = out.permute(1, 0, 2).reshape(-1, 14, 2)
        
                sum_distances = 0
                for actual, pred in zip(final_arr, out):
                    distances = torch.norm(actual - pred, dim=1)
                    sum_distances += distances.sum()
                
                sum_distances = sum_distances/len(final_arr)
                print(sum_distances)
                
                opt.zero_grad()
                sum_distances.backward()
                opt.step()
                final_arr = []
                images = []
                batch-=1
                c = 0
        
    
    torch.save(transformer, location + f'/models/model-transformer-{steps}.pt')