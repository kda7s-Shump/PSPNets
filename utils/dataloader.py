import glob
import os

import numpy as np
import torch.utils.data as data
from PIL import Image

from utils.data_augumentation import Compose, Scale, RandomRotation, RandomMirror, Resize, Normalize_Tensor_RGB

def make_anno_palette_dict(root:str, filename:str)->(dict, dict):
  palette_dict = dict()
  anno_dict = dict()
  
  with open(root+filename, 'r') as f:
    for idx, line in enumerate(f.readlines()):
      palette = line.rstrip().split(',')
      
      # パレットのディクショナリを作成
      rgb = tuple(map(int, palette[0].split()))
      palette_dict[rgb] = idx

      # パレットのインデックスの名前のディクショナリを作成
      name = palette[1]  if palette[1] != "" else palette[2]
      anno_dict[idx] = name

  return palette_dict, anno_dict

def make_datapath_list(root:str, dir:list = None, filename:str = '*')->(np.ndarray, np.ndarray, list, list):
  
  target_path = os.path.join(root, dir[0], filename)
  path_list = [path for path in glob.glob(target_path)]

  # データをtrain/valに分割する
  path_length = len(path_list)
  path_list = np.random.choice(path_list, size=path_length)
  
  '''アノテーションがないファイルは削除'''
  tmp = list(path_list)
  for p in tmp:
    if not os.path.exists(p.replace(dir[0], dir[1]).replace('jpg', 'bmp')):
      tmp.remove(p)
  path_list = np.array(tmp)
  '''   '''

  train_img_list, val_img_list = np.split(path_list, [int(path_length*0.8)])
  train_anno_list = [path.replace(dir[0], dir[1]).replace('jpg', 'bmp') for path in train_img_list]
  val_anno_list   = [path.replace(dir[0], dir[1]).replace('jpg', 'bmp') for path in val_img_list]

  return train_img_list, train_anno_list, val_img_list, val_anno_list # データへのパスを格納したリスト

class ImagePreprocessing():
    """
    画像とアノテーションの前処理クラス。訓練時と検証時で異なる動作をする。
    画像のサイズをinput_size x input_sizeにする。
    訓練時はデータオーギュメンテーションする。

    Attributes
    ----------
    input_size : int
        リサイズ先の画像の大きさ。
    """

    def __init__(self, input_size:int, palette_dict:dict):
        self.data_transform = {
            'train': Compose([
                Scale(scale=[0.5, 1.5]),  # 画像の拡大
                RandomRotation(angle=[-10, 10]),  # 回転
                RandomMirror(),  # ランダムミラー
                Resize(input_size),  # リサイズ(input_size)
                Normalize_Tensor_RGB(palette_dict)  # 色情報の標準化とテンソル化
            ]),
            'val': Compose([
                Resize(input_size),  # リサイズ(input_size)
                Normalize_Tensor_RGB(palette_dict)  # 色情報の標準化とテンソル化
            ])
        }

    def __call__(self, phase, img, anno_class_img):
        """
        Parameters
        ----------
        phase : 'train' or 'val'
            前処理のモードを指定。
        """
        return self.data_transform[phase](img, anno_class_img)