import sys
import os
import cv2
import random
import torch
import numpy as np
import _pickle as cPickle
import torch.utils.data as data
import copy
import pytorch3d
from PIL import Image
sys.path.insert(0, '../')

from ipdb import set_trace
from tqdm import tqdm
from torch.utils.data.distributed import DistributedSampler
from utils.data_augmentation import defor_2D, get_rotation
from utils.data_augmentation import data_augment
from utils.datasets_utils import aug_bbox_DZI, get_2d_coord_np, crop_resize_by_warp_affine, load_obj, apply_transformation
from utils.sgpa_utils import load_depth, get_bbox
from configs.config import get_config
from utils.misc import get_rot_matrix, pc_normalize

def depth_to_pointcloud(K, depth_map):
    """
    将深度图像反投影为相机坐标系下的点云。
    
    参数:
    - K: 相机内参矩阵 (3x3)
    - depth_map: 深度图像，形状为 (H, W)，每个像素值表示深度 Z_{cam}

    返回:
    - 相机坐标系下的点云，形状为 (N, 3)，其中 N 为非零深度值的像素数量
    """
    # 获取图像尺寸
    height, width = depth_map.shape

    # 创建每个像素的 u, v 坐标网格
    u, v = np.meshgrid(np.arange(width), np.arange(height))

    # 将像素坐标转换为齐次坐标 (u, v, 1)
    pixel_coords = np.stack([u, v, np.ones_like(u)], axis=-1).reshape(-1, 3)

    # 获取有效深度 (Z > 0 的像素)
    valid_depth_mask = depth_map > 0
    valid_depths = depth_map[valid_depth_mask].reshape(-1)

    # 过滤有效的像素坐标
    valid_pixel_coords = pixel_coords[valid_depth_mask.reshape(-1)]

    # 计算相机内参的逆矩阵
    K_inv = np.linalg.inv(K)
    
    # 反投影：将像素坐标转换到相机坐标系
    points_3D_cam = (K_inv @ valid_pixel_coords.T).T * valid_depths[:, np.newaxis]
    
    return points_3D_cam

categories = {
    'airplane': '02691156',
    'ashcan': '02747177',
    'bag': '02773838',
    'basket': '02801938', 
    'bathtub': '02808440', 
    'bed': '02818832', 
    'bench': '02828884', 
    'birdhouse': '02843684', 
    'bookshelf': '02871439', 
    'bottle': '02876657', 
    'bowl': '02880940', 
    'bus': '02924116', 
    'cabinet': '02933112', 
    'camera': '02942699', 
    'can': '02946921', 
    'cap': '02954340', 
    'car': '02958343', 
    'chair': '03001627', 
    'clock': '03046257', 
    'keypad': '03085013', 
    'dishwasher': '03207941', 
    'display': '03211117', 
    'earphone': '03261776', 
    'faucet': '03325088', 
    'file': '03337140', 
    'guitar': '03467517', 
    'helmet': '03513137', 
    'jar': '03593526', 
    'knife': '03624134', 
    'lamp': '03636649', 
    'laptop': '03642806', 
    'loudspeaker': '03691459', 
    'mailbox': '03710193', 
    'microphone': '03759954', 
    'microwave': '03761084', 
    'motorcycle': '03790512', 
    'mug': '03797390', 
    'piano': '03928116', 
    'pillow': '03938244', 
    'pistol': '03948459', 
    'pot': '03991062', 
    'printer': '04004475', 
    'remote': '04074963', 
    'rifle': '04090263', 
    'rocket': '04099429', 
    'skateboard': '04225987', 
    'sofa': '04256520', 
    'stove': '04330267', 
    'table': '04379243', 
    'telephone': '04401088', 
    'cellphone': '02992529', 
    'tower': '04460130', 
    'train': '04468005', 
    'vessel': '04530566', 
    'washer': '04554684'
    }
DEBUG = False

class ShapeNetDataSet(data.Dataset):
    def __init__(self,
                dynamic_zoom_in_params,
                deform_2d_params,
                source=None, 
                mode='train', 
                data_dir=None,
                n_pts=1024, 
                img_size=256, 
                per_obj='',
                ):

        # TODO： config和NOCS接口整合
        self.data_root = "./data/shapenet" # config.DATA_PATH # /home/fudan248/zhangjinyu/code_repo/GenPose
        self.subset = mode # config.subset
        self.obj_path = "./data/shapenet/objs" # config.OBJ_PATH
        self.n_pts = n_pts
        self.camera_intrinsics = np.array([[761.1827392578125, 0, 320], [0, 761.1827392578125, 240], [0, 0, 1]])
        
        # self.add_gaussian_noise = config.GAUSSIAN_NOISE
        
        # self.cate_num = config.CATE_NUM
        # self.data_list_file = os.path.join(self.data_root, f'{self.subset}.txt')
        if self.subset == 'train':
            self.data_list_file = os.path.join(self.data_root, f'500view_rgbd_shapenet_train_list.txt')
        elif self.subset == 'val' or self.subset == 'test':
            self.data_list_file = os.path.join(self.data_root, f'500view_rgbd_shapenet_test_list.txt')
        # self.data_list_file = os.path.join(self.data_root, f'500view_shapenet_train_list.txt')

        print(f'[DATASET] Open file {self.data_list_file}')
        with open(self.data_list_file, 'r') as f:
            lines = f.readlines()
        
        self.file_list = []
        for line in lines:
            line = line.strip()
            taxonomy_id = categories[line.split('/')[-2]]
            
            # # DEBUG:
            # if taxonomy_id not in ["03797390", "02946921", '02876657']:
            #     continue
            
            model_id = line.split('/')[-1]
            line = line.replace('./', '/home/fudan248/zhangjinyu/code_repo/GenPose/')
            if self.subset == 'train':
                for idx in range(500):
                    self.file_list.append({
                            'taxonomy_id': taxonomy_id,
                            'model_id': model_id,
                            # 'obj_path': os.path.join(obj_path, taxonomy_id, model_id, 'models', 'model_normalized.obj'),
                            # 'obj_path': os.path.join(self.obj_path, f'{taxonomy_id}-{model_id}.npy'),
                            'pose_path': os.path.join(line, f'{idx:04}_pose.txt'),
                            # 'pcd_path': os.path.join(line, f'{idx:04}_pcd.obj'),
                            'depth_path': os.path.join(line, f'{idx:04}_depth.png'),
                        })
                    
                if DEBUG:
                    break
            elif self.subset == 'val':
                for idx in range(0, 500, 13):
                    self.file_list.append({
                            'taxonomy_id': taxonomy_id,
                            'model_id': model_id,
                            # 'obj_path': os.path.join(obj_path, taxonomy_id, model_id, 'models', 'model_normalized.obj'),
                            # 'obj_path': os.path.join(self.obj_path, f'{taxonomy_id}-{model_id}.npy'),
                            'pose_path': os.path.join(line, f'{idx:04}_pose.txt'),
                            # 'pcd_path': os.path.join(line, f'{idx:04}_pcd.obj'),
                            'depth_path': os.path.join(line, f'{idx:04}_depth.png'),
                        })
                if DEBUG:
                    break
            else:
                # if taxonomy_id != '02942699':
                #     continue
                for idx in range(0, 500, 33):
                    self.file_list.append({
                            'taxonomy_id': taxonomy_id,
                            'model_id': model_id,
                            # 'obj_path': os.path.join(obj_path, taxonomy_id, model_id, 'models', 'model_normalized.obj'),
                            # 'obj_path': os.path.join(self.obj_path, f'{taxonomy_id}-{model_id}.npy'),
                            'pose_path': os.path.join(line, f'{idx:04}_pose.txt'),
                            # 'pcd_path': os.path.join(line, f'{idx:04}_pcd.obj'),
                            'depth_path': os.path.join(line, f'{idx:04}_depth.png'),
                        })
                if DEBUG:
                    break

        if DEBUG:
            self.file_list = self.file_list[:400]
        
        print(f'[DATASET] {len(self.file_list)} instances were loaded')
        
    def _sample_points(self, pcl, n_pts):
        """ Down sample the point cloud using farthest point sampling.

        Args:
            pcl (torch tensor or numpy array):  NumPoints x 3
            num (int): target point number
        """
        total_pts_num = pcl.shape[0]
        if total_pts_num < n_pts:
            pcl = np.concatenate([np.tile(pcl, (n_pts // total_pts_num, 1)), pcl[:n_pts % total_pts_num]], axis=0)
        elif total_pts_num > n_pts:
            ids = np.random.permutation(total_pts_num)[:n_pts]
            pcl = pcl[ids]
        return pcl

    def __getitem__(self, idx):
        # TODO: 查看这里的file_list的格式
        sample = self.file_list[idx]        
        _pose = np.loadtxt(sample['pose_path'])
        # _complete_pc = np.load(sample['obj_path'])
        cat_id = sample['taxonomy_id']
        depth_path = sample['depth_path']
        depth = np.array(Image.open(depth_path))
        # NOTE: 更改为从depth获得点云
        # pcl_in, _ = load_obj(sample['pcd_path'])
        pcl_in = depth_to_pointcloud(self.camera_intrinsics, depth)
        pcl_in *= 0.001
        # NOTE：更改为genpose的fps, 后续1024要用self.n_pts代替
        pcl_in = self._sample_points(pcl_in, n_pts=self.n_pts)
        # complete_pc = apply_transformation(_complete_pc, _pose) # NOTE: 现在还不需要

        # TODO：检查genpose是不是在dataset中进行的中心化
        pcl_in, centroid, scale = pc_normalize(pcl_in)
        # data['gt'] = (complete_pc - centroid) / scale
        

        # NOTE：genpose dataset里传入了nocs_scale(缩放因子)和fsnet_scale(真实值),我们没有fsnet_scale, 
        trans_mat = (_pose[:3, 3].flatten() - centroid) / scale
        # min_x, max_x = np.min(_complete_pc[:, 0]), np.max(_complete_pc[:, 0])
        # min_y, max_y = np.min(_complete_pc[:, 1]), np.max(_complete_pc[:, 1])
        # min_z, max_z = np.min(_complete_pc[:, 2]), np.max(_complete_pc[:, 2])
        # size_mat = np.array(((max_x - min_x) / scale, (max_y - min_y) / scale, (max_z - min_z) / scale)) 
        
        
        rotation = _pose[:3, :3]
        # NOTE：训练集和验证集的处理方式相同 
        # TODO：查看这些设置为NONE的变量是否有作用
        data_dict = {}
        data_dict['pcl_in'] = torch.as_tensor(pcl_in.astype(np.float32)).contiguous()
        data_dict['cat_id'] = torch.as_tensor(int(cat_id), dtype=torch.int32).contiguous() # TODO: 这里应该是分类id
        data_dict['rotation'] = torch.as_tensor(rotation, dtype=torch.float32).contiguous()
        data_dict['translation'] = torch.as_tensor(trans_mat, dtype=torch.float32).contiguous()
        # data_dict['fsnet_scale'] = torch.as_tensor(size_mat, dtype=torch.float32).contiguous()
        # data_dict['sym_info'] = None
        # data_dict['mean_shape'] = None
        # data_dict['aug_bb'] = None
        # data_dict['aug_rt_t'] = None
        # data_dict['aug_rt_R'] = None
        # data_dict['model_point'] = None #NOTE:后续可以用complete_pc代替
        # data_dict['nocs_scale'] = None
        data_dict['handle_visibility'] = 1 #NOTE: 1是忽略
        # data_dict['path'] = None
        if data_dict is None:
            print('None data_dict')
            raise ValueError('None data_dict')
        return data_dict


    def __len__(self):
        return len(self.file_list)
    

if __name__ == '__main__':
    # check if the dataset is working correctly
    dataset = ShapeNetDataSet(dynamic_zoom_in_params=None, deform_2d_params=None, source=None, mode='train', data_dir=None, n_pts=1024, img_size=256, per_obj='')
    for i in range(len(dataset)):
        item = dataset[i]
        print(f"Item {i} has {len(item['pcl_in'])} points")