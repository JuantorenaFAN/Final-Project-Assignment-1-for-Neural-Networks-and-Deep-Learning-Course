import os
import torch
import numpy as np
import cv2
from PIL import Image
import json
from torch.utils.data import Dataset
from utils import get_ray_directions, get_rays
import glob

class NeRFDataset(Dataset):
    def __init__(self, root_dir, split='train', img_wh=(800, 800), white_bg=True, cache_dir='cache'):
        """
        数据集类初始化
        root_dir: 包含图像的数据目录
        split: 数据集分割类型 ('train', 'val', 'test')
        img_wh: 图像的目标宽度和高度
        white_bg: 是否使用白色背景
        cache_dir: 缓存目录，用于存储处理过的数据
        """
        self.root_dir = root_dir
        self.split = split
        self.img_wh = img_wh
        self.white_bg = white_bg
        self.cache_dir = cache_dir
        
        self.scene_bbox = torch.tensor([[-1.5, -1.5, -1.5], [1.5, 1.5, 1.5]], dtype=torch.float32)
        self.blender2opencv = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        
        # 加载数据
        self.load_data()
        
    def load_data(self):
        """加载或处理数据"""
        cache_file = os.path.join(self.cache_dir, f'data_{self.split}_{self.img_wh[0]}x{self.img_wh[1]}.pth')
        
        if os.path.exists(cache_file):
            print(f"从缓存加载数据: {cache_file}")
            data = torch.load(cache_file)
            self.all_rays = data['rays']
            self.all_rgbs = data['rgbs']
            self.focal = data['focal']
            self.poses = data['poses']
            self.img_files = data['img_files']
        else:
            print("处理原始数据...")
            self.process_raw_data()
            
            # 创建缓存目录
            os.makedirs(self.cache_dir, exist_ok=True)
            
            # 保存处理后的数据
            torch.save({
                'rays': self.all_rays,
                'rgbs': self.all_rgbs,
                'focal': self.focal,
                'poses': self.poses,
                'img_files': self.img_files
            }, cache_file)
            print(f"数据已缓存至: {cache_file}")
    
    def process_raw_data(self):
        """处理原始图像数据，估计相机参数"""
        # 获取所有图像文件
        img_files = sorted(glob.glob(os.path.join(self.root_dir, '*.jpg')) + 
                          glob.glob(os.path.join(self.root_dir, '*.png')) +
                          glob.glob(os.path.join(self.root_dir, '*.JPG')))
        
        if len(img_files) == 0:
            raise ValueError(f"在 {self.root_dir} 中没有找到图像文件")
        
        print(f"找到 {len(img_files)} 张图像")
        
        # 估计相机参数（简化版本，假设所有图像使用相同的内参）
        # 在实际应用中，这里应该使用COLMAP
        sample_img = cv2.imread(img_files[0])
        H, W = sample_img.shape[:2]
        
        # 估计焦距（假设水平视场角为50度）
        focal = 0.5 * W / np.tan(0.5 * 50 * np.pi / 180)
        self.focal = focal
        
        # 数据集分割
        n_imgs = len(img_files)
        if self.split == 'train':
            img_indices = list(range(0, n_imgs, 1))  # 所有图像用于训练
            img_indices = img_indices[:-10]  # 留最后10张作为测试
        elif self.split == 'val':
            img_indices = list(range(n_imgs-10, n_imgs-5))  # 倒数5-10张作为验证
        else:  # test
            img_indices = list(range(n_imgs-5, n_imgs))  # 最后5张作为测试
        
        # 生成相机姿态（简化版本：圆形轨道）
        poses = []
        for i in img_indices:
            # 围绕物体的圆形轨道
            angle = 2 * np.pi * i / n_imgs
            radius = 3.0
            x = radius * np.cos(angle)
            z = radius * np.sin(angle)
            y = 0.5  # 稍微向上倾斜
            
            # 构建相机姿态矩阵
            camera_pos = np.array([x, y, z])
            target = np.array([0, 0, 0])  # 看向原点
            up = np.array([0, 1, 0])
            
            # 计算相机方向
            forward = target - camera_pos
            forward = forward / np.linalg.norm(forward)
            right = np.cross(forward, up)
            right = right / np.linalg.norm(right)
            up = np.cross(right, forward)
            
            # 构建相机到世界的变换矩阵
            c2w = np.eye(4)
            c2w[:3, 0] = right
            c2w[:3, 1] = up
            c2w[:3, 2] = -forward
            c2w[:3, 3] = camera_pos
            
            poses.append(c2w[:3, :4])
        
        self.poses = np.array(poses)
        self.img_files = [img_files[i] for i in img_indices]
        
        # 处理图像和生成射线
        all_rays = []
        all_rgbs = []
        
        # 获取射线方向
        directions = get_ray_directions(self.img_wh[1], self.img_wh[0], self.focal)
        
        for img_idx, img_file in enumerate(self.img_files):
            print(f"处理图像 {img_idx+1}/{len(self.img_files)}: {os.path.basename(img_file)}")
            
            # 加载和预处理图像
            img = Image.open(img_file).convert('RGB')
            img = img.resize(self.img_wh, Image.LANCZOS)
            img = np.array(img) / 255.0
            
            if self.white_bg:
                # 假设黑色区域为背景，替换为白色
                mask = np.all(img < 0.1, axis=-1)
                img[mask] = 1.0
            
            img = torch.FloatTensor(img)
            
            # 生成射线
            c2w = torch.FloatTensor(self.poses[img_idx])
            rays_o, rays_d = get_rays(directions, c2w)
            
            # 展平
            rays_o = rays_o.view(-1, 3)
            rays_d = rays_d.view(-1, 3)
            rgb = img.view(-1, 3)
            
            rays = torch.cat([rays_o, rays_d], dim=-1)  # [N, 6]
            
            all_rays.append(rays)
            all_rgbs.append(rgb)
        
        self.all_rays = torch.cat(all_rays, dim=0)
        self.all_rgbs = torch.cat(all_rgbs, dim=0)
        
        print(f"数据加载完成: {len(self.all_rays)} 条射线")
    
    def __len__(self):
        return len(self.all_rays)
    
    def __getitem__(self, idx):
        return {
            'rays': self.all_rays[idx],
            'rgbs': self.all_rgbs[idx]
        }

def get_spiral_path(poses, n_frames=120, n_rots=2, zrate=.5):
    """生成螺旋路径用于视频渲染"""
    c2w = poses_avg(poses)
    
    # 获取螺旋参数
    up = normalize(poses[:, :3, 1].sum(0))
    close_depth, inf_depth = bds.min()*.9, bds.max()*5.
    dt = .75
    mean_dz = 1./(((1.-dt)/close_depth + dt/inf_depth))
    focal = mean_dz
    
    # 生成螺旋路径
    shrink_factor = .8
    zdelta = close_depth * .2
    tt = poses[:,:3,3]
    rads = np.percentile(np.abs(tt), 90, 0)
    c2w_path = c2w
    N_views = n_frames
    N_rots = n_rots
    
    # 螺旋参数
    render_poses = []
    rads = np.array(list(rads) + [1.])
    hwf = c2w_path[:, 4:5]
    
    for theta in np.linspace(0., 2. * np.pi * N_rots, N_views+1)[:-1]:
        c = np.dot(c2w[:3,:4], np.array([np.cos(theta), -np.sin(theta), -np.sin(theta*zrate), 1.]) * rads)
        z = normalize(c - np.dot(c2w[:3,:4], np.array([0,0,-focal, 1.])))
        render_poses.append(np.concatenate([viewmatrix(z, up, c), hwf], 1))
    return render_poses

def poses_avg(poses):
    """计算平均姿态"""
    hwf = poses[0, :3, -1:]
    center = poses[:, :3, 3].mean(0)
    vec2 = normalize(poses[:, :3, 2].sum(0))
    up = poses[:, :3, 1].sum(0)
    c2w = np.concatenate([viewmatrix(vec2, up, center), hwf], 1)
    return c2w

def normalize(x):
    """归一化向量"""
    return x / np.linalg.norm(x)

def viewmatrix(z, up, pos):
    """从z轴、up向量和位置构建视图矩阵"""
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, pos], 1)
    return m 