import torch
import torch.nn.functional as F
import numpy as np
import cv2
import imageio

# 位置编码函数
def positional_encoding(positions, freqs):
    freq_bands = (2**torch.arange(freqs).float()).to(positions.device)
    pts = (positions[..., None] * freq_bands).reshape(
        positions.shape[:-1] + (freqs * positions.shape[-1], ))
    pts = torch.cat([torch.sin(pts), torch.cos(pts)], dim=-1)
    return pts

def get_embedder(multires, input_dims=3):
    import torch.nn as nn
    if multires < 0:
        return nn.Identity(), input_dims
    
    embed_kwargs = {
        'include_input': True,
        'input_dims': input_dims,
        'max_freq_log2': multires-1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [torch.sin, torch.cos],
    }
    
    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj: eo.embed(x)
    return embed, embedder_obj.out_dim

class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()
        
    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x : x)
            out_dim += d
            
        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']
        
        if self.kwargs['log_sampling']:
            freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)
            
        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * freq))
                out_dim += d
                    
        self.embed_fns = embed_fns
        self.out_dim = out_dim
        
    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)

# 射线采样函数
def get_rays(directions, c2w):
    """
    从相机directions和相机到世界的变换矩阵c2w生成射线
    directions: (H, W, 3) 相机方向
    c2w: (3, 4) 相机到世界坐标变换矩阵
    """
    # 旋转矩阵部分
    rays_d = directions @ c2w[:3, :3].T
    # 平移部分作为射线起点
    rays_o = c2w[:3, 3].expand(rays_d.shape)
    return rays_o, rays_d

def get_ray_directions(H, W, focal):
    """
    获取相机射线方向
    """
    grid = create_meshgrid(H, W, normalized_coordinates=False)[0]
    i, j = grid.unbind(-1)
    
    # 相机坐标系中的方向
    directions = torch.stack([(i-W/2)/focal,
                             -(j-H/2)/focal,
                             -torch.ones_like(i)], -1)
    
    return directions

def create_meshgrid(height, width, normalized_coordinates=True, device=torch.device('cpu')):
    """创建网格坐标"""
    xs = torch.linspace(0, width - 1, width, device=device)
    ys = torch.linspace(0, height - 1, height, device=device)
    if normalized_coordinates:
        xs = (xs / (width - 1) - 0.5) * 2
        ys = (ys / (height - 1) - 0.5) * 2
    # 生成网格
    base_grid = torch.stack(torch.meshgrid([xs, ys]), dim=0)
    return base_grid.permute(1, 2, 0).unsqueeze(0)  # 1xHxWx2

def sample_along_rays(rays_o, rays_d, depth_range, n_samples, perturb=True):
    """沿射线采样点"""
    near, far = depth_range
    t_vals = torch.linspace(0., 1., steps=n_samples, device=rays_o.device)
    z_vals = near * (1.-t_vals) + far * (t_vals)
    
    if perturb:
        # 添加随机扰动
        mids = .5 * (z_vals[...,1:] + z_vals[...,:-1])
        upper = torch.cat([mids, z_vals[...,-1:]], -1)
        lower = torch.cat([z_vals[...,:1], mids], -1)
        t_rand = torch.rand(z_vals.shape, device=z_vals.device)
        z_vals = lower + (upper - lower) * t_rand
    
    pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None]
    return pts, z_vals

def raw2outputs(raw, z_vals, rays_d, raw_noise_std=0, white_bkgd=False):
    """
    将网络输出转换为RGB和深度
    raw: [num_rays, num_samples, 4] 预测的颜色和密度
    z_vals: [num_rays, num_samples] 采样深度
    rays_d: [num_rays, 3] 射线方向
    """
    raw2alpha = lambda raw, dists, act_fn=F.relu: 1.-torch.exp(-act_fn(raw)*dists)
    
    dists = z_vals[...,1:] - z_vals[...,:-1]
    dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[...,:1].shape).to(dists.device)], -1)
    
    dists = dists * torch.norm(rays_d[...,None,:], dim=-1)
    
    rgb = torch.sigmoid(raw[...,:3])  # [N_rays, N_samples, 3]
    noise = 0.
    if raw_noise_std > 0.:
        noise = torch.randn(raw[...,3].shape) * raw_noise_std
        
    alpha = raw2alpha(raw[...,3] + noise, dists)  # [N_rays, N_samples]
    weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)).to(alpha.device), 1.-alpha + 1e-10], -1), -1)[:, :-1]
    rgb_map = torch.sum(weights[...,None] * rgb, -2)  # [N_rays, 3]
    
    depth_map = torch.sum(weights * z_vals, -1)
    disp_map = 1./torch.max(1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weights, -1))
    acc_map = torch.sum(weights, -1)
    
    if white_bkgd:
        rgb_map = rgb_map + (1.-acc_map[...,None])
        
    return rgb_map, disp_map, acc_map, weights, depth_map

def img2mse(x, y): 
    return torch.mean((x - y) ** 2)

def mse2psnr(x): 
    return -10. * torch.log(x) / torch.log(torch.tensor(10.).to(x.device))

def to8b(x): 
    return (255*np.clip(x,0,1)).astype(np.uint8)

# TV损失
def total_variation_loss(inputs, TVLoss_weight, logalpha=False):
    batch_size = inputs.shape[0]
    h_x = inputs.shape[2]
    w_x = inputs.shape[3]
    count_h = (inputs.shape[2]-1) * inputs.shape[3]
    count_w = inputs.shape[2] * (inputs.shape[3] - 1)
    if logalpha:
        h_tv = torch.pow((inputs[:,:,1:,:]-inputs[:,:,:h_x-1,:]),2).sum()
        w_tv = torch.pow((inputs[:,:,:,1:]-inputs[:,:,:,:w_x-1]),2).sum()
        return TVLoss_weight*2*(h_tv/count_h+w_tv/count_w)/batch_size
    else:
        h_tv = torch.pow((inputs[:,:,1:,:]-inputs[:,:,:h_x-1,:]),2).sum()
        w_tv = torch.pow((inputs[:,:,:,1:]-inputs[:,:,:,:w_x-1]),2).sum()
        return TVLoss_weight*2*(h_tv/count_h+w_tv/count_w)/batch_size

# L1损失
def L1_loss_sparse(inputs, TVLoss_weight):
    return TVLoss_weight * torch.mean(torch.abs(inputs))

# 正交损失
def OrthLoss(inputs):
    n_comp, n_size = inputs.shape[0], inputs.shape[1]
    inputs = inputs / (inputs.norm(dim=1,keepdim=True)+1e-8)
    cosine_sim = inputs @ inputs.T
    return ((cosine_sim-torch.eye(n_comp,device=inputs.device))**2).mean()

# 渐进式缩放
def N_to_reso(n_voxels, bbox):
    xyz_min, xyz_max = bbox
    dim = len(xyz_min)
    voxel_size = ((xyz_max - xyz_min).prod() / n_voxels).pow(1/dim)
    return ((xyz_max - xyz_min) / voxel_size).long().tolist()

def cal_n_samples(reso, step_ratio=0.5):
    return int(np.linalg.norm(reso)/step_ratio)

# 修复缺失的函数
def get_ray_directions_nerf(H, W, focal, center=None):
    """获取NeRF风格的射线方向"""
    if center is None:
        center = [W//2, H//2]
    
    i, j = np.meshgrid(np.arange(W, dtype=np.float32),
                       np.arange(H, dtype=np.float32), indexing='xy')
    
    dirs = np.stack([(i-center[0])/focal, -(j-center[1])/focal, -np.ones_like(i)], -1)
    return dirs 