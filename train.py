import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import time
import imageio

from config import get_opts
from dataset import NeRFDataset
from model import TensoRF, MLPRenderFeature
from utils import img2mse, mse2psnr, to8b, total_variation_loss, L1_loss_sparse, OrthLoss, N_to_reso, cal_n_samples

def setup_model(args, device):
    """初始化模型"""
    # 场景边界框
    aabb = torch.tensor([[-1.5, -1.5, -1.5], [1.5, 1.5, 1.5]], dtype=torch.float32)
    
    # 初始网格大小
    reso_cur = [128, 128, 128]  # 简化的初始分辨率
    nSamples = 64  # 简化的采样数
    
    # 初始化TensoRF模型
    tensorf = TensoRF(aabb, reso_cur, device,
                      n_lamb_sigma=args.n_lamb_sigma, 
                      n_lamb_sh=args.n_lamb_sh,
                      shadingMode=args.shadingMode,
                      pos_pe=args.pos_pe,
                      view_pe=args.view_pe,
                      fea_pe=args.fea_pe,
                      featureC=args.featureC,
                      step_ratio=args.step_ratio)
    
    # 初始化渲染MLP
    if args.shadingMode == 'MLP_PE':
        tensorf.renderModule = None
    elif args.shadingMode == 'MLP_Fea':
        tensorf.renderModule = MLPRenderFeature(args.featureC, args.view_pe, args.fea_pe, args.featureC).to(device)
    
    tensorf = tensorf.to(device)
    
    # 设置优化器参数组
    grad_vars = tensorf.get_optparam_groups(args.lr_init, args.lr_basis)
    if args.lr_decay_iters > 0:
        lr_factor = args.lr_decay_target_ratio**(1/args.lr_decay_iters)
    else:
        args.lr_decay_iters = args.n_iters
        lr_factor = args.lr_decay_target_ratio**(1/args.n_iters)
    
    # 优化器
    optimizer = torch.optim.Adam(grad_vars, betas=(0.9, 0.99))
    
    return tensorf, optimizer, lr_factor

def setup_data(args):
    """设置数据加载器"""
    train_dataset = NeRFDataset(args.data_dir, split='train', img_wh=args.img_wh)
    val_dataset = NeRFDataset(args.data_dir, split='val', img_wh=args.img_wh) 
    test_dataset = NeRFDataset(args.data_dir, split='test', img_wh=args.img_wh)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                             shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, 
                           shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, 
                            shuffle=False, num_workers=args.num_workers)
    
    return train_loader, val_loader, test_loader, train_dataset

def compute_loss(rgb_pred, rgb_gt, args, tensorf, global_step):
    """计算损失函数"""
    # 主要重建损失
    loss = img2mse(rgb_pred, rgb_gt)
    psnr = mse2psnr(loss).item()
    
    # L1损失
    if args.L1_weight_inital > 0:
        L1_reg_loss = L1_loss_sparse(tensorf.density_plane, args.L1_weight_inital) + \
                      L1_loss_sparse(tensorf.app_plane, args.L1_weight_inital)
        loss += L1_reg_loss
    
    if args.L1_weight_rest > 0:
        L1_reg_loss = L1_loss_sparse(tensorf.density_line, args.L1_weight_rest) + \
                      L1_loss_sparse(tensorf.app_line, args.L1_weight_rest)
        loss += L1_reg_loss
    
    # TV损失
    if args.TV_weight_density > 0:
        TV_weight_density = args.TV_weight_density
        loss += total_variation_loss(tensorf.density_plane, TV_weight_density) + \
                total_variation_loss(tensorf.density_line, TV_weight_density)
    
    if args.TV_weight_app > 0:
        TV_weight_app = args.TV_weight_app
        loss += total_variation_loss(tensorf.app_plane, TV_weight_app) + \
                total_variation_loss(tensorf.app_line, TV_weight_app)
    
    # 正交损失
    if args.Ortho_weight > 0:
        loss += OrthLoss(tensorf.density_plane) * args.Ortho_weight + \
                OrthLoss(tensorf.app_plane) * args.Ortho_weight
    
    return loss, psnr

def update_learning_rate(optimizer, global_step, lr_factor, args):
    """更新学习率"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * lr_factor

def evaluate_model(model, data_loader, device, split_name='val'):
    """评估模型"""
    model.eval()
    total_loss = 0
    total_psnr = 0
    count = 0
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc=f'Evaluating {split_name}'):
            rays = batch['rays'].to(device)
            rgbs_gt = batch['rgbs'].to(device)
            
            # 分批渲染以节省内存
            chunk_size = 1024
            rgb_pred_list = []
            
            for i in range(0, rays.shape[0], chunk_size):
                rays_chunk = rays[i:i+chunk_size]
                rgb_chunk = model(rays_chunk, white_bg=True, is_train=False)['rgb_map']
                rgb_pred_list.append(rgb_chunk)
            
            rgb_pred = torch.cat(rgb_pred_list, dim=0)
            loss = img2mse(rgb_pred, rgbs_gt)
            psnr = mse2psnr(loss)
            
            total_loss += loss.item()
            total_psnr += psnr.item()
            count += 1
    
    model.train()
    return total_loss / count, total_psnr / count

def save_checkpoint(model, optimizer, global_step, exp_path):
    """保存检查点"""
    ckpt_path = os.path.join(exp_path, 'checkpoints', f'step_{global_step:06d}.pth')
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    
    torch.save({
        'global_step': global_step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, ckpt_path)
    
    print(f'保存检查点至: {ckpt_path}')

def train():
    """主训练函数"""
    # 解析参数
    parser = get_opts()
    args = parser.parse_args()
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'使用设备: {device}')
    
    # 创建实验目录
    exp_path = os.path.join(args.basedir, args.expname)
    os.makedirs(exp_path, exist_ok=True)
    os.makedirs(os.path.join(exp_path, 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(exp_path, 'results'), exist_ok=True)
    
    # 设置Tensorboard
    writer = SummaryWriter(os.path.join(exp_path, 'logs'))
    
    # 保存配置
    with open(os.path.join(exp_path, 'config.txt'), 'w') as f:
        for arg in vars(args):
            f.write(f'{arg}: {getattr(args, arg)}\n')
    
    # 设置数据和模型
    train_loader, val_loader, test_loader, train_dataset = setup_data(args)
    model, optimizer, lr_factor = setup_model(args, device)
    
    print(f'训练数据: {len(train_loader.dataset)} 条射线')
    print(f'验证数据: {len(val_loader.dataset)} 条射线')
    print(f'测试数据: {len(test_loader.dataset)} 条射线')
    
    # 训练循环
    global_step = 0
    start_time = time.time()
    
    # 初始验证
    val_loss, val_psnr = evaluate_model(model, val_loader, device, 'val')
    print(f'初始验证 - Loss: {val_loss:.6f}, PSNR: {val_psnr:.2f}')
    writer.add_scalar('Loss/val', val_loss, global_step)
    writer.add_scalar('PSNR/val', val_psnr, global_step)
    
    for epoch in range(args.n_iters // len(train_loader) + 1):
        for batch_idx, batch in enumerate(train_loader):
            if global_step >= args.n_iters:
                break
                
            rays = batch['rays'].to(device)
            rgbs_gt = batch['rgbs'].to(device)
            
            # 前向传播
            output = model(rays, white_bg=True, is_train=True)
            rgb_pred = output['rgb_map']
            
            # 计算损失
            loss, psnr = compute_loss(rgb_pred, rgbs_gt, args, model, global_step)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 更新学习率
            if global_step < args.lr_decay_iters:
                update_learning_rate(optimizer, global_step, lr_factor, args)
            
            # 记录训练指标
            if global_step % args.progress_refresh_rate == 0:
                writer.add_scalar('Loss/train', loss.item(), global_step)
                writer.add_scalar('PSNR/train', psnr, global_step)
                writer.add_scalar('lr', optimizer.param_groups[0]['lr'], global_step)
                
                elapsed_time = time.time() - start_time
                print(f'Step {global_step:6d} | Loss: {loss.item():.6f} | PSNR: {psnr:.2f} | '
                      f'lr: {optimizer.param_groups[0]["lr"]:.6f} | Time: {elapsed_time:.1f}s')
            
            # 验证
            if global_step > 0 and global_step % (args.n_iters // 10) == 0:
                val_loss, val_psnr = evaluate_model(model, val_loader, device, 'val')
                print(f'验证 Step {global_step} - Loss: {val_loss:.6f}, PSNR: {val_psnr:.2f}')
                writer.add_scalar('Loss/val', val_loss, global_step)
                writer.add_scalar('PSNR/val', val_psnr, global_step)
                
                # 保存检查点
                if global_step % (args.n_iters // 5) == 0:
                    save_checkpoint(model, optimizer, global_step, exp_path)
            
            # 渐进式上采样（简化版本）
            if global_step in args.upsamp_list:
                print(f"上采样步骤 {global_step}")
                # 简化处理，只增加采样数
                if hasattr(model, 'nSamples'):
                    model.nSamples = min(model.nSamples + 16, 128)
            
            global_step += 1
            
        if global_step >= args.n_iters:
            break
    
    # 最终评估
    print("\n==== 最终评估 ====")
    val_loss, val_psnr = evaluate_model(model, val_loader, device, 'val')
    test_loss, test_psnr = evaluate_model(model, test_loader, device, 'test')
    
    print(f'最终验证 - Loss: {val_loss:.6f}, PSNR: {val_psnr:.2f}')
    print(f'最终测试 - Loss: {test_loss:.6f}, PSNR: {test_psnr:.2f}')
    
    # 保存最终模型
    save_checkpoint(model, optimizer, global_step, exp_path)
    
    # 保存结果
    results = {
        'final_val_loss': val_loss,
        'final_val_psnr': val_psnr,
        'final_test_loss': test_loss,
        'final_test_psnr': test_psnr,
        'total_steps': global_step,
        'training_time': time.time() - start_time
    }
    
    import json
    with open(os.path.join(exp_path, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    writer.close()
    print(f"\n训练完成! 结果保存至: {exp_path}")
    print(f"总训练时间: {time.time() - start_time:.1f}秒")

if __name__ == '__main__':
    train() 