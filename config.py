import configargparse

def get_opts():
    parser = configargparse.ArgumentParser()
    
    # 数据集参数
    parser.add_argument('--data_dir', type=str, default='./data',
                       help='数据集目录路径')
    parser.add_argument('--img_wh', nargs=2, type=int, default=[800, 800],
                       help='图像宽度和高度')
    parser.add_argument('--spheric_poses', default=False, action="store_true",
                       help='是否使用球面姿态')
    parser.add_argument('--use_disp', default=False, action="store_true",
                       help='是否使用视差而不是深度')
    
    # 模型参数
    parser.add_argument('--model_name', type=str, default='TensoRF',
                       choices=['TensoRF'], help='模型名称')
    parser.add_argument('--n_voxel_init', type=int, default=128**3,
                       help='初始体素分辨率')
    parser.add_argument('--n_voxel_final', type=int, default=300**3,
                       help='最终体素分辨率')
    parser.add_argument('--upsamp_list', type=int, nargs='+', default=[2000,3000,4000,5500,7000],
                       help='上采样步数列表')
    parser.add_argument('--update_AlphaMask_list', type=int, nargs='+', default=[2000,4000],
                       help='更新alpha mask的步数列表')
    
    parser.add_argument('--N_vis', type=int, default=5,
                       help='可视化频率')
    parser.add_argument('--vis_every', type=int, default=10000,
                       help='每多少步可视化一次')
    
    parser.add_argument('--render_train', type=int, default=1,
                       help='是否渲染训练图像')
    parser.add_argument('--render_test', type=int, default=1,
                       help='是否渲染测试图像')
    parser.add_argument('--render_path', type=int, default=1,
                       help='是否渲染路径')
    
    parser.add_argument('--n_lamb_sigma', type=int, nargs='+', default=[16,16,16],
                       help='sigma张量分解的rank')
    parser.add_argument('--n_lamb_sh', type=int, nargs='+', default=[48,48,48],
                       help='sh张量分解的rank')
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=4096,
                       help='训练批大小')
    parser.add_argument('--lr_init', type=float, default=0.02,
                       help='初始学习率')
    parser.add_argument('--lr_basis', type=float, default=1e-3,
                       help='基函数学习率')
    parser.add_argument('--lr_decay_iters', type=int, default=-1,
                       help='学习率衰减步数')
    parser.add_argument('--lr_decay_target_ratio', type=float, default=0.1,
                       help='学习率衰减目标比率')
    parser.add_argument('--lr_upsample_reset', type=int, default=1,
                       help='上采样时是否重置学习率')
    
    parser.add_argument('--n_iters', type=int, default=30000,
                       help='训练总步数')
    parser.add_argument('--progress_refresh_rate', type=int, default=10,
                       help='进度条刷新频率')
    
    # 损失函数参数
    parser.add_argument('--L1_weight_inital', type=float, default=0.0,
                       help='L1损失初始权重')
    parser.add_argument('--L1_weight_rest', type=float, default=0.0,
                       help='L1损失其余权重')
    parser.add_argument('--Ortho_weight', type=float, default=0.0,
                       help='正交损失权重')
    parser.add_argument('--TV_weight_density', type=float, default=0.0,
                       help='密度TV损失权重')
    parser.add_argument('--TV_weight_app', type=float, default=0.0,
                       help='外观TV损失权重')
    
    # 渲染参数
    parser.add_argument('--rm_weight_mask_thre', type=float, default=0.0001,
                       help='移除权重mask阈值')
    parser.add_argument('--alpha_mask_thre', type=float, default=0.0001,
                       help='alpha mask阈值')
    parser.add_argument('--distance_scale', type=float, default=25,
                       help='距离缩放')
    parser.add_argument('--density_shift', type=float, default=-10,
                       help='密度偏移')
    
    parser.add_argument('--shadingMode', type=str, default="MLP_PE",
                       help='着色模式')
    parser.add_argument('--pos_pe', type=int, default=6,
                       help='位置编码层数')
    parser.add_argument('--view_pe', type=int, default=6,
                       help='视角编码层数')
    parser.add_argument('--fea_pe', type=int, default=6,
                       help='特征编码层数')
    parser.add_argument('--featureC', type=int, default=128,
                       help='特征通道数')
    
    # 系统参数
    parser.add_argument('--num_workers', type=int, default=4,
                       help='数据加载器工作进程数')
    parser.add_argument('--ckpt', type=str, default=None,
                       help='检查点路径')
    parser.add_argument('--render_only', type=int, default=0,
                       help='是否只进行渲染')
    parser.add_argument('--basedir', type=str, default='./logs',
                       help='实验日志目录')
    parser.add_argument('--expname', type=str, default='dvgo_hotdog',
                       help='实验名称')
    
    parser.add_argument('--perturb', type=float, default=1.,
                       help='在采样点添加随机扰动')
    parser.add_argument('--accumulate_decay', type=float, default=0.998,
                       help='accumulate衰减因子')
    parser.add_argument('--fea2denseAct', type=str, default='softplus',
                       help='特征到密度的激活函数')
    parser.add_argument('--ndc_ray', type=int, default=0,
                       help='是否使用NDC光线')
    parser.add_argument('--nSamples', type=int, default=1e6,
                       help='采样点数量')
    parser.add_argument('--step_ratio', type=float, default=0.5,
                       help='步长比例')
    
    
    return parser 