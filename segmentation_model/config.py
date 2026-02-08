import argparse
from typing import Dict, Any

class SegmentationConfig:
    """
    Segmentation Model Configuration
    管理两阶段模型的所有超参数
    """
    
    def __init__(self):
        # 通用参数
        self.batch_size = 8
        self.epochs = 50
        self.input_size = 480
        self.device = 'cuda'
        self.num_workers = 8
        self.pin_mem = True
        
        # 数据集参数
        self.data_path = './dataset'
        self.data_set = 'dataset4380_split'
        
        # 第一阶段 (Leaf Model) 参数
        self.leaf_model_config = {
            'model_name': 'LMLS',
            'lr': 5e-5,
            'lr_backbone': 2.5e-5,
            'weight_decay': 1e-4,
            'epochs': 25,  # 第一阶段训练轮数
            'opt': 'adamw',
            'sched': 'cosine',
            'warmup_epochs': 0,
            'min_lr': 1e-6,
            'warmup_lr': 1e-6,
            'drop': 0.0,
            'drop_path': 0.1,
            'clip_grad': None,
            'model_size': 'base'
        }
        
        # 第二阶段 (Lesion Model) 参数  
        self.lesion_model_config = {
            'model_name': 'TMLS',
            'lr': 3e-5,
            'lr_backbone': 1.5e-5,
            'lr_decoder': 3e-5,
            'lr_vssm': 1.5e-5,
            'weight_decay': 1e-4,
            'epochs': 25,  # 第二阶段训练轮数
            'opt': 'adamw',
            'sched': 'cosine',
            'warmup_epochs': 0,
            'min_lr': 1e-6,
            'warmup_lr': 1e-6,
            'drop': 0.0,
            'drop_path': 0.1,
            'clip_grad': None,
            'model_size': 'base'
        }
        
        # 训练策略参数
        self.training_config = {
            'stage1_pretrain_path': './pretrain',
            'stage2_pretrain_path': './pretrain', 
            'freeze_leaf_model': True,  # 第二阶段是否冻结叶片模型
            'if_amp': True,  # 混合精度训练
            'model_ema': False,
            'model_ema_decay': 0.9999,
            'distributed': False
        }
        
    @classmethod
    def from_args(cls, args: argparse.Namespace):
        """从命令行参数创建配置"""
        config = cls()
        
        # 更新通用参数
        if hasattr(args, 'batch_size'):
            config.batch_size = args.batch_size
        if hasattr(args, 'epochs'):
            config.epochs = args.epochs
        if hasattr(args, 'input_size'):
            config.input_size = args.input_size
        if hasattr(args, 'data_path'):
            config.data_path = args.data_path
        if hasattr(args, 'data_set'):
            config.data_set = args.data_set
            
        # 更新第一阶段参数
        if hasattr(args, 'leaf_lr'):
            config.leaf_model_config['lr'] = args.leaf_lr
        if hasattr(args, 'leaf_epochs'):
            config.leaf_model_config['epochs'] = args.leaf_epochs
            
        # 更新第二阶段参数
        if hasattr(args, 'lesion_lr'):
            config.lesion_model_config['lr'] = args.lesion_lr  
        if hasattr(args, 'lesion_epochs'):
            config.lesion_model_config['epochs'] = args.lesion_epochs
            
        return config
    
    def get_stage_config(self, stage: int) -> Dict[str, Any]:
        """获取指定阶段的配置"""
        if stage == 1:
            return self.leaf_model_config
        elif stage == 2:
            return self.lesion_model_config
        else:
            raise ValueError(f"Invalid stage: {stage}. Must be 1 or 2.")
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'input_size': self.input_size,
            'data_path': self.data_path,
            'data_set': self.data_set,
            'leaf_model_config': self.leaf_model_config,
            'lesion_model_config': self.lesion_model_config,
            'training_config': self.training_config
        }


def get_args_parser():
    """获取命令行参数解析器"""
    parser = argparse.ArgumentParser('Segmentation Model training and evaluation', add_help=False)
    
    # 基本参数
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--input-size', default=480, type=int, help='images input size')
    parser.add_argument('--lr', type=float, default=5e-5, help='base learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='weight decay')
    
    # 数据集参数
    parser.add_argument('--data-path', default='./dataset', type=str, help='dataset path')
    parser.add_argument('--data-set', default='dataset4380_split', type=str, help='数据集名称')
    
    # 第一阶段参数
    parser.add_argument('--leaf-lr', type=float, default=5e-5, help='learning rate for leaf model')
    parser.add_argument('--leaf-epochs', type=int, default=25, help='epochs for stage 1')
    parser.add_argument('--leaf-weight-decay', type=float, default=1e-4, help='weight decay for leaf model')
    parser.add_argument('--lr-backbone', type=float, default=2.5e-5, help='backbone learning rate')
    
    # 第二阶段参数  
    parser.add_argument('--lesion-lr', type=float, default=3e-5, help='learning rate for lesion model')
    parser.add_argument('--lesion-epochs', type=int, default=25, help='epochs for stage 2')
    parser.add_argument('--lesion-weight-decay', type=float, default=1e-4, help='weight decay for lesion model')
    parser.add_argument('--lr-decoder', type=float, default=3e-5, help='decoder learning rate')
    parser.add_argument('--lr-vssm', type=float, default=2.5e-5, help='vssm learning rate')
    
    # 训练策略
    parser.add_argument('--stage', type=int, default=0, choices=[0, 1, 2], 
                       help='training stage: 0=both, 1=leaf only, 2=lesion only')
    parser.add_argument('--freeze-leaf', action='store_true', default=True,
                       help='freeze leaf model in stage 2')
    parser.add_argument('--stage1-results', type=str, default='', 
                       help='path to stage1 leaf segmentation results for stage2 training')
    parser.add_argument('--generate-stage1-results', action='store_true',
                       help='generate stage1 results after stage1 training')
    
    # 模型参数
    parser.add_argument('--pretrain-path', default='./pretrain', type=str)
    parser.add_argument('--output_dir', default='./output', help='path where to save')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    
    # 评估模式专用参数
    parser.add_argument('--leaf-checkpoint', type=str, default='', 
                       help='path to leaf model checkpoint for evaluation')
    parser.add_argument('--lesion-checkpoint', type=str, default='', 
                       help='path to lesion model checkpoint for evaluation')
    parser.add_argument('--eval-stage', type=int, default=0, choices=[0],
                       help='evaluation stage: 0=combined full pipeline (only supported mode)')
    parser.add_argument('--save-predictions', action='store_true', 
                       help='save prediction results during evaluation')
    parser.add_argument('--save-visualization', action='store_true',
                       help='save visualization results during evaluation')
    
    # 系统参数
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--pin-mem', action='store_true', help='Pin CPU memory in DataLoader')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)
    
    # AMP参数
    parser.add_argument('--if_amp', action='store_true', default=True)
    parser.add_argument('--no_amp', action='store_false', dest='if_amp')
    
    # 分布式训练
    parser.add_argument('--distributed', action='store_true', default=False)
    parser.add_argument('--world_size', default=1, type=int)
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--local-rank', default=0, type=int)
    
    # 优化器和调度器参数
    parser.add_argument('--opt', default='adamw', type=str, help='Optimizer')
    parser.add_argument('--sched', default='cosine', type=str, help='LR scheduler')
    parser.add_argument('--warmup-epochs', type=int, default=0, help='warmup epochs')
    parser.add_argument('--min-lr', type=float, default=1e-6, help='minimum learning rate')
    parser.add_argument('--warmup-lr', type=float, default=1e-6, help='warmup learning rate')
    parser.add_argument('--decay-epochs', type=float, default=30, help='epoch interval to decay LR')
    parser.add_argument('--cooldown-epochs', type=int, default=10, help='cooldown epochs')
    parser.add_argument('--patience-epochs', type=int, default=10, help='patience epochs')
    parser.add_argument('--decay-rate', type=float, default=0.1, help='LR decay rate')
    parser.add_argument('--drop', type=float, default=0.0, help='Dropout rate')
    parser.add_argument('--drop-path', type=float, default=0.1, help='Drop path rate')
    
    return parser 
