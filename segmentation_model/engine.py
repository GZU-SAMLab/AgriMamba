import time
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple
from contextlib import suppress
import sys
import os.path as osp

# 添加路径
sys.path.append(osp.join(osp.dirname(__file__), '..'))
import lesion_utils as utils
from .utils import calculate_segmentation_metrics


def train_one_epoch_stage(model: nn.Module, 
                         data_loader, 
                         optimizer, 
                         device: torch.device, 
                         epoch: int,
                         stage: int,
                         config,
                         loss_scaler="none",
                         amp_autocast=suppress,
                         max_norm: float = 0,
                         model_ema: Optional = None,
                         set_training_mode: bool = True) -> Dict[str, Any]:
    """
    分阶段训练一个epoch
    
    Args:
        model: 组合模型
        data_loader: 数据加载器
        optimizer: 优化器
        device: 设备
        epoch: 当前epoch
        stage: 训练阶段 (1: 叶片, 2: 病害)
        config: 配置
        loss_scaler: 损失缩放器
        amp_autocast: 混合精度上下文
        max_norm: 梯度裁剪最大范数
        model_ema: EMA模型
        set_training_mode: 是否设置训练模式
        
    Returns:
        train_stats: 训练统计信息
    """
    model.train(set_training_mode)
    model.set_stage(stage)
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = f'Epoch: [{epoch}] Stage: [{stage}]'
    print_freq = 50
    
    if stage == 1:
        loss_name = 'leaf_loss'
        pred_name = 'leaf_pred'
        mask_name = 'leaf_mask'
        org_gt_name = 'leaf_org_gt'
    elif stage == 2:
        loss_name = 'lesion_loss'
        pred_name = 'lesion_pred'
        mask_name = 'lesion_mask'
        org_gt_name = 'lesion_org_gt'
    else:
        raise ValueError(f"Invalid stage: {stage}")
    
    for batch_idx, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # 将数据移到设备
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(device, non_blocking=True)
                # 确保mask是float类型
                if key in ['leaf_mask', 'lesion_mask'] and batch[key].dtype != torch.float32:
                    batch[key] = batch[key].float()
        
        # 使用更新的autocast API
        if str(device).startswith('cuda') and amp_autocast != suppress:
            with torch.amp.autocast(device_type='cuda'):
                # 前向传播
                results = model(batch)
                
                if loss_name not in results:
                    continue
                    
                loss = results[loss_name]
        else:
            with amp_autocast():
                # 前向传播
                results = model(batch)
                
                if loss_name not in results:
                    continue
                    
                loss = results[loss_name]
        
        loss_value = loss.item()
        
        if not torch.isfinite(loss):
            print(f"Loss is {loss_value}, stopping training")
            continue
        
        # 反向传播
        optimizer.zero_grad()
        
        if loss_scaler != "none":
            # 使用PyTorch的GradScaler
            if hasattr(loss_scaler, 'scale'):
                # PyTorch GradScaler
                scaled_loss = loss_scaler.scale(loss)
                scaled_loss.backward()
                
                if max_norm:
                    loss_scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                
                loss_scaler.step(optimizer)
                loss_scaler.update()
            else:
                # 其他实现
                loss_scaler.scale_loss(loss).backward()
                if max_norm:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                loss_scaler.step(optimizer)
                loss_scaler.update()
        else:
            loss.backward()
            if max_norm:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()
        
        # 更新EMA
        if model_ema is not None:
            model_ema.update(model)
        
        # 记录指标
        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        
        # 计算分割指标
        if pred_name in results and mask_name in batch:
            with torch.no_grad():
                pred = results[pred_name]
                target = batch[mask_name]
                metrics = calculate_segmentation_metrics(pred, target)
                metric_logger.update(iou=metrics['iou'])
                metric_logger.update(miou=metrics['miou'])
                metric_logger.update(dice=metrics['dice'])
                metric_logger.update(precision=metrics['precision'])
                metric_logger.update(recall=metrics['recall'])
    
    # 收集统计信息
    metric_logger.synchronize_between_processes()
    train_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    
    print(f"Averaged stats for stage {stage}:", train_stats)
    return train_stats


def evaluate_stage(model: nn.Module,
                  data_loader,
                  device: torch.device,
                  stage: int,
                  config,
                  amp_autocast=suppress,
                  log_every: int = 50) -> Dict[str, Any]:
    """
    分阶段评估
    
    Args:
        model: 组合模型
        data_loader: 数据加载器
        device: 设备
        stage: 评估阶段
        config: 配置
        amp_autocast: 混合精度上下文
        log_every: 日志频率
        
    Returns:
        eval_stats: 评估统计信息
    """
    model.eval()
    model.set_stage(stage)
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = f'Evaluation Stage: [{stage}]'
    
    if stage == 1:
        pred_name = 'leaf_pred'
        mask_name = 'leaf_mask'
        org_gt_name = 'leaf_org_gt'
    elif stage == 2:
        pred_name = 'lesion_pred'
        mask_name = 'lesion_mask'
        org_gt_name = 'lesion_org_gt'
    else:
        raise ValueError(f"Invalid stage: {stage}")
    
    all_ious = []
    all_mious = []
    all_dices = []
    all_precisions = []
    all_recalls = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(metric_logger.log_every(data_loader, log_every, header)):
            # 将数据移到设备
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device, non_blocking=True)
                    # 确保mask是float类型
                    if key in ['leaf_mask', 'lesion_mask'] and batch[key].dtype != torch.float32:
                        batch[key] = batch[key].float()
            
            # 确保文本数据格式正确
            if 'sentence' in batch and stage == 2:
                if batch['sentence'] is None:
                    batch['sentence'] = [""] * batch['query_img'].shape[0]
                elif isinstance(batch['sentence'], str):
                    batch['sentence'] = [batch['sentence']]
                elif not isinstance(batch['sentence'], list):
                    try:
                        batch['sentence'] = batch['sentence'].tolist()
                    except:
                        batch['sentence'] = [str(batch['sentence'])] * batch['query_img'].shape[0]
            
            # 使用更新的autocast API
            if str(device).startswith('cuda') and amp_autocast != suppress:
                with torch.amp.autocast(device_type='cuda'):
                    # 前向传播
                    results = model(batch)
                    # 如果是单值返回（如leaf_pred），转换为字典格式
                    if pred_name not in results:
                        # 如果是单值返回（如leaf_pred），转换为字典格式
                        if not isinstance(results, dict):
                            if stage == 1:
                                results = {pred_name: results}
                            elif stage == 2:
                                results = {pred_name: results}
                        else:
                            continue
            else:
                with amp_autocast():
                    # 前向传播
                    results = model(batch)
                    # 如果是单值返回（如leaf_pred），转换为字典格式
                    if pred_name not in results:
                        # 如果是单值返回（如leaf_pred），转换为字典格式
                        if not isinstance(results, dict):
                            if stage == 1:
                                results = {pred_name: results}
                            elif stage == 2:
                                results = {pred_name: results}
                        else:
                            continue
            
            if pred_name not in results:
                continue
                
            pred = results[pred_name]
            
            # 应用sigmoid获取概率
            if isinstance(pred, torch.Tensor):
                pred = torch.sigmoid(pred)
            
            # 计算指标
            if mask_name in batch:
                target = batch[mask_name]
                metrics = calculate_segmentation_metrics(pred, target)
                
                all_ious.append(metrics['iou'])
                all_mious.append(metrics['miou'])
                all_dices.append(metrics['dice'])
                all_precisions.append(metrics['precision'])
                all_recalls.append(metrics['recall'])
                
                metric_logger.update(iou=metrics['iou'])
                metric_logger.update(miou=metrics['miou'])
                metric_logger.update(dice=metrics['dice'])
                metric_logger.update(precision=metrics['precision'])
                metric_logger.update(recall=metrics['recall'])
    
    # 收集统计信息
    metric_logger.synchronize_between_processes()
    eval_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    
    if all_ious:
        eval_stats['mean_iou'] = sum(all_ious) / len(all_ious)
        eval_stats['mean_miou'] = sum(all_mious) / len(all_mious)
        eval_stats['mean_dice'] = sum(all_dices) / len(all_dices)
        eval_stats['mean_precision'] = sum(all_precisions) / len(all_precisions)
        eval_stats['mean_recall'] = sum(all_recalls) / len(all_recalls)
    
    print(f"Evaluation stats for stage {stage}:", eval_stats)
    return eval_stats


def evaluate_combined(model: nn.Module,
                     data_loader,
                     device: torch.device,
                     config,
                     amp_autocast=suppress,
                     log_every: int = 50,
                     save_predictions: bool = False) -> Dict[str, Any]:
    """
    完整的两阶段评估
    
    Args:
        model: 组合模型
        data_loader: 数据加载器
        device: 设备
        config: 配置
        amp_autocast: 混合精度上下文
        log_every: 日志频率
        save_predictions: 是否保存预测结果
        
    Returns:
        eval_stats: 完整评估统计信息
    """
    model.eval()
    model.set_stage(0)  # 两阶段模式
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Combined Evaluation'
    
    # 各种指标收集
    leaf_ious = []
    lesion_ious = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(metric_logger.log_every(data_loader, log_every, header)):
            # 将数据移到设备
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device, non_blocking=True)
                    # 确保mask是float类型
                    if key in ['leaf_mask', 'lesion_mask'] and batch[key].dtype != torch.float32:
                        batch[key] = batch[key].float()
            
            # 确保文本数据格式正确
            if 'sentence' in batch:
                if batch['sentence'] is None:
                    batch['sentence'] = [""] * batch['query_img'].shape[0]
                elif isinstance(batch['sentence'], str):
                    batch['sentence'] = [batch['sentence']]
                elif not isinstance(batch['sentence'], list):
                    try:
                        batch['sentence'] = batch['sentence'].tolist()
                    except:
                        batch['sentence'] = [str(batch['sentence'])] * batch['query_img'].shape[0]
            
            # 使用更新的autocast API
            if str(device).startswith('cuda') and amp_autocast != suppress:
                with torch.amp.autocast(device_type='cuda'):
                    # 前向传播
                    results = model(batch)
            else:
                with amp_autocast():
                    # 前向传播
                    results = model(batch)
            
            # 评估叶片分割
            if 'leaf_pred' in results and 'leaf_mask' in batch:
                # 应用sigmoid获取概率
                if isinstance(results['leaf_pred'], torch.Tensor):
                    leaf_pred = torch.sigmoid(results['leaf_pred'])
                else:
                    leaf_pred = results['leaf_pred']
                
                leaf_metrics = calculate_segmentation_metrics(leaf_pred, batch['leaf_mask'])
                leaf_ious.append(leaf_metrics['iou'])
                metric_logger.update(leaf_iou=leaf_metrics['iou'])
            
            # 评估病害分割
            if 'lesion_pred' in results and 'lesion_mask' in batch:
                # 应用sigmoid获取概率
                if isinstance(results['lesion_pred'], torch.Tensor):
                    lesion_pred = torch.sigmoid(results['lesion_pred'])
                else:
                    lesion_pred = results['lesion_pred']
                
                lesion_metrics = calculate_segmentation_metrics(lesion_pred, batch['lesion_mask'])
                lesion_ious.append(lesion_metrics['iou'])
                metric_logger.update(lesion_iou=lesion_metrics['iou'])
            
    # 收集统计信息
    metric_logger.synchronize_between_processes()
    eval_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    
    # 添加整体统计
    if leaf_ious:
        eval_stats['mean_leaf_iou'] = sum(leaf_ious) / len(leaf_ious)
    if lesion_ious:
        eval_stats['mean_lesion_iou'] = sum(lesion_ious) / len(lesion_ious)
    
    print("Combined evaluation stats:", eval_stats)
    return eval_stats


def train_stage_pipeline(model: nn.Module,
                        train_loader,
                        val_loader, 
                        config,
                        stage: int,
                        optimizer,
                        lr_scheduler,
                        device: torch.device,
                        start_epoch: int = 0,
                        amp_autocast=suppress,
                        loss_scaler="none",
                        output_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    完整的单阶段训练流水线
    
    Args:
        model: 组合模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        config: 配置
        stage: 训练阶段
        optimizer: 优化器
        lr_scheduler: 学习率调度器
        device: 设备
        start_epoch: 起始epoch
        amp_autocast: 混合精度上下文
        loss_scaler: 损失缩放器
        output_dir: 输出目录
        
    Returns:
        training_stats: 训练统计信息
    """
    import json
    
    stage_config = config.get_stage_config(stage)
    epochs = stage_config['epochs']
    
    print(f"Starting Stage {stage} training for {epochs} epochs")
    print(f"Stage {stage} config: {stage_config}")
    
    best_iou = 0.0
    training_stats = {'train_stats': [], 'val_stats': []}
    
    # 计算模型参数数量
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # 创建训练日志文件
    if output_dir and utils.is_main_process():
        log_file = osp.join(output_dir, f'stage{stage}_training_log.jsonl')
        
    for epoch in range(start_epoch, epochs):
        print(f"\n=== Stage {stage} Epoch {epoch}/{epochs-1} ===")
        
        # 训练
        train_stats = train_one_epoch_stage(
            model=model,
            data_loader=train_loader,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            stage=stage,
            config=config,
            loss_scaler=loss_scaler,
            amp_autocast=amp_autocast,
            max_norm=stage_config.get('clip_grad', None),
            model_ema=None,  # 暂时不使用EMA
            set_training_mode=True
        )
        
        # 更新学习率调度器
        if lr_scheduler is not None:
            # 兼容不同类型的学习率调度器
            try:
                # 尝试使用epoch参数
                lr_scheduler.step(epoch)
            except TypeError:
                # 如果调度器不接受epoch参数，则不传参数
                lr_scheduler.step()
            except Exception as e:
                print(f"警告: 学习率调度器更新失败: {e}")
                
        # 验证
        val_stats = evaluate_stage(
            model=model,
            data_loader=val_loader,
            device=device,
            stage=stage,
            config=config,
            amp_autocast=amp_autocast,
            log_every=50
        )
        
        # 记录统计信息
        training_stats['train_stats'].append(train_stats)
        training_stats['val_stats'].append(val_stats)
        
        # 保存每个epoch的日志（一行一个JSON）
        if output_dir and utils.is_main_process():
            current_lr = optimizer.param_groups[0]['lr']
            epoch_log = {
                f"stage{stage}_train_lr": current_lr,
                f"stage{stage}_train_loss": train_stats.get('loss', 0.0),
                f"stage{stage}_train_iou": train_stats.get('iou', 0.0),
                f"stage{stage}_train_dice": train_stats.get('dice', 0.0),
                f"stage{stage}_train_precision": train_stats.get('precision', 0.0),
                f"stage{stage}_train_recall": train_stats.get('recall', 0.0),
                f"stage{stage}_val_iou": val_stats.get('iou', 0.0),
                f"stage{stage}_val_dice": val_stats.get('dice', 0.0),
                f"stage{stage}_val_precision": val_stats.get('precision', 0.0),
                f"stage{stage}_val_recall": val_stats.get('recall', 0.0),
                "epoch": epoch,
                "stage": stage,
                "n_parameters": n_parameters
            }
            
            # 追加写入日志文件
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(epoch_log, ensure_ascii=False) + '\n')
        
        # 保存最佳模型
        current_iou = val_stats.get('iou', 0.0)
        if current_iou > best_iou:
            best_iou = current_iou
            if output_dir:
                checkpoint_path = osp.join(output_dir, f'best_stage{stage}_checkpoint.pth')
                utils.save_on_master({
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'stage': stage,
                    'best_iou': best_iou,
                    'config': config.to_dict(),
                }, checkpoint_path)
                print(f"Saved best stage {stage} model with IoU: {best_iou:.3f}")
        
        print(f"Stage {stage} Epoch {epoch} - Train Loss: {train_stats.get('loss', 0):.4f}, "
              f"Val IoU: {current_iou:.3f}, Best IoU: {best_iou:.3f}")
    
    print(f"Stage {stage} training completed. Best IoU: {best_iou:.3f}")
    return training_stats 
