import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple, List
from timm.models import create_model
import sys
import os.path as osp


sys.path.append(osp.join(osp.dirname(__file__), '..'))
import model.leaf_model.leaf_model
import model.lesion_model.lesion_model

from .utils import get_leaf_rgb_from_mask


class CombinedSegmentationModel(nn.Module):


    def __init__(self,
                 config,
                 leaf_model_name: str = 'LMLS',
                 lesion_model_name: str = 'TMLS',
                 img_size: int = 480,
                 model_size: str = "base",
                 stage: int = 0):

        super().__init__()

        self.config = config
        self.img_size = img_size
        self.model_size = model_size
        self.stage = stage


        if self.stage == 1 or self.stage == 0:
            print(f"Creating leaf model: {leaf_model_name}")
            self.leaf_model, self.leaf_new_param = create_model(
                leaf_model_name,
                img_size=img_size,
                model_size=model_size,
            )
        else:
            self.leaf_model = None
            self.leaf_new_param = None


        if self.stage == 2 or self.stage == 0:
            print(f"Creating lesion model: {lesion_model_name}")
            self.lesion_model, self.lesion_new_param = create_model(
                lesion_model_name,
                img_size=img_size,
                model_size=model_size,
            )
        else:
            self.lesion_model = None
            self.lesion_new_param = None


        self._set_training_mode()

    def _set_training_mode(self):

        if self.stage == 1:
            if self.leaf_model is not None:
                self.leaf_model.train()
        elif self.stage == 2:
            if self.lesion_model is not None:
                self.lesion_model.train()

            if self.leaf_model is not None and self.config.training_config.get('freeze_leaf_model', True):
                self.leaf_model.eval()
                for param in self.leaf_model.parameters():
                    param.requires_grad = False
        elif self.stage == 0:
            if self.leaf_model is not None:
                self.leaf_model.train()
            if self.lesion_model is not None:
                self.lesion_model.train()

    def freeze_leaf_model(self):

        if self.leaf_model is not None:
            self.leaf_model.eval()
            for param in self.leaf_model.parameters():
                param.requires_grad = False

    def unfreeze_leaf_model(self):

        if self.leaf_model is not None:
            self.leaf_model.train()
            for param in self.leaf_model.parameters():
                param.requires_grad = True

    def forward_stage1(self, x: torch.Tensor, leaf_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:

        if self.leaf_model is None:
            raise ValueError("Leaf model not initialized for stage 1")


        if leaf_mask is not None and leaf_mask.dtype != torch.float32:
            leaf_mask = leaf_mask.float()

        if self.training and leaf_mask is not None:

            leaf_pred, leaf_mask_processed, leaf_loss = self.leaf_model(x, leaf_mask)
            return leaf_pred, leaf_loss
        else:

            leaf_pred = self.leaf_model(x)
            return leaf_pred, None

    def forward_stage2(self, x: torch.Tensor, text: List[str],
                      lesion_mask: Optional[torch.Tensor] = None,
                      leaf_pred: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:

        if self.lesion_model is None:
            raise ValueError("Lesion model not initialized for stage 2")


        if leaf_pred is not None:

            if leaf_pred.dtype == torch.float32 and (leaf_pred.min() < 0 or leaf_pred.max() > 1):
                leaf_pred_prob = torch.sigmoid(leaf_pred)
            else:
                leaf_pred_prob = leaf_pred
            leaf_rgb = get_leaf_rgb_from_mask(x, leaf_pred_prob)
            x = leaf_rgb


        if lesion_mask is not None and lesion_mask.dtype != torch.float32:
            lesion_mask = lesion_mask.float()


        if isinstance(text, str):
            text = [text]
        elif not isinstance(text, list):
            try:
                text = text.tolist()
            except Exception as e:
                raise ValueError(f"Cannot convert text to list: {text}, error: {e}") from e


        batch_size = x.shape[0]
        if len(text) == 1 and batch_size > 1:
            text = text * batch_size
        elif len(text) != batch_size:
            if len(text) < batch_size:
                text = text + [text[0]] * (batch_size - len(text))
            else:
                text = text[:batch_size]

        if self.training and lesion_mask is not None:

            lesion_pred, lesion_mask_processed, lesion_loss = self.lesion_model(x, text, lesion_mask)
            return lesion_pred, lesion_loss
        else:

            lesion_pred = self.lesion_model(x, text)
            return lesion_pred, None

    def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:

        x = batch['query_img']
        results = {}


        if self.stage == 1 or self.stage == 0:
            leaf_mask = batch.get('leaf_mask', None)
            leaf_pred, leaf_loss = self.forward_stage1(x, leaf_mask)

            results['leaf_pred'] = leaf_pred
            if leaf_loss is not None:
                results['leaf_loss'] = leaf_loss
        else:
            leaf_pred = None


        if self.stage == 2 or self.stage == 0:
            text = batch.get('sentence', None)
            lesion_mask = batch.get('lesion_mask', None)


            if text is None:
                text = [""] * x.shape[0]
            elif isinstance(text, str):
                text = [text]
            elif not isinstance(text, list):
                try:
                    text = text.tolist()
                except:
                    text = [str(text)] * x.shape[0]


            if len(text) == 1 and x.shape[0] > 1:
                text = text * x.shape[0]
            elif len(text) != x.shape[0]:
                if len(text) < x.shape[0]:
                    text = text + [text[0]] * (x.shape[0] - len(text))
                else:
                    text = text[:x.shape[0]]


            if self.stage == 2:

                lesion_pred, lesion_loss = self.forward_stage2(x, text, lesion_mask, None)
            else:

                lesion_pred, lesion_loss = self.forward_stage2(x, text, lesion_mask, leaf_pred)

            results['lesion_pred'] = lesion_pred
            if lesion_loss is not None:
                results['lesion_loss'] = lesion_loss
        else:
            lesion_pred = None


        total_loss = 0.0
        loss_count = 0

        if 'leaf_loss' in results:
            total_loss += results['leaf_loss']
            loss_count += 1

        if 'lesion_loss' in results:
            total_loss += results['lesion_loss']
            loss_count += 1

        if loss_count > 0:
            results['total_loss'] = total_loss / loss_count

        return results

    def load_leaf_checkpoint(self, checkpoint_path: str):

        if self.leaf_model is None:
            raise ValueError("Leaf model not initialized")

        print(f"Loading leaf model checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')


        if isinstance(checkpoint, dict):
            if 'model' in checkpoint:
                model_state = checkpoint['model']
            elif 'state_dict' in checkpoint:
                model_state = checkpoint['state_dict']
            elif 'model_state_dict' in checkpoint:
                model_state = checkpoint['model_state_dict']
            else:

                model_state = checkpoint
        else:
            model_state = checkpoint


        has_leaf_prefix = any(k.startswith('leaf_model.') for k in model_state.keys())

        if has_leaf_prefix:

            new_state_dict = {}
            for k, v in model_state.items():
                if k.startswith('leaf_model.'):
                    new_state_dict[k[11:]] = v
                else:
                    new_state_dict[k] = v
            model_state = new_state_dict


        try:
            ret = self.leaf_model.load_state_dict(model_state, strict=False)
            print(f"Leaf model load result: missing={len(ret.missing_keys)}, unexpected={len(ret.unexpected_keys)}")


            if len(ret.missing_keys) > 10 or len(ret.unexpected_keys) > 10:
                print("Warning: the leaf model checkpoint was partially loaded with many mismatched keys.")
                print(f"First 5 missing keys: {ret.missing_keys[:5]}")
                print(f"First 5 unexpected keys: {ret.unexpected_keys[:5]}")
        except Exception as e:
            raise RuntimeError(f"Failed to load leaf model checkpoint from {checkpoint_path}: {e}") from e

    def load_lesion_checkpoint(self, checkpoint_path: str):

        if self.lesion_model is None:
            raise ValueError("Lesion model not initialized")

        print(f"Loading lesion model checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')


        if isinstance(checkpoint, dict):
            if 'model' in checkpoint:
                model_state = checkpoint['model']
            elif 'state_dict' in checkpoint:
                model_state = checkpoint['state_dict']
            elif 'model_state_dict' in checkpoint:
                model_state = checkpoint['model_state_dict']
            else:

                model_state = checkpoint
        else:
            model_state = checkpoint


        has_lesion_prefix = any(k.startswith('lesion_model.') for k in model_state.keys())

        if has_lesion_prefix:

            new_state_dict = {}
            for k, v in model_state.items():
                if k.startswith('lesion_model.'):
                    new_state_dict[k[13:]] = v
                else:
                    new_state_dict[k] = v
            model_state = new_state_dict


        try:
            ret = self.lesion_model.load_state_dict(model_state, strict=False)
            print(f"Lesion model load result: missing={len(ret.missing_keys)}, unexpected={len(ret.unexpected_keys)}")


            if len(ret.missing_keys) > 10 or len(ret.unexpected_keys) > 10:
                print("Warning: the lesion model checkpoint was partially loaded with many mismatched keys.")
                print(f"First 5 missing keys: {ret.missing_keys[:5]}")
                print(f"First 5 unexpected keys: {ret.unexpected_keys[:5]}")
        except Exception as e:
            raise RuntimeError(f"Failed to load lesion model checkpoint from {checkpoint_path}: {e}") from e

    def get_model_for_stage(self, stage: int) -> nn.Module:

        if stage == 1:
            return self.leaf_model
        elif stage == 2:
            return self.lesion_model
        else:
            raise ValueError(f"Invalid stage: {stage}")

    def get_new_params_for_stage(self, stage: int):

        if stage == 1:
            return self.leaf_new_param
        elif stage == 2:
            return self.lesion_new_param
        else:
            raise ValueError(f"Invalid stage: {stage}")

    def set_stage(self, stage: int):

        self.stage = stage
        self._set_training_mode()

    def eval_mode(self):

        if self.leaf_model is not None:
            self.leaf_model.eval()
        if self.lesion_model is not None:
            self.lesion_model.eval()

    def train_mode(self):

        self._set_training_mode()


def create_combined_model(config, stage: int = 0, device: str = 'cuda') -> CombinedSegmentationModel:


    if device == 'cuda' and not torch.cuda.is_available():
        print(f"Warning: CUDA is not available, falling back to CPU")
        device = 'cpu'
    elif device == 'cuda':
        print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    else:
        print(f"Using device: {device}")

    model = CombinedSegmentationModel(
        config=config,
        img_size=config.input_size,
        model_size="base",
        stage=stage
    )

    model.to(device)
    print(f"Model successfully moved to {device}")

    return model
