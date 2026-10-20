#!/usr/bin/env python3
"""models.py —— 模型工厂(唯一权威来源)

模型类本体仍在各自文件里(dcnn.py / transformer.py / conformer.py, README 对应
"不同模型不同文件"的约定); 这里只做 re-export + 统一构造入口 build_model,
消灭原先 data_aug.py 与 plot_confusion_matrix.py 的两份工厂
(后者不传 conformer 超参, 对调参 run 的 checkpoint 是形状错误级 bug)。
"""
from utils.cli import CONFORMER_DEFAULTS
from conformer import ConformerIMU
from dcnn import DCNN
from transformer import TransformerIMU, TransformerIMU2

# CLI 名 -> (类, conformer 超参默认; None=无超参模型)
_REGISTRY = {
    "dcnn": (DCNN, None),
    "transformer": (TransformerIMU, None),
    "transformer2": (TransformerIMU2, None),
    "conformer": (ConformerIMU, CONFORMER_DEFAULTS),
}

# checkpoint cfg 键(data_aug.py 训练时写入) -> ConformerIMU 构造参数名
_CFG_KEYMAP = {"d_model": "d_model", "nhead": "nhead", "layers": "num_layers",
               "dim_ff": "dim_ff", "patch": "patch", "kernel": "kernel",
               "dropout": "dropout", "gate": "gate", "conv_norm": "conv_norm",
               "attn": "attn_mode"}

_ALIAS = {"DCNN": "dcnn", "TransformerIMU": "transformer",
          "TransformerIMU2": "transformer2", "ConformerIMU": "conformer"}


def build_model(name: str, input_size: int, num_classes: int,
                cfg: dict = None, **hp):
    """统一模型构造。

    name: CLI 名(dcnn/transformer/transformer2/conformer)或类名。
    cfg : checkpoint 里的 config dict(键名见 _CFG_KEYMAP)。
    hp  : 显式超参(优先级最高) > cfg > CONFORMER_DEFAULTS; 键名用 CLI 名
          (d_model/nhead/layers/dim_ff/patch/kernel/dropout/gate/conv_norm/attn)。
    """
    name = _ALIAS.get(name, name)
    if name not in _REGISTRY:
        raise ValueError(f"未知模型: {name}")
    cls, defaults = _REGISTRY[name]
    if defaults is None:
        return cls(input_size=input_size, num_classes=num_classes)
    src = cfg or {}
    kw = {}
    for cfg_key, ctor_key in _CFG_KEYMAP.items():
        if cfg_key in hp and hp[cfg_key] is not None:
            kw[ctor_key] = hp[cfg_key]
        elif cfg_key in src:
            kw[ctor_key] = src[cfg_key]
        else:
            kw[ctor_key] = defaults[ctor_key]
    return ConformerIMU(input_size=input_size, num_classes=num_classes, **kw)


__all__ = ["DCNN", "TransformerIMU", "TransformerIMU2", "ConformerIMU",
           "CONFORMER_DEFAULTS", "build_model"]
