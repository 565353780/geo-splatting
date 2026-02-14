"""
COLMAP 数据集的 GeoSplatting 训练并导出。

与 rfstudio.engine.train.TrainTask.run() 逻辑一致：直接运行，无 CLI。
需在 path 下存在 sparse/0/cameras.bin, images.bin, points3D.bin 与 images/。
"""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch

from rfstudio.data import MultiViewDataset
from rfstudio.engine.experiment import Experiment
from rfstudio.engine.task import _setup
from rfstudio.engine.train import TrainTask
from rfstudio.model import GeoSplatter
from rfstudio.trainer import GeoSplatTrainer
from rfstudio.ui import console
from rfstudio.utils.pretty import P

# -----------------------------------------------------------------------------
# 配置（按需修改）
# -----------------------------------------------------------------------------
data_id = 'haizei_1_v4'
home = Path.home()
colmap_data_folder_path = home / 'chLi' / 'Dataset' / 'GS' / data_id / 'colmap_normalized'
save_result_folder_path = home / 'chLi' / 'Dataset' / 'GS' / data_id / 'geosplatting'

cuda = 0
seed = 1
num_steps = 500
batch_size = 8
num_steps_per_val = 0
hold_after_train = False
full_test_after_train = False
mixed_precision = False

# -----------------------------------------------------------------------------
# 构建与引擎一致的 TrainTask
# -----------------------------------------------------------------------------
experiment = Experiment(
    name='geosplat-colmap',
    timestamp='test',
    output_dir=Path(save_result_folder_path),
)
dataset = MultiViewDataset(
    path=Path(colmap_data_folder_path),
)
model = GeoSplatter(
    background_color='white',
    resolution=128,
    scale=0.95,
    initial_guess='specular',
)
trainer = GeoSplatTrainer(
    num_steps=num_steps,
    batch_size=batch_size,
    num_steps_per_val=num_steps_per_val,
    hold_after_train=hold_after_train,
    full_test_after_train=full_test_after_train,
    mixed_precision=mixed_precision,
)

task = TrainTask(
    dataset=dataset,
    model=model,
    experiment=experiment,
    trainer=trainer,
    cuda=cuda,
    seed=seed,
)


def main() -> None:
    # 与 engine.task._entrypoint 一致的设备与随机种子
    torch.multiprocessing.set_start_method('spawn', force=True)
    torch.set_float32_matmul_precision('high')
    torch.set_printoptions(precision=3, threshold=16)
    device = task.device
    if device.type != 'cpu':
        torch.cuda.set_device(device)
    if task.seed is not None:
        random.seed(task.seed)
        np.random.seed(task.seed)
        torch.manual_seed(task.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(task.seed)

    _setup(task)
    task.run()

    # 训练结束后导出模型
    export_path = Path(save_result_folder_path) / 'geosplat' / 'result.pt'
    export_path.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        assert isinstance(task.model, GeoSplatter)
        task.model.export_model(export_path)
    console.print(P @ f'Exported to {export_path}')


if __name__ == '__main__':
    main()
