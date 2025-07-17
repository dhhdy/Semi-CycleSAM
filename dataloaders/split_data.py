import os
import random
import glob


def split_h5_dataset(data_dir, train_ratio=0.7, val_ratio=0.1, test_ratio=0.2, shuffle=True, seed=42):
    """
    随机划分HDF5数据集为训练集、验证集和测试集，并保存文件名到对应的.list文件。

    参数:
        data_dir (str): 存放.h5文件的目录路径
        train_ratio (float): 训练集比例 (默认 0.7)
        val_ratio (float): 验证集比例 (默认 0.2)
        test_ratio (float): 测试集比例 (默认 0.1)
        shuffle (bool): 是否打乱文件顺序 (默认 True)
        seed (int): 随机种子 (默认 42)
    """
    # 检查比例总和是否为1
    # assert (train_ratio + val_ratio + test_ratio) == 1.0, "比例总和必须等于1!"

    # 获取所有.h5文件
    h5_files = glob.glob(os.path.join(data_dir, "*"))
    if not h5_files:
        raise FileNotFoundError(f"在目录 {data_dir} 中未找到.h5文件!")

    # 打乱文件顺序（可选）
    if shuffle:
        random.seed(seed)
        random.shuffle(h5_files)

    # 计算划分点
    num_files = len(h5_files)
    num_train = int(num_files * train_ratio)
    num_val = int(num_files * val_ratio)
    # 测试集数量 = 剩余部分（避免因四舍五入导致总数不一致）

    # 划分数据集
    train_files = h5_files[:num_train]
    val_files = h5_files[num_train:num_train + num_val]
    test_files = h5_files[num_train + num_val:]

    # 保存文件名到.list文件
    def save_to_list(file_list, output_file):
        with open(output_file, "w") as f:
            for file in file_list:
                f.write(f"{os.path.basename(file)}\n")  # 只保存文件名（不含路径）

    save_to_list(train_files, "train.list")
    save_to_list(val_files, "validation.list")
    save_to_list(test_files, "test.list")

    print(f"划分完成！\n"
          f"训练集: {len(train_files)} 个文件 (保存到 train.list)\n"
          f"验证集: {len(val_files)} 个文件 (保存到 validation.list)\n"
          f"测试集: {len(test_files)} 个文件 (保存到 test.list)")

import os
import shutil
from pathlib import Path

def reorganize_files(src_dir):
    """
    将文件名格式为 {prefix}.{suffix}.h5 的文件重组为:
    {prefix}/suffix.h5

    例如:
    - 输入: 10888638_0000.mri_norm2.h5
    - 输出: 10888638_0000/mri_norm2.h5
    """
    src_dir = Path(src_dir)
    if not src_dir.exists():
        raise FileNotFoundError(f"目录不存在: {src_dir}")

    # 遍历所有.h5文件
    for h5_file in src_dir.glob("*.h5"):
        if h5_file.is_file():
            # 分割文件名 (e.g., "10888638_0000.mri_norm2.h5" -> ["10888638_0000", "mri_norm2"])
            parts = h5_file.stem.split(".")
            if len(parts) != 2:
                print(f"跳过不符合命名规则的文件: {h5_file.name}")
                continue

            prefix, suffix = parts
            target_dir = src_dir / prefix
            target_file = target_dir / f"{suffix}.h5"

            # 创建目标文件夹
            target_dir.mkdir(exist_ok=True)

            # 移动并重命名文件
            shutil.move(str(h5_file), str(target_file))
            print(f"已重组: {h5_file.name} -> {target_file}")

# if __name__ == "__main__":
#     source_directory = "/data1/data/lung/I_IIIA_LUAD/GD_preprocess/imagesTr/"  # 替换为你的文件目录
#     reorganize_files(source_directory)

# 使用示例
if __name__ == "__main__":
    data_directory = "/data1/data/GD_master/h5"  # 替换为你的.h5文件目录
    split_h5_dataset(data_directory)