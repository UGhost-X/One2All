import os
import numpy as np
from PIL import Image
from pathlib import Path
from anomalib.data import Folder
from anomalib.models import Padim
from anomalib.engine import Engine

def prepare_data(base_path: Path):
    """创建一些模拟数据用于演示"""
    print("正在准备模拟数据...")
    train_good = base_path / "train" / "good"
    test_good = base_path / "test" / "good"
    test_bad = base_path / "test" / "bad"
    
    for path in [train_good, test_good, test_bad]:
        path.mkdir(parents=True, exist_ok=True)
        
    # 生成正常的训练图片 (纯随机噪声或某种模式)
    for i in range(10):
        img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        Image.fromarray(img).save(train_good / f"{i}.png")
        
    # 生成正常的测试图片
    for i in range(2):
        img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        Image.fromarray(img).save(test_good / f"{i}.png")
        
    # 生成带“缺陷”的测试图片 (在中间画个黑块)
    for i in range(2):
        img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        img[100:150, 100:150, :] = 0  # 模拟缺陷
        Image.fromarray(img).save(test_bad / f"{i}.png")
    print(f"数据准备完成: {base_path}")

def run_demo():
    # --- 超参数配置区 ---
    CONFIG = {
        "model": {
            "backbone": "resnet18",         # PaDiM 默认常用 resnet18，兼顾速度与效果
            "layers": ["layer1", "layer2", "layer3"], # 融合多层特征
        },
        "dataset": {
            "image_size": (256, 256),
            "train_batch_size": 4,
            "eval_batch_size": 4,
        },
        "engine": {
            "max_epochs": 1,                # PaDiM 只需要 1 次遍历统计特征
            "default_root_dir": "./results",
        }
    }
    # ------------------

    # 1. 准备数据
    data_root = Path("./datasets/dummy")
    prepare_data(data_root)
    
    # 2. 配置数据模块
    datamodule = Folder(
        name="dummy",
        root=data_root,
        normal_dir="train/good",
        abnormal_dir="test/bad",
        normal_test_dir="test/good",
        train_batch_size=CONFIG["dataset"]["train_batch_size"],
        eval_batch_size=CONFIG["dataset"]["eval_batch_size"],
    )
    datamodule.setup()
    
    # 3. 初始化模型
    model = Padim(
        backbone=CONFIG["model"]["backbone"],
        layers=CONFIG["model"]["layers"],
    )
    
    # 4. 初始化引擎
    engine = Engine(
        default_root_dir=CONFIG["engine"]["default_root_dir"],
        accelerator="cpu",     
        max_epochs=CONFIG["engine"]["max_epochs"],
    )
    
    # 5. 训练/拟合
    print("开始训练 (Fitting)...")
    engine.fit(model=model, datamodule=datamodule)
    
    # 6. 测试
    print("开始测试...")
    test_results = engine.test(model=model, datamodule=datamodule)
    print("测试结果:", test_results)
    
    # 7. 预测单张图片
    print("预测单张图片...")
    test_img_path = data_root / "test" / "bad" / "0.png"
    predictions = engine.predict(model=model, data_path=test_img_path)
    
    if predictions and len(predictions) > 0:
        batch = predictions[0]
        score = batch["pred_score"]
        label = batch["pred_label"]
        
        if hasattr(score, "item"): score = score.item()
        if hasattr(label, "item"): label = label.item()
            
        print(f">>> 预测结果:")
        print(f">>> 图片路径: {test_img_path}")
        print(f">>> 异常得分: {score:.4f}")
        print(f">>> 预测类别: {'异常' if label else '正常'}")

if __name__ == "__main__":
    run_demo()
