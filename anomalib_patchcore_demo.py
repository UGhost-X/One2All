import os
import numpy as np
from PIL import Image
from pathlib import Path
from anomalib.data import Folder
from anomalib.models import Patchcore
from anomalib.engine import Engine

def prepare_data(base_path: Path):
    """创建一些模拟数据用于演示"""
    print("正在准备模拟数据...")
    train_good = base_path / "train" / "good"
    test_good = base_path / "test" / "good"
    test_bad = base_path / "test" / "bad"
    
    for path in [train_good, test_good, test_bad]:
        path.mkdir(parents=True, exist_ok=True)
        
    # 生成正常的训练图片 (使用结构化背景而非纯随机噪声，这对预训练模型更友好)
    for i in range(20): 
        # 创建一个渐变背景作为“正常”模式
        x = np.linspace(0, 1, 256)
        y = np.linspace(0, 1, 256)
        xv, yv = np.meshgrid(x, y)
        base = (xv * 100 + yv * 50 + 50).astype(np.uint8)
        img = np.stack([base, base, base], axis=-1)
        # 加入极小量的扰动
        noise = np.random.randint(-5, 5, (256, 256, 3))
        img = np.clip(img + noise, 0, 255).astype(np.uint8)
        Image.fromarray(img).save(train_good / f"{i}.png")
        
    # 生成正常的测试图片
    for i in range(5):
        x = np.linspace(0, 1, 256)
        y = np.linspace(0, 1, 256)
        xv, yv = np.meshgrid(x, y)
        base = (xv * 100 + yv * 50 + 50).astype(np.uint8)
        img = np.stack([base, base, base], axis=-1)
        Image.fromarray(img).save(test_good / f"{i}.png")
        
    # 生成带“缺陷”的测试图片 (在正常背景上放一个异常形状)
    for i in range(5):
        x = np.linspace(0, 1, 256)
        y = np.linspace(0, 1, 256)
        xv, yv = np.meshgrid(x, y)
        base = (xv * 100 + yv * 50 + 50).astype(np.uint8)
        img = np.stack([base, base, base], axis=-1)
        # 模拟缺陷：在随机位置放一个亮块
        img[120:160, 120:160, :] = 255 
        Image.fromarray(img).save(test_bad / f"{i}.png")
    print(f"数据准备完成: {base_path}")

def run_demo():
    # --- 超参数配置区 ---
    CONFIG = {
        "model": {
            "backbone": "wide_resnet50_2",
            "layers": ["layer2", "layer3"],
            "coreset_sampling_ratio": 0.5,  # 提高采样率，保留更多特征细节（适合小样本）
        },
        "dataset": {
            "image_size": (256, 256),       # 提高分辨率可增强微小缺陷检测
            "train_batch_size": 4,
            "eval_batch_size": 4,
        },
        "engine": {
            "max_epochs": 1,                # PatchCore 只需要 1 次遍历提取特征
            "default_root_dir": "./results_patchcore",
        }
    }
    # ------------------

    # 1. 准备数据
    data_root = Path("./datasets/dummy_patchcore")
    prepare_data(data_root)
    
    # 2. 配置数据模块
    datamodule = Folder(
        name="dummy_patchcore",
        root=data_root,
        normal_dir="train/good",
        abnormal_dir="test/bad",
        normal_test_dir="test/good",
        train_batch_size=CONFIG["dataset"]["train_batch_size"],
        eval_batch_size=CONFIG["dataset"]["eval_batch_size"],
    )
    datamodule.setup()
    
    # 3. 初始化模型
    # 使用超参数初始化 Patchcore
    model = Patchcore(
        backbone=CONFIG["model"]["backbone"],
        layers=CONFIG["model"]["layers"],
        coreset_sampling_ratio=CONFIG["model"]["coreset_sampling_ratio"],
    )
    
    # 4. 初始化引擎
    engine = Engine(
        default_root_dir=CONFIG["engine"]["default_root_dir"],
        accelerator="gpu",
        max_epochs=CONFIG["engine"]["max_epochs"],
    )
    
    # 5. 训练/拟合
    print("开始训练 PatchCore (Fitting memory bank)...")
    engine.fit(model=model, datamodule=datamodule)
    
    # 6. 测试
    print("开始测试...")
    test_results = engine.test(model=model, datamodule=datamodule)
    print("测试结果:", test_results)
    
    # 7. 预测单张图片
    print("\n正在预测单张“缺陷”图片...")
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
