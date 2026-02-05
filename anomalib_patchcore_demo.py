import os
# 设置Hugging Face Hub国内镜像
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import shutil
import numpy as np
import cv2  # 引入OpenCV用来做模糊，模拟真实纹理
from PIL import Image
from pathlib import Path
from anomalib.data import Folder
from anomalib.models import Patchcore
from anomalib.engine import Engine

def prepare_data(base_path: Path):
    """
    改进的数据生成：模拟工业表面的“纹理”
    PatchCore 需要特征提取，纯渐变图无法提取有效特征。
    """
    print(f"正在清理并准备模拟数据: {base_path}...")
    if base_path.exists():
        shutil.rmtree(base_path)
    
    train_good = base_path / "train" / "good"
    test_good = base_path / "test" / "good"
    test_bad = base_path / "test" / "bad"
    
    for path in [train_good, test_good, test_bad]:
        path.mkdir(parents=True, exist_ok=True)
    
    def generate_textured_image(seed, defect=False):
        np.random.seed(seed)
        size = 256
        base_noise = np.random.normal(128, 30, (size, size)).astype(np.uint8)
        texture = cv2.GaussianBlur(base_noise, (5, 5), 0)
        
        # 转成3通道
        img = cv2.cvtColor(texture, cv2.COLOR_GRAY2RGB)
        
        if defect:
            if np.random.rand() > 0.5:
                cv2.line(img, (50, 50), (150, 150), (0, 0, 0), thickness=3)
            # 缺陷2: 污渍 (颜色块)
            else:
                cv2.circle(img, (128, 128), 30, (255, 50, 50), thickness=-1)
                
        return img
    
    # 生成训练数据 (增加到30张，覆盖纹理的随机性)
    for i in range(30):
        img = generate_textured_image(i, defect=False)
        Image.fromarray(img).save(train_good / f"{i}.png")
    
    # 生成测试正常数据
    for i in range(10):
        img = generate_textured_image(100 + i, defect=False)
        Image.fromarray(img).save(test_good / f"{i}.png")
    
    # 生成异常数据
    for i in range(10):
        img = generate_textured_image(200 + i, defect=True)
        Image.fromarray(img).save(test_bad / f"{i}.png")

def run_demo():
    # --- 关键配置修改 ---
    CONFIG = {
        "model": {
            "backbone": "resnet18", 
            "layers": ["layer2", "layer3"],
            "coreset_sampling_ratio": 0.1,
        },
        "dataset": {
            "image_size": (256, 256),
            "train_batch_size": 4,
            "eval_batch_size": 4,
        },
        "engine": {
            "max_epochs": 1,
            "default_root_dir": "./results_patchcore_optimized",
        }
    }

    # 1. 准备数据
    data_root = Path("./datasets/dummy_texture")
    prepare_data(data_root)
    
    # 2. 数据模块
    datamodule = Folder(
        name="dummy_texture",
        root=data_root,
        normal_dir="train/good",
        abnormal_dir="test/bad",
        normal_test_dir="test/good",
        train_batch_size=CONFIG["dataset"]["train_batch_size"],
        eval_batch_size=CONFIG["dataset"]["eval_batch_size"],
        num_workers=4,
    )

    # 3. 模型
    model = Patchcore(
        backbone=CONFIG["model"]["backbone"],
        layers=CONFIG["model"]["layers"],
        coreset_sampling_ratio=CONFIG["model"]["coreset_sampling_ratio"],
    )
    
    # 4. 引擎
    engine = Engine(
        default_root_dir=CONFIG["engine"]["default_root_dir"],
        accelerator="auto",
        devices=1,
        max_epochs=CONFIG["engine"]["max_epochs"],
    )
    
    # 5. 训练
    print("\n--- 开始训练 (Fitting) ---")
    engine.fit(model=model, datamodule=datamodule)
    
    print("\n--- 开始测试 (Computing Metrics) ---")
    test_results = engine.test(model=model, datamodule=datamodule)
    
    # 7. 预测并保存可视化结果
    print("\n--- 正在生成可视化结果 ---")
    test_img_path = data_root / "test" / "bad" / "0.png"
    predictions = engine.predict(model=model, data_path=test_img_path)
    
    # 处理预测结果
    if predictions:
        pred = predictions[0]
        score = pred["pred_score"]
        # 如果是Tensor则转float
        if hasattr(score, "item"): score = score.item()
            
        print(f"图片: {test_img_path}")
        print(f"异常得分: {score:.4f} (越高越异常)")

        print(f"可视化结果已保存至: {CONFIG['engine']['default_root_dir']}")

if __name__ == "__main__":
    run_demo()

# import optuna
# from optuna.integration.pytorch_lightning import PyTorchLightningPruningCallback
# from torchmetrics import AUROC

# def run_hyperparameter_search():
#     # 1. 准备数据（复用你的prepare_data，仅运行1次）
#     data_root = Path("./datasets/dummy_patchcore")
#     prepare_data(data_root)

#     # 2. 定义超参数搜索的目标函数（核心：输入参数组合，输出模型效果指标）
#     def objective(trial: optuna.Trial):
#         # 【关键】定义待搜索的超参数空间（可根据需求扩展）
#         config = {
#             "model": {
#                 # 候选值搜索：backbone从指定列表中选
#                 "backbone": trial.suggest_categorical("backbone", ["resnet18", "resnet34"]),
#                 # 候选值搜索：layers从指定列表中选（PatchCore经典组合）
#                 "layers": trial.suggest_categorical("layers", [("layer1", "layer2"), ("layer1", "layer2", "layer3")]),
#                 # 数值范围搜索：coreset_sampling_ratio从0.1~0.5中选，步长0.1
#                 "coreset_sampling_ratio": trial.suggest_float("coreset_sampling_ratio", 0.1, 0.5, step=0.1),
#             },
#             "dataset": {
#                 "image_size": (256, 256),  # 固定，无需搜索
#                 # 数值候选搜索：batch_size从[4,8,16]中选
#                 "train_batch_size": trial.suggest_categorical("train_batch_size", [4, 8, 16]),
#                 "eval_batch_size": trial.suggest_categorical("eval_batch_size", [4, 8, 16]),
#             },
#             "engine": {
#                 "max_epochs": 1,  # PatchCore固定为1，绝对不搜索
#                 "default_root_dir": f"./results_patchcore_trial_{trial.number}",  # 每个试验单独保存结果
#             }
#         }

#         # 3. 配置数据模块（与原代码一致，使用当前试验的参数）
#         datamodule = Folder(
#             name="dummy_patchcore",
#             root=data_root,
#             normal_dir="train/good",
#             abnormal_dir="test/bad",
#             normal_test_dir="test/good",
#             train_batch_size=config["dataset"]["train_batch_size"],
#             eval_batch_size=config["dataset"]["eval_batch_size"],
#             num_workers=8,
#         )
#         datamodule.setup()

#         # 4. 初始化模型（使用当前试验的参数）
#         model = Patchcore(
#             backbone=config["model"]["backbone"],
#             layers=config["model"]["layers"],
#             coreset_sampling_ratio=config["model"]["coreset_sampling_ratio"],
#         )

#         # 5. 初始化引擎
#         engine = Engine(
#             default_root_dir=config["engine"]["default_root_dir"],
#             accelerator="gpu",
#             max_epochs=config["engine"]["max_epochs"],
#         )

#         # 6. 训练+测试（返回测试集核心指标：image_AUROC）
#         engine.fit(model=model, datamodule=datamodule)
#         test_results = engine.test(model=model, datamodule=datamodule)
#         auroc_score = test_results[0]["image_AUROC"]  # 取图像级AUROC作为优化目标

#         # 7. 返回指标（Optuna会最大化该值，找到AUROC最高的参数组合）
#         return auroc_score

#     # 3. 启动超参数搜索
#     print("===== 开始PatchCore超参数搜索 =====")
#     # 创建研究对象：优化目标为「最大化AUROC」，存储搜索结果到本地
#     study = optuna.create_study(
#         direction="maximize",  # 核心：AUROC越大效果越好，所以最大化
#         study_name="patchcore_anomaly_detection",
#         storage="sqlite:///patchcore_hpo.db",  # 搜索结果保存到sqlite数据库，可后续查看
#         load_if_exists=True,  # 若数据库已存在，加载原有结果（避免重复搜索）
#     )

#     # 运行搜索：指定试验次数（即尝试多少个参数组合，根据需求调整）
#     study.optimize(
#         objective,
#         n_trials=3,  # 尝试3个参数组合，参数空间大则增大（如20/30）
#         show_progress_bar=True,  # 显示搜索进度条
#     )

#     # 4. 输出搜索结果（核心：最优参数+最优指标）
#     print("\n===== 超参数搜索完成 - 最优结果 =====")
#     print(f"🏆 最优图像级AUROC: {study.best_value:.4f}")
#     print(f"🔧 最优参数组合: {study.best_params}")
#     print(f"📊 最优试验编号: {study.best_trial.number}")

#     # 可选：打印所有试验的详细结果
#     print("\n===== 所有试验结果汇总 =====")
#     for trial in study.trials:
#         value_str = f"{trial.value:.4f}" if trial.value is not None else "N/A"
#         print(f"试验{trial.number} | AUROC: {value_str} | 参数: {trial.params}")

#     # 5. 使用最优参数重新训练最终模型（可选，得到最优模型）
#     print("\n===== 使用最优参数训练最终模型 =====")
#     best_config = {
#         "model": {
#             "backbone": study.best_params["backbone"],
#             "layers": study.best_params["layers"],
#             "coreset_sampling_ratio": study.best_params["coreset_sampling_ratio"],
#         },
#         "dataset": {
#             "image_size": (256, 256),
#             "train_batch_size": study.best_params["train_batch_size"],
#             "eval_batch_size": study.best_params["eval_batch_size"],
#         },
#         "engine": {
#             "max_epochs": 1,
#             "default_root_dir": "./results_patchcore_best",
#         }
#     }

#     # 用最优参数初始化组件并训练
#     datamodule_best = Folder(
#         name="dummy_patchcore",
#         root=data_root,
#         normal_dir="train/good",
#         abnormal_dir="test/bad",
#         normal_test_dir="test/good",
#         train_batch_size=best_config["dataset"]["train_batch_size"],
#         eval_batch_size=best_config["dataset"]["eval_batch_size"],
#         num_workers=8,
#     )
#     datamodule_best.setup()

#     model_best = Patchcore(**best_config["model"])
#     engine_best = Engine(
#         default_root_dir=best_config["engine"]["default_root_dir"],
#         accelerator="gpu",
#         max_epochs=best_config["engine"]["max_epochs"],
#     )
#     engine_best.fit(model=model_best, datamodule=datamodule_best)

#     # 用最优模型预测单张缺陷图
#     print("\n===== 最优模型单张缺陷图预测 =====")
#     test_img_path = data_root / "test" / "bad" / "0.png"
#     predictions = engine_best.predict(model=model_best, data_path=test_img_path)
#     if predictions and len(predictions) > 0:
#         batch = predictions[0]
#         score = batch["pred_score"].item() if hasattr(batch["pred_score"], "item") else batch["pred_score"]
#         label = batch["pred_label"].item() if hasattr(batch["pred_label"], "item") else batch["pred_label"]
#         print(f">>> 图片路径: {test_img_path}")
#         print(f">>> 异常得分: {score:.4f}")
#         print(f">>> 预测类别: {'异常' if label else '正常'}")

# if __name__ == "__main__":
#     # 替换原有run_demo，启动超参数搜索
#     run_hyperparameter_search()
