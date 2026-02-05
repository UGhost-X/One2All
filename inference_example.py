import os
from pathlib import Path
from anomalib.models import Patchcore
from anomalib.engine import Engine
from PIL import Image
import numpy as np

def run_inference():
    # 1. 定义模型权重路径
    ckpt_path = "/home/software/One2All/results_patchcore_trial_11/Patchcore/dummy_patchcore/v0/weights/lightning/model.ckpt"
    
    # 2. 定义测试图片路径 (这里使用 dummy 数据集中的一张图片作为示例)
    test_img_path = "/home/software/One2All/datasets/dummy_patchcore/test/bad/0.png"
    
    if not os.path.exists(ckpt_path):
        print(f"错误: 找不到权重文件 {ckpt_path}")
        return

    if not os.path.exists(test_img_path):
        print(f"提示: 找不到测试图片 {test_img_path}，请确保数据集已生成。")
        # 如果找不到图片，我们尝试生成一张模拟图片用于演示
        test_img_path = Path("inference_test_temp.png")
        img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        img[100:150, 100:150, 0] = 255 # 添加一个红色异常块
        Image.fromarray(img).save(test_img_path)
        print(f"已生成临时测试图片: {test_img_path}")

    print(f"正在加载模型: {ckpt_path}")
    
    # 3. 加载模型
    # 在 Anomalib 2.x 中，可以直接使用 Patchcore.load_from_checkpoint
    model = Patchcore.load_from_checkpoint(ckpt_path)
    
    # 4. 初始化推理引擎
    # 指定 accelerator="cpu" 或 "gpu"
    engine = Engine(accelerator="auto")
    
    # 5. 执行推理
    print(f"正在对图片进行推理: {test_img_path}")
    predictions = engine.predict(model=model, data_path=test_img_path)
    
    # 6. 处理结果
    if predictions and len(predictions) > 0:
        # predictions 是一个 list，每个元素是一个 TorchDict (包含 pred_score, pred_label, anomaly_map 等)
        output = predictions[0]
        
        score = output["pred_score"]
        label = output["pred_label"]
        
        # 转换为标量
        if hasattr(score, "item"): score = score.item()
        if hasattr(label, "item"): label = label.item()
        
        print("\n" + "="*30)
        print(f"推理结果:")
        print(f"图片路径: {test_img_path}")
        print(f"异常得分: {score:.4f}")
        print(f"预测结果: {'异常 (Anomaly)' if label else '正常 (Normal)'}")
        print("="*30)
    else:
        print("推理失败，未获取到结果。")

if __name__ == "__main__":
    run_inference()
