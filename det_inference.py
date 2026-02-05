import os
import cv2
from ultralytics import YOLO
from pathlib import Path

def run_det_inference():
    # 1. 加载你训练好的分割模型
    model_path = "/home/software/One2All/runs/segment/runs/detect/ele_det/weights/best.pt"
    
    if not os.path.exists(model_path):
        print(f"提示: 尚未找到自定义训练好的模型 {model_path}。")
        alt_path = "/home/software/One2All/runs/detect/ele_det/weights/best.pt"
        if os.path.exists(alt_path):
            model_path = alt_path
        else:
            model_path = "yolov8n-seg.pt"
            print(f"正在加载基础分割模型 {model_path} 进行演示...")

    print(f"正在加载模型: {model_path}")
    model = YOLO(model_path)
    
    # 2. 测试图片目录
    test_dir = "/home/software/One2All/datasets/ele_type/test"
    # 修改输出目录，方便查看
    output_dir = Path("runs/detect/test_results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not os.path.exists(test_dir):
        print(f"错误: 找不到测试目录 {test_dir}")
        return

    # 3. 运行推理
    print(f"开始对目录 {test_dir} 进行推理测试 (阈值设为 0.1)...")
    # 设置较低的阈值，并强制开启 save_txt 和 save_crop 观察中间结果
    results = model.predict(
        source=test_dir, 
        save=True,          # 自动保存带有绘制框/掩码的图片
        conf=0.1,           # 降低阈值到 0.1
        project="runs/detect", 
        name="test_results", 
        exist_ok=True,
        line_width=3        # 增加线宽方便观察
    )
    
    print("-" * 50)
    print(f"测试完成！")
    # 获取真正的保存路径
    actual_save_dir = results[0].save_dir
    print(f"1. 可视化结果已保存至: {actual_save_dir}")
    
    # 统计检测情况
    total_images = len(results)
    detected_count = 0
    for i, r in enumerate(results):
        if len(r.boxes) > 0:
            detected_count += 1
            print(f"   [图片 {i+1}] 识别到 {len(r.boxes)} 个目标, 最高置信度: {r.boxes.conf.max().item():.4f}")
        else:
            print(f"   [图片 {i+1}] 未识别到目标")
            
    print(f"2. 统计数据:")
    print(f"   - 测试图片总数: {total_images}")
    print(f"   - 成功检测到目标的图片数: {detected_count}")
    print(f"   - 检测率: {(detected_count/total_images)*100:.2f}%")
    print("-" * 50)
    print("提示：如果图片中还是没框，请检查 runs/detect/test_results 下生成的图片。")

if __name__ == "__main__":
    run_det_inference()
