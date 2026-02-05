import cv2
import numpy as np
from ultralytics import YOLO
import os
from pathlib import Path

def rotate_image(image, angle):
    """旋转图片"""
    if angle == 0:
        return image
    elif angle == 90:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif angle == 180:
        return cv2.rotate(image, cv2.ROTATE_180)
    elif angle == 270:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return image

def apply_perspective_tta(image, mode):
    """模拟俯仰/透视形变 (TTA专用)"""
    h, w = image.shape[:2]
    src_pts = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    
    if mode == "tilt_up":
        # 模拟向上仰拍 (底部变宽)
        dst_pts = np.float32([[w*0.1, h*0.1], [w*0.9, h*0.1], [0, h], [w, h]])
    elif mode == "tilt_down":
        # 模拟向下俯拍 (顶部变宽)
        dst_pts = np.float32([[0, 0], [w, 0], [w*0.1, h*0.9], [w*0.9, h*0.9]])
    else:
        return image
        
    matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
    return cv2.warpPerspective(image, matrix, (w, h))

def get_crops_via_detection(image, detector):
    """使用检测器寻找物体并裁剪"""
    # 降低一点阈值，因为微调模型可能还在初期
    results = detector.predict(source=image, verbose=False, conf=0.25)
    crops = []
    
    # results[0] 代表的是输入列表中的第一张图片（因为我们只传了一张图）
    # 而 results[0].boxes 包含了这张图片中检测到的【所有】目标框
    all_boxes = results[0].boxes
    
    if all_boxes is not None and len(all_boxes) > 0:
        # 如果你只想识别图中“最像”的那一个目标，可以只取置信度最高的一个
        # 我们按照置信度排序，取第一个
        best_box_idx = all_boxes.conf.argmax()
        box = all_boxes[best_box_idx]
        
        # 获取坐标 (x1, y1, x2, y2)
        xyxy = box.xyxy[0].cpu().numpy().astype(int)
        conf = box.conf[0].item()
        # 适当扩大一点裁剪范围，给分类器留一点上下文
        h, w = image.shape[:2]
        pad = 20 # 增加一点边距
        x1 = max(0, xyxy[0] - pad)
        y1 = max(0, xyxy[1] - pad)
        x2 = min(w, xyxy[2] + pad)
        y2 = min(h, xyxy[3] + pad)
        
        crop = image[y1:y2, x1:x2]
        if crop.size > 0:
            crops.append((crop, (x1, y1, x2, y2), conf))
            
    return crops

def run_inference():
    # 模型权重路径
    cls_model_path = "/home/software/One2All/runs/classify/runs/classify/ele_type/weights/best.pt"
    # 使用你刚训练好的检测/分割模型路径
    det_model_path = "/home/software/One2All/runs/segment/runs/detect/ele_det/weights/best.pt" 
    
    # 测试图片目录
    test_dir = "/home/software/One2All/datasets/ele_type/test"
    
    if not os.path.exists(cls_model_path):
        print(f"错误: 找不到分类模型权重文件 {cls_model_path}")
        return
    
    if not os.path.exists(det_model_path):
        # 尝试另一个可能的路径
        alt_det_path = "/home/software/One2All/runs/detect/runs/detect/ele_det/weights/best.pt"
        if os.path.exists(alt_det_path):
            det_model_path = alt_det_path
        else:
            print(f"警告: 找不到自定义检测模型 {det_model_path}，将回退使用 yolov8n-seg.pt")
            det_model_path = "yolov8n-seg.pt"

    # 加载模型
    print(f"正在加载分类模型: {cls_model_path}")
    cls_model = YOLO(cls_model_path)
    print(f"正在加载检测模型 (用于目标定位): {det_model_path}")
    det_model = YOLO(det_model_path)

    # 获取所有图片文件
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    image_files = [f for f in os.listdir(test_dir) if f.lower().endswith(image_extensions)]

    if not image_files:
        print(f"在 {test_dir} 中未找到图片。")
        return

    print(f"开始推理 (采用: 检测 -> 自动裁剪 -> 多角度分类 流程)")
    print("-" * 60)

    results_dir = Path("runs/inference/combined_results")
    results_dir.mkdir(parents=True, exist_ok=True)

    # 原型图根目录
    prototype_root = Path("/home/software/One2All/datasets/ele_type_cls/train")

    for img_name in image_files:
        img_path = os.path.join(test_dir, img_name)
        img = cv2.imread(img_path)
        if img is None: continue
        
        display_img = img.copy() # 用于画框显示的图
        
        # 步骤 1: 找物体
        crops_info = get_crops_via_detection(img, det_model)
        
        # 如果没检测到物体，就对原图进行操作（兜底）
        if not crops_info:
            print(f"图片: {img_name} - ⚠️ 未检测到明确目标 (检测模型未发现物体)")
            crops_info = [(img, (0, 0, img.shape[1], img.shape[0]), 0.0)]
        
        for idx, (crop, box, det_conf) in enumerate(crops_info):
            # 在原图画框
            x1, y1, x2, y2 = box
            cv2.rectangle(display_img, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.putText(display_img, f"Det: {det_conf:.2f}", (x1, y1-15), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            best_conf = -1.0
            best_result = None
            best_angle = 0
            best_flip = "None"
            best_persp = "None"
            
            # 步骤 2: 对裁剪出的物体进行 TTA 推理
            CONF_THRESHOLD = 0.5
            transforms = [
                (0, None, None), (90, None, None), (180, None, None), (270, None, None),
                (0, "horizontal", None), (0, "vertical", None),
                (0, None, "tilt_up"), (0, None, "tilt_down")
            ]
            
            for angle, flip, persp in transforms:
                processed_crop = crop.copy()
                if angle != 0: processed_crop = rotate_image(processed_crop, angle)
                if flip == "horizontal": processed_crop = cv2.flip(processed_crop, 1)
                elif flip == "vertical": processed_crop = cv2.flip(processed_crop, 0)
                if persp: processed_crop = apply_perspective_tta(processed_crop, persp)
                    
                results = cls_model.predict(source=processed_crop, verbose=False)
                result = results[0]
                
                if result.probs is not None:
                    conf = result.probs.top1conf.item()
                    if conf > best_conf:
                        best_conf = conf
                        best_result = result
                        best_angle = angle
                        best_flip = flip if flip else "None"
                        best_persp = persp if persp else "None"
            
            if best_result:
                class_id = best_result.probs.top1
                label = best_result.names[class_id]
                
                # 状态判定
                is_recognized = best_conf >= CONF_THRESHOLD
                status_text = f"{label} ({best_conf:.2f})" if is_recognized else "Unknown"
                color = (0, 255, 0) if is_recognized else (0, 0, 255)

                print(f"图片: {img_name} (目标 #{idx+1})")
                if not is_recognized:
                    print(f"  结果: ⚠️ 无法识别该物体 (最高置信度 {best_conf:.4f} 属于 {label})")
                else:
                    print(f"  识别结果: {label} (置信度: {best_conf:.4f})")
                
                # 获取原型图
                proto_img = None
                proto_dir = prototype_root / label
                if proto_dir.exists():
                    proto_files = list(proto_dir.glob("*.jpg")) + list(proto_dir.glob("*.png"))
                    if proto_files:
                        proto_img = cv2.imread(str(proto_files[0]))

                # 拼接对比图
                h_orig, w_orig = display_img.shape[:2]
                target_h = h_orig
                
                # 1. 调整识别出的裁剪图大小
                aspect_crop = crop.shape[1] / crop.shape[0]
                target_w_crop = int(target_h * aspect_crop)
                resized_crop = cv2.resize(crop, (target_w_crop, target_h))
                
                # 2. 如果有原型图，也调整大小
                if proto_img is not None:
                    aspect_proto = proto_img.shape[1] / proto_img.shape[0]
                    target_w_proto = int(target_h * aspect_proto)
                    resized_proto = cv2.resize(proto_img, (target_w_proto, target_h))
                    
                    # 在原型图上标出“原型”字样
                    cv2.putText(resized_proto, "Prototype", (20, 50), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 0), 3)
                    
                    # 三图拼接：原图 + 识别裁剪图 + 训练集原型图
                    combined = np.hstack((display_img, resized_crop, resized_proto))
                    text_x = w_orig + target_w_crop + 20
                else:
                    # 两图拼接
                    combined = np.hstack((display_img, resized_crop))
                    text_x = w_orig + 20
                
                # 3. 在拼接图上写字 (增大字号)
                cv2.putText(combined, f"Result: {status_text}", (w_orig + 20, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 2.0, color, 4)
                cv2.putText(combined, f"View: {best_angle}deg, {best_flip}, {best_persp}", (w_orig + 20, 130), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)

                save_path = results_dir / f"result_{img_name}"
                cv2.imwrite(str(save_path), combined)
                print(f"  对比结果已保存: {save_path}")
                print("-" * 60)

    print(f"\n推理完成。对比结果保存在: {results_dir}")

if __name__ == "__main__":
    run_inference()
