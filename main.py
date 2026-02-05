from __future__ import annotations

import argparse
from pathlib import Path
from ultralytics import YOLO

def _download_file(urls: list[str], target_path: Path) -> Path:
    import requests

    last_exc: Exception | None = None
    for url in urls:
        try:
            response = requests.get(url, timeout=60)
            response.raise_for_status()
            target_path.parent.mkdir(parents=True, exist_ok=True)
            target_path.write_bytes(response.content)
            return target_path
        except Exception as exc:
            last_exc = exc
    raise RuntimeError(f"下载失败: {last_exc}")


def _download_default_image(target_path: Path) -> Path:
    urls = [
        "https://ultralytics.com/images/bus.jpg",
        "https://raw.githubusercontent.com/ultralytics/assets/main/bus.jpg",
    ]
    return _download_file(urls, target_path)


def _ensure_yolov8_weights(weights: str) -> Path:
    weights_path = Path(weights)
    if weights_path.exists():
        return weights_path

    if weights_path.parent != Path("."):
        raise FileNotFoundError(f"找不到权重文件: {weights_path}")

    dest = Path.cwd() / "weights" / weights_path.name
    if dest.exists():
        return dest

    # 根据文件名选择不同的下载路径（如果需要的话，目前通用）
    urls = [
        f"https://www.modelscope.cn/api/v1/models/ultralytics/YOLOv8/repo?Revision=master&FilePath={weights_path.name}",
        f"https://hf-mirror.com/Ultralytics/YOLOv8/resolve/main/{weights_path.name}",
        f"https://huggingface.co/Ultralytics/YOLOv8/resolve/main/{weights_path.name}",
        f"https://ghproxy.com/https://github.com/ultralytics/assets/releases/download/v8.4.0/{weights_path.name}",
        f"https://github.com/ultralytics/assets/releases/download/v8.4.0/{weights_path.name}",
    ]
    return _download_file(urls, dest)


def yolov8_train_cls(data_dir: str, epochs: int, imgsz: int, model_name: str = "yolov8n-cls.pt") -> int:
    try:
        from ultralytics import YOLO
    except Exception:
        print("未检测到 ultralytics。")
        return 2

    print(f"开始分类模型微调，使用数据集: {data_dir}")
    try:
        weights_path = _ensure_yolov8_weights(model_name)
    except Exception as exc:
        print(f"权重准备失败：{exc}")
        return 2

    model = YOLO(str(weights_path))
    
    # 开始训练
    # 对于极小数据集（如只有3张图），我们开启数据增强，并增加训练轮数
    results = model.train(
        data=data_dir,
        epochs=epochs,
        imgsz=imgsz,
        project="runs/classify",
        name="ele_type",
        exist_ok=True,
        amp=False,
        degrees=180,  # 允许 180 度旋转增强
        flipud=0.5,   # 允许上下翻转
        fliplr=0.5,   # 允许左右翻转
        mixup=0.2,    # 增加 mixup 增强以提高泛化能力
        perspective=0.005, # 增加透视变换，模拟俯仰角度带来的形变 (0.0 - 0.01)
        shear=10      # 增加剪切变换，模拟侧视角度
    )
    print(f"训练完成。模型保存在: {results.save_dir}")
    return 0


def yolov8_demo(weights: str, image: str | None, output: str, conf: float, device: str | None) -> int:
    if image is None:
        default_image_path = Path.cwd() / "yolov8_bus.jpg"
        if not default_image_path.exists():
            try:
                _download_default_image(default_image_path)
            except Exception as exc:
                print(f"无法下载默认示例图片：{exc}")
                print("请使用 --image 指定本地图片路径。")
                return 2
        image_path = default_image_path
    else:
        image_path = Path(image)

    if not image_path.exists():
        print(f"找不到图片：{image_path}")
        return 2

    try:
        weights_path = _ensure_yolov8_weights(weights)
    except Exception as exc:
        print(f"权重准备失败：{exc}")
        print("请使用 --weights 指定本地 .pt 权重路径（例如 yolov8n.pt）。")
        return 2

    model = YOLO(str(weights_path))
    results = model.predict(source=str(image_path), conf=conf, device=device)
    result = results[0]

    boxes = []
    if result.boxes is not None and len(result.boxes) > 0:
        for box in result.boxes:
            cls_id = int(box.cls.item()) if box.cls is not None else -1
            score = float(box.conf.item()) if box.conf is not None else 0.0
            xyxy = box.xyxy[0].tolist() if box.xyxy is not None else []
            name = result.names.get(cls_id, str(cls_id)) if hasattr(result, "names") else str(cls_id)
            boxes.append({"class_id": cls_id, "class_name": name, "score": score, "xyxy": xyxy})

    annotated = result.plot()
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        import cv2

        cv2.imwrite(str(output_path), annotated)
    except Exception:
        from PIL import Image

        Image.fromarray(annotated[:, :, ::-1]).save(output_path)

    print(f"输入图片: {image_path}")
    print(f"输出图片: {output_path}")
    print(f"检测框数量: {len(boxes)}")
    if boxes:
        for b in boxes[:10]:
            print(b)
    return 0


def yolov8_train_det(data_yaml: str, epochs: int, imgsz: int, model_name: str = "yolov8n-seg.pt") -> int:    
    try:
        from ultralytics import YOLO
    except Exception:
        print("未检测到 ultralytics。")
        return 2

    # 如果是多边形标注，使用分割模型 (-seg) 效果更好
    print(f"开始目标检测/分割模型微调，使用数据集配置: {data_yaml}")
    try:
        weights_path = _ensure_yolov8_weights(model_name)
    except Exception as exc:
        print(f"权重准备失败：{exc}")
        return 2

    model = YOLO(str(weights_path))
    
    # 开始训练 (YOLOv8 会自动根据权重类型识别任务是 detect 还是 segment)
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        project="runs/detect",
        name="ele_det",
        exist_ok=True,
        amp=False,
        # 针对多边形和复杂角度的增强
        degrees=180,    # 允许大幅度旋转
        flipud=0.5,
        fliplr=0.5,
        perspective=0.001,
        shear=10,
        mosaic=1.0
    )
    print(f"训练完成。模型保存在: {results.save_dir}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(prog="one2all")
    subparsers = parser.add_subparsers(dest="command", required=True)

    yolo_parser = subparsers.add_parser("yolov8", help="运行 YOLOv8 单张图片检测 demo")
    yolo_parser.add_argument("--weights", default="yolov8n.pt")
    yolo_parser.add_argument("--image", default=None)
    yolo_parser.add_argument("--output", default=str(Path.cwd() / "yolov8_out.jpg"))
    yolo_parser.add_argument("--conf", type=float, default=0.25)
    yolo_parser.add_argument("--device", default="gpu")

    train_cls_parser = subparsers.add_parser("train-cls", help="微调 YOLOv8 分类模型")
    train_cls_parser.add_argument("--data", default=str(Path.cwd() / "datasets/ele_type_cls"), help="分类数据集根目录")
    train_cls_parser.add_argument("--epochs", type=int, default=50, help="训练轮数")
    train_cls_parser.add_argument("--imgsz", type=int, default=224, help="输入图片尺寸")
    train_cls_parser.add_argument("--model", default="yolov8n-cls.pt", help="基础权重文件名")

    train_det_parser = subparsers.add_parser("train-det", help="微调 YOLOv8 检测模型")
    train_det_parser.add_argument("--data", required=True, help="数据集 yaml 配置文件路径")
    train_det_parser.add_argument("--epochs", type=int, default=50, help="训练轮数")
    train_det_parser.add_argument("--imgsz", type=int, default=640, help="输入图片尺寸")
    train_det_parser.add_argument("--model", default="yolov8n-seg.pt", help="基础权重文件名")

    args = parser.parse_args()

    if args.command == "yolov8":
        return yolov8_demo(args.weights, args.image, args.output, args.conf, args.device)
    elif args.command == "train-cls":
        return yolov8_train_cls(args.data, args.epochs, args.imgsz, args.model)
    elif args.command == "train-det":
        return yolov8_train_det(args.data, args.epochs, args.imgsz, args.model)
    return 0


if __name__ == "__main__":
    # yolov8_train_cls("/home/software/One2All/datasets/ele_type_cls", 50, 224, "/home/software/One2All/runs/classify/runs/classify/ele_type/weights/best.pt")
    yolov8_train_det("/home/software/One2All/datasets/ele_obj_det/ele_obj_det.yaml",200, 640, "yolov8n-seg.pt")