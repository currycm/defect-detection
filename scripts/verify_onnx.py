"""验证 ONNXRuntime 推理结果与 ultralytics(PyTorch) 是否一致。

自己实现的解码 + NMS 必须与原框架对齐，否则部署就跑偏了。
做法：同一批图分别用两条链路推理，按 IoU+类别配对后比较框坐标与置信度。
"""
import os

from _bootstrap import ensure_project_root

PROJECT = ensure_project_root()

import cv2
import numpy as np
from ultralytics import YOLO

from src.constants import CLASS_NAMES
from src.inference.onnx_predictor import ONNXDefectPredictor
from src.utils import paths

ONNX_PATH = str(paths.ONNX_PATH)
PT_PATH = str(paths.BEST_PT)
TEST_DIR = str(paths.PROCESSED_DIR / "images" / "test")
N_IMAGES = 6
CONF, IOU = 0.25, 0.5


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(ix2 - ix1, 0) * max(iy2 - iy1, 0)
    ua = (max(a[2] - a[0], 0) * max(a[3] - a[1], 0)
          + max(b[2] - b[0], 0) * max(b[3] - b[1], 0) - inter)
    return inter / ua if ua > 0 else 0.0


def main():
    onnx_p = ONNXDefectPredictor(ONNX_PATH, CLASS_NAMES, imgsz=640,
                                 conf=CONF, iou=IOU)
    pt_model = YOLO(PT_PATH)
    print("ONNX provider:", onnx_p.providers)

    # 按类别均匀取样，确保 NMS / 解码在每一类上都对齐（不能只测单一类别）
    imgs = []
    for cls in CLASS_NAMES:
        cls_imgs = sorted([f for f in os.listdir(TEST_DIR)
                           if f.lower().startswith(cls.lower())
                           and f.lower().endswith((".jpg", ".png"))])
        imgs.extend(cls_imgs[:N_IMAGES])
    print(f"取样 {len(imgs)} 张（每类 {N_IMAGES} 张）\n")

    total_pt = total_onnx = matched = 0
    max_box_diff = 0.0
    max_conf_diff = 0.0

    for name in imgs:
        path = os.path.join(TEST_DIR, name)
        img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            print(f"  跳过（读图失败）: {name}")
            continue

        res = pt_model.predict(img, imgsz=640, conf=CONF, iou=IOU, verbose=False)[0]
        pt_dets = []
        if res.boxes is not None:
            for b in res.boxes:
                pt_dets.append({
                    "xyxy": b.xyxy[0].tolist(),
                    "conf": float(b.conf[0]),
                    "cls": int(b.cls[0]),
                })
        onnx_dets = onnx_p.predict_image(img, conf=CONF, iou=IOU)

        total_pt += len(pt_dets)
        total_onnx += len(onnx_dets)

        # 配对：同类别且 IoU>0.5
        used = set()
        for pd in pt_dets:
            best, best_iou = None, 0.0
            for j, od in enumerate(onnx_dets):
                if j in used or od["cls"] != pd["cls"]:
                    continue
                v = iou(pd["xyxy"], od["xyxy"])
                if v > best_iou:
                    best, best_iou = j, v
            if best is not None and best_iou >= 0.5:
                used.add(best)
                matched += 1
                od = onnx_dets[best]
                # strict=True：两侧 xyxy 都应恰好 4 个值，长度不等说明某一侧
                # 后处理出错，此时宁可报错也不要静默截断成"看起来一致"。
                max_box_diff = max(max_box_diff,
                                   max(abs(a - b_) for a, b_ in
                                       zip(pd["xyxy"], od["xyxy"], strict=True)))
                max_conf_diff = max(max_conf_diff, abs(pd["conf"] - od["conf"]))

        print(f"  {name:<28} ultralytics={len(pt_dets):<3} onnx={len(onnx_dets):<3}")

    print(f"\n合计：ultralytics {total_pt} 框，ONNX {total_onnx} 框，配对成功 {matched} 框")
    print(f"配对框坐标最大差异: {max_box_diff:.3f} px")
    print(f"配对框置信度最大差异: {max_conf_diff:.4f}")
    agree = matched / max(total_pt, 1)
    print(f"\n一致率(matched/ultralytics): {agree:.3f}")
    print("结论:", "一致 ✓" if agree >= 0.95 and max_box_diff < 5 else "有偏差，需排查 ✗")


if __name__ == "__main__":
    main()
