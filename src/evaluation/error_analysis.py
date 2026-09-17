"""Step 6：难例 / 错误分析。

目标：Step5 消融已证明「继续盲目调参无效」，这一步改为**用证据找根因**——
把模型在验证集上的每一处错误归类并可视化，回答三个问题：

  - 漏检 FN(missed)   ：哪个类的缺陷根本没被框出来？
  - 误检 FP(background)：模型在什么区域凭空造框？
  - 混淆 confusion     ：框的位置对了但类别认错了？（IoU 达标但 cls 不同）

输出：
  - out_dir/summary.csv    逐类 TP / FP-混淆 / FP-背景 / FN 统计
  - out_dir/confusion.csv  含 background 行列的 7x7 混淆矩阵
  - out_dir/<class>/*.jpg  重点类的错误样本可视化
      绿框 = 真值 GT    蓝框 = 检对 TP    红框 = 模型错了（FP 或类别混淆）

本机环境注意：
  - 中文目录：cv2.imread / imwrite 会静默失败，统一走 utils.imageio 的
    load_image / save_image（全项目唯一实现）。
  - ultralytics 内部也用 cv2 读图，所以这里自己读成 ndarray 再喂给 model.predict，
    不把中文路径交给它。
"""
from __future__ import annotations

import csv
import os
from pathlib import Path

import cv2
import numpy as np

from ..utils.imageio import load_image, save_image

# BGR 颜色：绿=真值，蓝=检对，红=模型错了
C_GT = (0, 200, 0)
C_TP = (255, 150, 0)
C_ERR = (0, 0, 255)


def _xywhn_to_xyxy(box, w: int, h: int) -> list[float]:
    cx, cy, bw, bh = box
    return [
        (cx - bw / 2) * w,
        (cy - bh / 2) * h,
        (cx + bw / 2) * w,
        (cy + bh / 2) * h,
    ]


def _load_gt(label_path: str, w: int, h: int) -> list[dict]:
    """读 YOLO txt 真值，转成像素坐标 xyxy。"""
    gts: list[dict] = []
    if not os.path.exists(label_path):
        return gts
    with open(label_path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 5:
                continue
            cls = int(float(parts[0]))
            box = list(map(float, parts[1:5]))
            gts.append({"cls": cls, "xyxy": _xywhn_to_xyxy(box, w, h)})
    return gts


def _iou(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(ix2 - ix1, 0.0), max(iy2 - iy1, 0.0)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(ax2 - ax1, 0.0) * max(ay2 - ay1, 0.0)
    area_b = max(bx2 - bx1, 0.0) * max(by2 - by1, 0.0)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _greedy_match(preds: list[dict], gts: list[dict], iou_thr: float):
    """按 IoU 从高到低贪心一对一匹配，返回 [(gt_idx, pred_idx, iou), ...]。"""
    pairs = []
    for pi, p in enumerate(preds):
        for gi, g in enumerate(gts):
            iou = _iou(p["xyxy"], g["xyxy"])
            if iou >= iou_thr:
                pairs.append((iou, gi, pi))
    pairs.sort(key=lambda t: -t[0])
    used_g, used_p, matches = set(), set(), []
    for iou, gi, pi in pairs:
        if gi in used_g or pi in used_p:
            continue
        used_g.add(gi)
        used_p.add(pi)
        matches.append((gi, pi, iou))
    return matches


def analyze_image(model, img_path: str, label_path: str, imgsz: int,
                  conf: float, iou_thr: float):
    """单张图：推理 + 与 GT 比对，返回统计与可视化要素。"""
    img = load_image(img_path)
    if img is None:
        return None
    h, w = img.shape[:2]

    res = model.predict(img, imgsz=imgsz, conf=conf, verbose=False)[0]
    preds: list[dict] = []
    if res.boxes is not None and len(res.boxes) > 0:
        for b in res.boxes:
            xyxy = b.xyxy[0].tolist()
            preds.append({
                "cls": int(b.cls[0].item()),
                "conf": float(b.conf[0].item()),
                "xyxy": xyxy,
            })

    gts = _load_gt(label_path, w, h)
    matches = _greedy_match(preds, gts, iou_thr)

    tp, conf_err, fp_bg, fn = 0, 0, 0, 0
    confusions: list[tuple[int, int]] = []   # (gt_cls, pred_cls)
    matched_g = {gi: pi for gi, pi, _ in matches}

    for gi, pi, _ in matches:
        if preds[pi]["cls"] == gts[gi]["cls"]:
            tp += 1
        else:
            conf_err += 1
            confusions.append((gts[gi]["cls"], preds[pi]["cls"]))

    matched_p = {pi for _, pi, _ in matches}
    fp_classes = [preds[i]["cls"] for i in range(len(preds)) if i not in matched_p]
    fp_bg = len(fp_classes)

    fn_classes = []
    for gi in range(len(gts)):
        if gi not in matched_g:
            fn += 1
            fn_classes.append(gts[gi]["cls"])

    return {
        "img": img, "gts": gts, "preds": preds, "matches": matches,
        "tp": tp, "conf_err": conf_err, "fp_bg": fp_bg, "fn": fn,
        "confusions": confusions, "fp_classes": fp_classes, "fn_classes": fn_classes,
        "n_err": conf_err + fp_bg + fn,
    }


def _draw(img: np.ndarray, gts, preds, matches, names) -> np.ndarray:
    """绿=GT，蓝=检对 TP，红=错误（FP 或类别混淆）。"""
    out = img.copy()
    matched_p = {}
    for gi, pi, _ in matches:
        matched_p[pi] = gi

    for gi, g in enumerate(gts):
        x1, y1, x2, y2 = map(int, g["xyxy"])
        cv2.rectangle(out, (x1, y1), (x2, y2), C_GT, 2)
        cv2.putText(out, f"GT {names[g['cls']]}", (x1, max(y1 - 5, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, C_GT, 1, cv2.LINE_AA)

    for pi, p in enumerate(preds):
        x1, y1, x2, y2 = map(int, p["xyxy"])
        if pi in matched_p:
            correct = p["cls"] == gts[matched_p[pi]]["cls"]
            color = C_TP if correct else C_ERR
        else:
            color = C_ERR
        tag = f"{names[p['cls']]} {p['conf']:.2f}"
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        cv2.putText(out, tag, (x1, max(y2 - 5, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
    return out


def export_error_cases(weights: str, data_root: str, split: str = "val",
                       imgsz: int = 640, conf: float = 0.25, iou_thr: float = 0.5,
                       out_dir: str = "runs/error_analysis",
                       focus: tuple[str, ...] = ("crazing", "pitted_surface"),
                       max_save: int = 8) -> dict:
    """对 val 集逐张推理并与 GT 比对，导出统计与错误可视化。"""
    from ultralytics import YOLO

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(weights)
    names = model.names

    images_dir = Path(data_root) / "images" / split
    labels_dir = Path(data_root) / "labels" / split
    img_paths = sorted([p for p in images_dir.iterdir()
                        if p.suffix.lower() in (".jpg", ".jpeg", ".png")])
    print(f"待分析图片: {len(img_paths)} 张（{split} 集），conf={conf}, iou_thr={iou_thr}")

    n_cls = len(names)
    # 混淆矩阵：行列最后一位是 background
    conf_mat = np.zeros((n_cls + 1, n_cls + 1), dtype=int)
    per_cls = {i: {"tp": 0, "conf_out": 0, "conf_in": 0, "fp_bg": 0, "fn": 0}
               for i in range(n_cls)}
    records = []

    for idx, ip in enumerate(img_paths, 1):
        lp = labels_dir / (ip.stem + ".txt")
        r = analyze_image(model, str(ip), str(lp), imgsz, conf, iou_thr)
        if r is None:
            print(f"  [跳过] 读图失败: {ip.name}")
            continue

        for gi, pi, _ in r["matches"]:
            gc, pc = r["gts"][gi]["cls"], r["preds"][pi]["cls"]
            conf_mat[gc][pc] += 1
            if gc == pc:
                per_cls[gc]["tp"] += 1
            else:
                per_cls[gc]["conf_out"] += 1   # 该类被认成了别的类
                per_cls[pc]["conf_in"] += 1    # 别的类被认成了该类
        for pi in range(len(r["preds"])):
            if pi not in {p for _, p, _ in r["matches"]}:
                conf_mat[n_cls][r["preds"][pi]["cls"]] += 1
                per_cls[r["preds"][pi]["cls"]]["fp_bg"] += 1
        for gi in range(len(r["gts"])):
            if gi not in {g for g, _, _ in r["matches"]}:
                conf_mat[r["gts"][gi]["cls"]][n_cls] += 1
                per_cls[r["gts"][gi]["cls"]]["fn"] += 1

        records.append((ip, r))
        if idx % 40 == 0:
            print(f"  已处理 {idx}/{len(img_paths)}")

    # ---- 写 CSV ----
    with open(out_dir / "summary.csv", "w", newline="", encoding="utf-8-sig") as f:
        wcsv = csv.writer(f)
        wcsv.writerow(["class", "TP", "FN(漏检)", "FP-背景(凭空造框)",
                       "混淆-本类被认错", "混淆-他类被认成本类",
                       "GT总数", "Recall", "Precision"])
        for i in range(n_cls):
            s = per_cls[i]
            gt_total = s["tp"] + s["fn"] + s["conf_out"]
            n_pred = s["tp"] + s["conf_in"] + s["fp_bg"]
            rec = s["tp"] / gt_total if gt_total else 0.0
            prec = s["tp"] / n_pred if n_pred else 0.0
            wcsv.writerow([names[i], s["tp"], s["fn"], s["fp_bg"],
                           s["conf_out"], s["conf_in"], gt_total,
                           f"{rec:.3f}", f"{prec:.3f}"])

    labels = [names[i] for i in range(n_cls)] + ["background"]
    with open(out_dir / "confusion.csv", "w", newline="", encoding="utf-8-sig") as f:
        wcsv = csv.writer(f)
        wcsv.writerow(["GT\\Pred"] + labels)
        for i, row in enumerate(conf_mat):
            wcsv.writerow([labels[i]] + list(map(int, row)))

    # ---- 重点类错误可视化 ----
    focus_ids = {i for i, n in names.items() if n in focus}
    for cid in sorted(focus_ids):
        cname = names[cid]
        cdir = out_dir / cname
        cdir.mkdir(parents=True, exist_ok=True)

        def err_score(r, cid=cid):
            # cid=cid 显式绑定循环变量（B023）：不绑定的话闭包捕获的是最后一次
            # 迭代的 cid，一旦这个函数被存起来延后调用，全部结果都会算错。
            n = 0
            for gi, pi, _ in r["matches"]:
                if r["gts"][gi]["cls"] == cid or r["preds"][pi]["cls"] == cid:
                    if r["gts"][gi]["cls"] != r["preds"][pi]["cls"]:
                        n += 1
            n += sum(1 for c in r["fn_classes"] if c == cid)
            n += sum(1 for c in r["fp_classes"] if c == cid)
            return n

        # 每张图只算一次分数（原实现在推导式条件里又调用了一次）
        cand = sorted(((err_score(r), ip, r) for ip, r in records),
                      key=lambda t: -t[0])
        cand = [t for t in cand if t[0] > 0]
        for k, (score, ip, r) in enumerate(cand[:max_save], 1):
            vis = _draw(r["img"], r["gts"], r["preds"], r["matches"], names)
            save_image(cdir / f"{k:02d}_err{score}_{ip.stem}.jpg", vis)
        print(f"  [{cname}] 错误样本 {len(cand)} 张，已保存前 {min(max_save, len(cand))} 张")

    print(f"\n结果已写入: {out_dir}")
    return {"confusion": conf_mat, "per_cls": per_cls, "labels": labels}
