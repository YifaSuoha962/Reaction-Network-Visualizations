import random
from collections import defaultdict
import os
import seaborn as sns

import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from tqdm import tqdm
from mol_utils import add_result_args, get_rxn_fingerprint
from template_extractor import extract_from_reaction
from typing import Optional, Tuple
from tsne_rxn_classes import load_csv_extract_templ
from rdkit import DataStructs


def compute_similarity_matrix(fps_list):
    """
    给定 reaction fingerprint 列表，计算 pairwise Tanimoto 相似度矩阵。
    fps_list : List[ExplicitBitVect]
    返回:
        sim_mat : np.ndarray, shape (M, M)
    """
    M = len(fps_list)
    sim_mat = np.zeros((M, M), dtype=float)

    for i in range(M):
        sim_mat[i, i] = 1.0
        if i + 1 < M:
            # TanimotoSimilarity: http://bookshadow.com/weblog/2014/06/07/tanimoto-similarity-and-distance/
            sims = DataStructs.BulkTanimotoSimilarity(fps_list[i], fps_list[i + 1:])
            sim_mat[i, i + 1:] = sims
            sim_mat[i + 1:, i] = sims

    return sim_mat


def plot_similarity_heatmap(sim_mat, class_order, sample_per_class=10, figsize=(8, 7)):
    """
    画 (num_classes * sample_per_class) x (num_classes * sample_per_class) 的相似度 heatmap，
    并在坐标轴上标出每个“反应类型块”的标签。
    """
    num_classes = len(class_order)
    M = sim_mat.shape[0]
    assert M == num_classes * sample_per_class, \
        f"M={M} 和 num_classes*sample_per_class={num_classes * sample_per_class} 不一致，检查采样逻辑。"

    plt.figure(figsize=figsize)

    ax = sns.heatmap(
        sim_mat,
        cmap="viridis",
        square=True,
        cbar=True,
        xticklabels=False,
        yticklabels=False,
        vmin=0.0,
        vmax=1.0,
    )
    plt.title("Reaction fingerprint similarity (Tanimoto)")

    # 每个类别对应一个连续 block，在轴上标出 block 的中心位置
    centers = [i * sample_per_class + (sample_per_class / 2.0) for i in range(num_classes)]
    ax.set_xticks(centers)
    ax.set_yticks(centers)
    ax.set_xticklabels(class_order, rotation=90)
    ax.set_yticklabels(class_order)

    plt.tight_layout()
    plt.show()


def build_and_plot_reaction_similarity_from_fps(
    args,
    rxn_fps,
    rxn_class_list,
    sample_per_class=10,
    max_classes=None,
    random_state=42,
):
    """
    1. 根据 rxn_class_list 对 rxn_fps 分组
    2. 每个反应大类下采样 sample_per_class 个 reaction FP
    3. 计算 pairwise Tanimoto 相似度矩阵
    4. 画 heatmap

    参数
    ----
    rxn_fps : List[ExplicitBitVect 或 None]
        compute_rxn_fps 得到的指纹列表（与原始反应一一对应）
    rxn_class_list : List
        与 rxn_fps 对应的反应大类标签列表，比如 pred_parent_class
    sample_per_class : int
        每个类别下采样多少个反应
    max_classes : int or None
        最多保留多少个类别；None 表示用所有满足采样条件的类别
    random_state : int
        随机种子
    save_prefix : str or None
        如果不为 None，则额外保存一份 heatmap 到
        f"{save_prefix}_heatmap.png" / f"{save_prefix}_heatmap.pdf"

    返回
    ----
    sim_mat : np.ndarray
        相似度矩阵
    sampled_class_order : List
        实际参与采样的类别顺序
    """
    rng = random.Random(random_state)

    # 1) 先把 None / 无标签 的样本过滤掉
    assert len(rxn_fps) == len(rxn_class_list), "rxn_fps 和 rxn_class_list 长度必须一致"

    buckets = defaultdict(list)  # label -> list of fps
    for fp, lab in zip(rxn_fps, rxn_class_list):
        if fp is None:
            continue
        if lab is None or (isinstance(lab, float) and np.isnan(lab)):
            continue
        buckets[lab].append(fp)

    # 只保留样本数 >= sample_per_class 的类别
    eligible_labels = [lab for lab, lst in buckets.items() if len(lst) >= sample_per_class]
    eligible_labels = sorted(eligible_labels)  # 固定顺序

    if max_classes is not None:
        eligible_labels = eligible_labels[:max_classes]

    if len(eligible_labels) == 0:
        raise ValueError("没有任何类别的样本数 >= sample_per_class，无法构建相似度矩阵。")

    sampled_fps = []
    sampled_class_order = []

    for lab in eligible_labels:
        fps_list = buckets[lab]
        chosen = rng.sample(fps_list, sample_per_class)
        sampled_fps.extend(chosen)
        sampled_class_order.append(lab)

    print(f"[INFO] Selected {len(sampled_class_order)} classes, "
          f"{len(sampled_fps)} reactions (fps) in total.")

    # 2) 计算相似度矩阵
    sim_mat = compute_similarity_matrix(sampled_fps)

    # 3) 画图
    plot_similarity_heatmap(
        sim_mat,
        class_order=sampled_class_order,
        sample_per_class=sample_per_class,
        figsize=(8, 7),
    )

    save_prefix = "heatmap"
    if not os.path.exists(save_prefix):
        os.makedirs(save_prefix)
    # 可选：保存图像
    png_path = os.path.join(save_prefix, f"{args.data_mode}_similarities.svg")
    pdf_path = os.path.join(save_prefix, f"{args.data_mode}_similarities.pdf")
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    print(f"[INFO] Heatmap saved to {png_path} and {pdf_path}")

    return sim_mat, sampled_class_order




if __name__ == "__main__":
    args = add_result_args()

    """1. load rxn data"""

    rxn_extractions = load_csv_extract_templ(args.data_name, "train")
    print(f"Successfully extracted {len(rxn_extractions)} rxn templates")
    # ori_rxns = ori_df['reactants>reagents>production'].to_list()
    # rxn_types = ori_df['pred_parent_class'].to_list()

    # 获取表示化学反应的数值向量
    # rxn_pairs = []
    # for rxn in ori_rxns:
    #     rxn_fp = get_rxn_fingerprint(rxn)
    #     rxn_pairs.append(rxn_fp)

    """TODO: 改成计算 reaction template 的 fingerprint"""
    rxn_smarts_list = [d["reaction_smarts"] for d in rxn_extractions]
    rxn_sup_class_list = [d["pred_parent_class"] for d in rxn_extractions]
    rxn_class_list = [d["pred_rxn_class"] for d in rxn_extractions]
    """TODO: 照着hiclr的ipynb做一个基于mol transformer embedding的可视化"""
    rxn_fps = Parallel(n_jobs=args.n_jobs, backend="loky")(
         delayed(get_rxn_fingerprint)(rxn) for rxn in tqdm(rxn_smarts_list, desc="Computing reaction fingerprints")
    )

    sim_mat, class_order = build_and_plot_reaction_similarity_from_fps(
        args,
        rxn_fps=rxn_fps,
        rxn_class_list=rxn_class_list,
        sample_per_class=10,  # 每类下采样 10 个
        max_classes=10,  # 例如最多取 10 个大类，可按需调整或设成 None
        random_state=2025,
    )