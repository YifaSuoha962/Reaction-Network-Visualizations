import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from joblib import Parallel, delayed
from tqdm import tqdm
from rdkit.DataStructs import TanimotoSimilarity
from mol_utils import add_result_args, get_rxn_fingerprint
from template_extractor import extract_from_reaction
from mpl_toolkits.mplot3d import Axes3D  # 如已在别处 import 可忽略
from typing import Optional, Tuple

# 你可以在 args 里加一个 n_jobs 参数；没有的话就默认用全部核

def load_csv_extract_templ(data_name, split):
    """
    统一读取某个 split（train / test）的 mol_freq.csv 和 targets.csv

    返回：
        mol_info_df, route_info_df
    """
    base_dir = os.path.join("data", split)
    res_csv = os.path.join(base_dir, "collected_rxns.csv")

    rxn_extractions = extract_templates_from_csv(res_csv)

    return rxn_extractions



# 假设 extract_from_reaction 已经定义好 / import 好
# from your_module import extract_from_reaction

def _process_one_row(idx, rxn_str,
                     pred_rxn_class=None,
                     pred_parent_class=None,
                     pred_rxn_name=None):
    """
    处理单条 csv 行：
      - 构造 reaction dict
      - 调用 extract_from_reaction
      - 过滤无效结果
      - 附加 label 信息
    返回：
      - res: dict 或 None
    """
    # 跳过空行或非法格式
    if not isinstance(rxn_str, str) or ">>" not in rxn_str:
        print(f"[WARN] invalid reaction string at row idx={idx}, content={rxn_str}")
        return None

    # 构造 reaction 字典（也可以直接传字符串，但这样 _id 更清晰）
    reactants_str, products_str = rxn_str.split(">>", 1)
    reaction = {
        "reactants": reactants_str,
        "products": products_str,
        "_id": idx,  # 用行号作为 ID，方便追踪
    }

    res = extract_from_reaction(reaction)

    # 有些情况下你的函数会返回 None 或只包含 {"reaction_id": ...}，可以视情况过滤
    if not isinstance(res, dict):
        print(f"[WARN] extraction failed for reaction_id={idx} (res is not dict)")
        return None

    # 只保留真正提取成功、包含模板的结果（按你的代码是有 'reaction_smarts'）
    if "reaction_smarts" not in res:
        rid = res.get("reaction_id", idx)
        print(f"[WARN] no reaction_smarts for reaction_id={rid}")
        return None

    # 附带 label 信息
    if pred_rxn_class is not None:
        res["pred_rxn_class"] = pred_rxn_class
    if pred_parent_class is not None:
        res["pred_parent_class"] = pred_parent_class
    if pred_rxn_name is not None:
        res["pred_rxn_name"] = pred_rxn_name

    return res


def extract_templates_from_csv(
    csv_path,
    rxn_col="reactants>reagents>production",
    n_jobs=-1,
):
    """
    从包含多条反应记录的 csv 中并行调用 extract_from_reaction，
    仅返回抽取成功的反应的模板及其对应的
      - reaction_smarts
      - pred_rxn_name
      - pred_rxn_class
      - pred_parent_class

    返回：List[dict]
    """
    df = pd.read_csv(csv_path)

    # 先判断这些列是否存在，避免在 worker 里做判断
    has_rxn_class = "pred_rxn_class" in df.columns
    has_parent_class = "pred_parent_class" in df.columns
    has_rxn_name = "pred_rxn_name" in df.columns

    # 准备并行任务参数列表
    tasks = []
    for idx, row in df.iterrows():
        rxn_str = row[rxn_col]
        pred_rxn_class = row["pred_rxn_class"] if has_rxn_class else None
        pred_parent_class = row["pred_parent_class"] if has_parent_class else None
        pred_rxn_name = row["pred_rxn_name"] if has_rxn_name else None

        tasks.append((idx, rxn_str, pred_rxn_class, pred_parent_class, pred_rxn_name))

    # 并行执行
    raw_results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_process_one_row)(*t)
        for t in tqdm(tasks, desc="Extracting templates (parallel)")
    )

    # 过滤掉 None，并且只保留需要的字段
    results_all = []
    for r in raw_results:
        if r is None:
            continue
        results_all.append({
            "reaction_smarts": r["reaction_smarts"],
            "pred_rxn_name": r.get("pred_rxn_name"),
            "pred_rxn_class": r.get("pred_rxn_class"),
            "pred_parent_class": r.get("pred_parent_class"),
        })

    return results_all



def _get_rxn_fp_safe(rxn):
    # 可选：加一点异常捕获，防止某个反应坏掉把整个并行崩了
    try:
        return get_rxn_fingerprint(rxn)
    except Exception as e:
        # 你也可以在这里 print 一下 rxn 或记录日志
        return None

def visualize_embeddings(args,
                         embeddings,
                         labels,
                         n_components: int = 2,
                         random_state: int = 42):
    """
    可视化高维嵌入向量

    参数:
        embeddings: 形状为 [batch_size, hidden_size] 的PyTorch张量
        labels: 形状为 [batch_size] 的标签张量
        n_components: 降维后的维度 (2或3)
        title: 图像标题
        random_state: 随机种子
        figsize: 图像大小
    """
    # 转换数据到CPU和numpy
    embeddings_np = np.asarray(embeddings)
    labels_np = np.asarray(labels)

    # t-SNE降维
    # tsne = TSNE(n_components=n_components,
    #             random_state=random_state,
    #             perplexity=20,          # 小于默认值30，增强局部结构
    #             learning_rate=500,      # 增大步长强化分离
    #             early_exaggeration=24,  # 加大早期放大因子
    #             n_iter=1000,            # 增加迭代次数
    #             init='pca')

    tsne = TSNE(
        n_components=n_components,
        random_state=random_state,
        perplexity=20,  # 减小值增强局部紧凑性 (建议范围5-15)
        learning_rate=200,  # 减小步长使收敛更稳定 (建议范围100-300)
        early_exaggeration=8,  # 减小早期放大因子 (建议范围8-16)
        n_iter=500,  # 增加迭代次数确保充分收敛
        init='pca',  # 保持PCA初始化
        metric='cosine',
        angle=0.3  # 减小角度提高精度(0.2-0.5)
    )

    embeddings_tsne = tsne.fit_transform(embeddings_np)

    # 可视化
    plt.figure(figsize=(8, 6))

    # 离散化表示标签
    unique_labels = np.unique(labels_np)
    num_classes = len(unique_labels)

    # 为每个类别创建自定义颜色映射
    cmap = plt.get_cmap('viridis', num_classes)

    if n_components == 2:
        # 为每个类别单独绘制散点
        for label in unique_labels:
            mask = labels_np == label
            plt.scatter(embeddings_tsne[mask, 0],
                        embeddings_tsne[mask, 1],
                        color=cmap(label),
                        alpha=0.5,
                        label=f'{label}',
                        linewidth=0.3)
        # edgecolor = sns.dark_palette(cmap(label), n_colors=1)[0],   # 'w',  # 白色边缘增强区分度
        #
        # 添加离散图例
        plt.legend(title='Latent class $c$',
                   bbox_to_anchor=(0.99, 0.98),  # 右上角内部坐标 (x,y)
                   loc='upper right',  # 锚点定位到右上
                   borderaxespad=0.4,  # 增加边框间距
                   frameon=True,
                   fontsize=12,
                   title_fontsize=14,
                   ncol=2,
                   handletextpad=0.4,
                   columnspacing=0.7)

        plt.xticks([])  # 移除x轴刻度
        plt.yticks([])  # 移除y轴刻度

    elif n_components == 3:
        ax = plt.axes(projection='3d')
        for label in unique_labels:
            mask = labels_np == label
            ax.scatter3D(embeddings_tsne[mask, 0],
                         embeddings_tsne[mask, 1],
                         embeddings_tsne[mask, 2],
                         color=cmap(label),
                         alpha=0.4,
                         label=f'{label}',
                         edgecolor='w',
                         linewidth=0.3)

        ax.legend(title='Latent class $c$',
                  bbox_to_anchor=(1.1, 0.9),
                  loc='upper left',
                  fontsize=12,
                  title_fontsize=14,
                  ncol=2,  # 2列布局
                  handletextpad=0.5,  # 调整文本与图例标记的间距
                  columnspacing=0.8)

        ax.set_xticks([])  # 移除3D图的x轴刻度
        ax.set_yticks([])  # 移除3D图的y轴刻度
        ax.set_zticks([])  # 移除3D图的z轴刻度
    else:
        raise ValueError("n_components 只能是2或3")

    # 添加图例和标题
    # plt.colorbar(scatter, label='Class Label')
    # plt.title(f'{args.dataset_name}')
    # plt.xlabel('t-SNE Dimension 1')
    # plt.ylabel('t-SNE Dimension 2')

    # 显示图形
    save_dir = 'tsne-plot'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, f'{args.data_mode}-{n_components}dim.svg')
    plt.show()
    save_path = os.path.join(save_dir, f'{args.data_mode}-{n_components}dim.pdf')
    plt.savefig(save_path)


def reduce_and_tsne_plot(embeddings,
    labels, n_components: int = 2,
    max_points: int = 20000,
    random_state: int = 42,
    metric: str = "euclidean",
    type_name = 'Pure_FingerPrint'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    对高维向量做自动下采样 + t-SNE / UMAP 降维并可视化。

    参数
    ----
    embeddings : np.ndarray | torch.Tensor | list
        形状 [N, D] 的向量。
    labels : np.ndarray | torch.Tensor | list
        形状 [N] 的标签（可为数值或字符串）。
    method : "tsne" or "umap"
        使用 t-SNE 还是 UMAP。
    n_components : int
        降维后的维度（通常 2 或 3）。
    max_points : int
        若样本数 N > max_points，则随机下采样到 max_points 个点。
    random_state : int
        随机种子。
    metric : str
        距离度量，tsne 建议 "euclidean"，umap 常用 "euclidean" 或 "cosine"。
    save_dir : str | None
        若不为 None，则将图像保存到该目录下。
    save_name : str | None
        图像文件名（不含后缀）；若为 None，则自动根据 method 命名。

    返回
    ----
    emb_low : np.ndarray
        降维后的向量，形状 [M, n_components]（M 可能 < N）。
    used_idx : np.ndarray
        实际参与降维的样本在原始数组中的索引，形状 [M]。
    """
    # 1. 转成 numpy
    # import torch
    #
    # if isinstance(embeddings, torch.Tensor):
    #     embeddings_np = embeddings.detach().cpu().numpy()
    # else:
    #     embeddings_np = np.asarray(embeddings)
    #
    # if isinstance(labels, torch.Tensor):
    #     labels_np = labels.detach().cpu().numpy()
    # else:
    #     labels_np = np.asarray(labels)

    embeddings_np = np.asarray(embeddings)
    labels_np = np.asarray(labels)

    assert embeddings_np.shape[0] == labels_np.shape[0], "embeddings 与 labels 数量不一致"

    N = embeddings_np.shape[0]
    rng = np.random.RandomState(random_state)

    # 2. 自动下采样
    if N > max_points:
        used_idx = rng.choice(N, size=max_points, replace=False)
    else:
        used_idx = np.arange(N)

    X = embeddings_np[used_idx]
    y = labels_np[used_idx]  # ✅ 只保留被采样的标签

    # 3. 降维
    tsne = TSNE(
            n_components=n_components,
            random_state=random_state,
            perplexity=20,
            learning_rate=200,
            early_exaggeration=8,
            n_iter=500,
            init="pca",
            metric=metric,
            angle=0.3,
        )
    X_low = tsne.fit_transform(X)

    # 可视化
    plt.figure(figsize=(8, 6))

    unique_labels = np.unique(y)
    num_classes = len(unique_labels)
    cmap = plt.get_cmap("viridis", num_classes)

    if n_components == 2:
        # 为每个类别单独绘制散点
        for label in unique_labels:
            mask = (y == label)
            plt.scatter(X_low[mask, 0],
                        X_low[mask, 1],
                        color=cmap(label),
                        alpha=0.5,
                        label=f'{label}',
                        linewidth=0.3)
        # edgecolor = sns.dark_palette(cmap(label), n_colors=1)[0],   # 'w',  # 白色边缘增强区分度
        #
        # 添加离散图例
        plt.legend(title='rxn classes',
                   bbox_to_anchor=(0.99, 0.98),  # 右上角内部坐标 (x,y)
                   loc='upper right',  # 锚点定位到右上
                   borderaxespad=0.4,  # 增加边框间距
                   frameon=True,
                   fontsize=12,
                   title_fontsize=14,
                   ncol=2,
                   handletextpad=0.4,
                   columnspacing=0.7)

        plt.xticks([])  # 移除x轴刻度
        plt.yticks([])  # 移除y轴刻度

    elif n_components == 3:
        ax = plt.axes(projection='3d')
        for label in unique_labels:
            mask = (y == label)
            ax.scatter3D(X_low[mask, 0],
                         X_low[mask, 1],
                         X_low[mask, 2],
                         color=cmap(label),
                         alpha=0.4,
                         label=f'{label}',
                         edgecolor='w',
                         linewidth=0.3)

        ax.legend(title='rxn classes',
                  bbox_to_anchor=(1.1, 0.9),
                  loc='upper left',
                  fontsize=12,
                  title_fontsize=14,
                  ncol=2,  # 2列布局
                  handletextpad=0.5,  # 调整文本与图例标记的间距
                  columnspacing=0.8)

        ax.set_xticks([])  # 移除3D图的x轴刻度
        ax.set_yticks([])  # 移除3D图的y轴刻度
        ax.set_zticks([])  # 移除3D图的z轴刻度
    else:
        raise ValueError("n_components 只能是2或3")

    # 保存
    save_dir = 'tsne-plot'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, f'{args.data_mode}-{type_name}-{n_components}dim.svg')
    plt.show()
    save_path = os.path.join(save_dir, f'{args.data_mode}-{type_name}-{n_components}dim.pdf')
    plt.savefig(save_path)

        # if save_dir is not None:
        #     os.makedirs(save_dir, exist_ok=True)
        #
        # svg_path = os.path.join(save_dir, save_name + ".svg")
        # pdf_path = os.path.join(save_dir, save_name + ".pdf")
        # plt.savefig(svg_path, bbox_inches="tight")
        # plt.savefig(pdf_path, bbox_inches="tight")
        # plt.show()

    return X_low, used_idx



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
    rxn_class_list = [d["pred_parent_class"] for d in rxn_extractions]

    """TODO: 照着hiclr的ipynb做一个基于mol transformer embedding的可视化"""
    rxn_fps = Parallel(n_jobs=args.n_jobs, backend="loky")(
         delayed(get_rxn_fingerprint)(rxn) for rxn in tqdm(rxn_smarts_list, desc="Computing reaction fingerprints")
    )
    test_sim = TanimotoSimilarity(rxn_fps[0], rxn_fps[1])
    print(f"demo similarity = {test_sim}")
    # assert 1 == 2
    #
    # print(f"pass! ")
    #
    # visualize_embeddings(args, rxn_fps, rxn_class_list, n_components=args.tsne_dim)

    reduce_and_tsne_plot(rxn_fps,
    rxn_class_list, n_components= args.tsne_dim,
    max_points=10000,
    random_state=42,
    metric= "euclidean",)
