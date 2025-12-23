import os
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

from mol_utils import canonicalize_smiles  # 如果不需要标准化，可以去掉这行和相关调用
from mol_utils import add_result_args

def build_reaction_network(
    collected_rxns_csv: str,
    mol_freq_csv: str,
    rxn_col: str = "reactants>reagents>production",
):
    """
    根据:
      - collected_rxns.csv: 其中 'reactants>reagents>production' 列存反应字符串
      - mol_freq.csv: 其中 'Mol_SMILES' 列存所有分子
    构建一个有向图:
      节点: 所有分子（Mol_SMILES）
      边:   反应左侧分子(reactants+reagents) 指向 右侧分子(product)
    """
    # 读入分子列表，建立节点
    mol_df = pd.read_csv(mol_freq_csv)
    all_mols = mol_df["Mol_SMILES"].astype(str).unique().tolist()
    mol_set = set(all_mols)

    G = nx.DiGraph()
    # 先把所有节点加进图里（以免某些分子只在 product 侧）
    for smi in all_mols:
        G.add_node(smi)

    # 读入反应记录
    rxn_df = pd.read_csv(collected_rxns_csv)

    for _, row in rxn_df.iterrows():
        rxn_str = str(row[rxn_col])
        if not rxn_str:
            continue

        left = None
        right = None

        # 兼容两种写法:
        #   1) 原始: reactants>reagents>production
        #   2) 你前面处理过的: reactants.reagents>>production
        if ">>" in rxn_str:
            # 形如 "A.B>>C"
            left, right = rxn_str.split(">>", 1)
        else:
            parts = rxn_str.split(">")
            if len(parts) == 3:
                reactants, reagents, product = parts
                left = reactants
                if reagents:  # 有时可能是空
                    left = reactants + "." + reagents
                right = product
            else:
                # 格式异常，跳过
                continue

        if right is None:
            continue

        # 左侧所有分子
        left_mols = [m.strip() for m in left.split(".") if m.strip()]
        # 右侧所有分子（一般是一个，如果有多个也一起连边）
        right_mols = [m.strip() for m in right.split(".") if m.strip()]

        # 可选：把 SMILES 做 canonicalize，使和 mol_freq 中一致
        left_mols_can = []
        for m in left_mols:
            try:
                cm = canonicalize_smiles(m)
            except Exception:
                cm = m
            left_mols_can.append(cm)

        right_mols_can = []
        for m in right_mols:
            try:
                cm = canonicalize_smiles(m)
            except Exception:
                cm = m
            right_mols_can.append(cm)

        # 建边：左侧每个分子 -> 右侧每个分子
        for lm in left_mols_can:
            # 如果某个分子不在 mol_set，可选择仍然加入 G（这里就加入）
            if lm not in G:
                G.add_node(lm)
            for rm in right_mols_can:
                if rm not in G:
                    G.add_node(rm)
                G.add_edge(lm, rm)

    return G


def plot_reaction_network(
    G: nx.DiGraph,
    out_path: str = "reaction_network.png",
    max_nodes_to_label: int = 50,
    node_size: int = 80,
):
    """
    使用 NetworkX + Matplotlib 绘制整体有向关系图，不显示分子图像，只显示节点与箭头。

    参数
    ----
    G : DiGraph
        build_reaction_network 构建的图
    out_path : str
        输出图片路径
    max_nodes_to_label : int
        最多显示多少个节点的 SMILES 文本（太多会糊）
    node_size : int
        节点大小
    """
    # spring_layout 更适合一般网络
    pos = nx.spring_layout(G, k=0.25, iterations=100, seed=42)

    plt.figure(figsize=(12, 10))

    # 画节点
    nx.draw_networkx_nodes(
        G,
        pos,
        node_size=node_size,
        node_color="tab:blue",
        alpha=0.8,
    )

    # 画有向边
    nx.draw_networkx_edges(
        G,
        pos,
        arrowstyle="->",
        arrowsize=10,
        width=0.6,
        alpha=0.7,
    )

    # 只给一部分节点加 label，避免太挤
    labels = {}
    for i, n in enumerate(G.nodes()):
        if i >= max_nodes_to_label:
            break
        labels[n] = n  # 用 SMILES 作为标签
    nx.draw_networkx_labels(
        G,
        pos,
        labels=labels,
        font_size=6,
    )

    plt.axis("off")
    plt.tight_layout()

    out_dir = os.path.dirname(out_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[INFO] Reaction network saved to: {out_path}")


if __name__ == "__main__":
    args = add_result_args()
    # 根据你的目录结构自行改路径
    collected_rxns_csv = f"data/{args.data_mode}/collected_rxns.csv"
    mol_freq_csv = f"data/{args.data_mode}/mol_freq.csv"

    G = build_reaction_network(
        collected_rxns_csv=collected_rxns_csv,
        mol_freq_csv=mol_freq_csv,
        rxn_col="reactants>reagents>production",
    )
    print("Num nodes:", G.number_of_nodes())
    print("Num edges:", G.number_of_edges())

    plot_reaction_network(
        G,
        out_path="reaction_network.png",
        max_nodes_to_label=60,   # 可调小一点
        node_size=80,
    )
