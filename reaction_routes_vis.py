import os
import matplotlib.pyplot as plt
import networkx as nx

from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from rdkit import Chem
from rdkit.Chem import Draw
from mol_utils import canonicalize_smiles

def smi_to_img_pil(smi, size=(200, 200)):
    """
    基于 RDKit 生成分子图（PIL.Image），
    等价于你原来的 smi_to_img 的“画图逻辑”，只是换成返回 PIL，
    方便直接嵌入 Matplotlib。
    """
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    img = Draw.MolToImage(mol, size=size)
    return img


def hierarchy_pos(G, root, width=1.0, vert_gap=0.3, vert_loc=0.0, xcenter=0.5):
    """
    给有向树/无向树生成一个自顶向下(hierarchy)布局坐标。
    这个实现不依赖 Graphviz，只用 NetworkX 自己。
    """
    def _hierarchy_pos(G, root, left, right, vert_loc, pos):
        children = list(G.successors(root))
        if not children:  # 叶子
            pos[root] = ((left + right) / 2.0, vert_loc)
        else:
            dx = (right - left) / max(len(children), 1)
            next_left = left
            pos[root] = ((left + right) / 2.0, vert_loc)
            for c in children:
                next_right = next_left + dx
                _hierarchy_pos(G, c, next_left, next_right, vert_loc - vert_gap, pos)
                next_left = next_right
        return pos

    return _hierarchy_pos(G, root, 0, width, vert_loc, {})


def build_route_tree_for_product_nx_with_imgs(product_smiles: str,
                                              product_info: dict,
                                              out_path: str = "route_tree_example_nx.png",
                                              img_size=(200, 200),
                                              zoom=0.35):
    """
    使用 NetworkX + Matplotlib + RDKit 分子图，可视化合成树：
    - 根节点：product_smiles
    - 中间节点：每条 retro_routes 中 '>>' 左侧按 '.' 拆分的 SMILES
    - 叶子节点：materials 中的所有分子
    节点以“分子结构图像”形式展示，而不是纯文本 SMILES。
    """

    G = nx.DiGraph()

    # 根节点用一个特殊 id
    root_id = "ROOT"
    G.add_node(root_id, label=product_smiles, kind="product")

    # label 去重用
    label2id = {}
    node_counter = 0

    def get_node_id(label: str, kind: str):
        nonlocal node_counter
        key = (kind, label)
        if key in label2id:
            return label2id[key]
        node_counter += 1
        node_id = f"{kind}_{node_counter}"
        G.add_node(node_id, label=label, kind=kind)
        label2id[key] = node_id
        return node_id

    # ==== 构图：跟你原来的逻辑一致 ====
    for sub_key, tree_obj in product_info.items():
        # 无论原来是 "1" 还是 1，这里统一转成字符串判断
        key_str = str(sub_key)
        if not key_str.isdigit():
            continue

        retro_routes = tree_obj.get("retro_routes", [])
        materials = tree_obj.get("materials", [])

        for route_idx, route in enumerate(retro_routes):
            prev_nodes = [root_id]

            for step_idx, rxn_smi in enumerate(route):
                if ">>" not in rxn_smi:
                    continue
                left, right = rxn_smi.split(">>", 1)
                left_mols = [m for m in left.split(".") if m.strip()]

                check_smi = left_mols[0]
                check_smi = canonicalize_smiles(check_smi)
                if check_smi is None:
                    print("fail to parse SMILES")
                    assert 1 == 2

                # 如果你需要 canonicalize，可以保留这段；要防止报错的话最好加 try/except
                left_mols_back = []
                for mol in left_mols:
                    try:
                        cano_mol = canonicalize_smiles(mol)
                    except Exception:
                        cano_mol = mol  # 兜底：出问题就用原串
                    left_mols_back.append(cano_mol)
                left_mols = left_mols_back

                next_prev_nodes = []
                for prev_id in prev_nodes:
                    for lm in left_mols:
                        mid_id = get_node_id(lm, kind="intermediate")
                        edge_label = f"{key_str}.{route_idx}.{step_idx}"
                        G.add_edge(prev_id, mid_id, label=edge_label)
                        next_prev_nodes.append(mid_id)

                prev_nodes = sorted(set(next_prev_nodes))

            # route 结束后，最后一层中间节点指向 materials
            for prev_id in prev_nodes:
                for mat in materials:
                    mat_id = get_node_id(mat, kind="material")
                    G.add_edge(prev_id, mat_id)

    # print(product_info.keys())
    # for k in product_info.keys():
    #     print(k, type(k))
    #
    # print("Num nodes:", G.number_of_nodes())
    # print("Num edges:", G.number_of_edges())

    # assert 1 == 2

    # ==== 开始画图 ====
    pos = hierarchy_pos(G, root=root_id, vert_gap=0.8)

    fig, ax = plt.subplots(figsize=(10, 8))

    # 先画边（为了让节点图盖在上面）
    for u, v in G.edges():
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        ax.annotate(
            "",
            xy=(x2, y2),
            xytext=(x1, y1),
            arrowprops=dict(arrowstyle="->", lw=1),
        )

    # 设置坐标轴范围，让所有节点都在视野里
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    pad = 0.5
    ax.set_xlim(min(xs) - pad, max(xs) + pad)
    ax.set_ylim(min(ys) - pad, max(ys) + pad)
    ax.invert_yaxis()  # 可选：让树从上往下

    # 预先缓存每个 SMILES 对应的一张分子图（避免重复计算）
    smiles2img = {}

    # 再画节点图像
    for n, d in G.nodes(data=True):
        label = d.get("label", "")
        kind = d.get("kind", "intermediate")

        # 根节点 / 其他节点同样用 SMILES 画分子
        smi = label

        if smi not in smiles2img:
            img = smi_to_img_pil(smi, size=img_size)
            smiles2img[smi] = img
        else:
            img = smiles2img[smi]

        if img is None:
            # 如果 SMILES 解析失败，就画成一个方框+文字兜底
            x, y = pos[n]
            ax.scatter([x], [y], s=200, c="gray")
            ax.text(x, y, smi, fontsize=6, ha="center", va="center")
            continue

        x, y = pos[n]
        imagebox = OffsetImage(img, zoom=zoom)

        # 根据 kind 控制是否加边框 / 颜色（简单一点，用 frameon=True + edgecolor）
        if kind == "product":
            frameon = True
        elif kind == "material":
            frameon = True
        else:
            frameon = False

        ab = AnnotationBbox(
            imagebox,
            (x, y),
            frameon=frameon,
            pad=0.1,
        )
        ax.add_artist(ab)

    ax.set_axis_off()
    plt.tight_layout()

    # 确保目录存在
    out_dir = os.path.dirname(out_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[INFO] Route tree with molecule images saved to: {out_path}")


if __name__ == "__main__":

    # 截取自 test_dataset.json
    data = {
        "O=C(OC1CCCC1)c1ccc(OCc2ccccc2)cc1": {
            "depth": 2,
            "num_reaction_trees": 2,
            "1": {
                "retro_routes": [
                    [
                        "[CH:1]1([O:6][C:19]([c:18]2[cH:17][cH:16][c:15]([O:14][CH2:7][c:8]3[cH:9][cH:10][cH:11][cH:12][cH:13]3)[cH:23][cH:22]2)=[O:20])[CH2:2][CH2:3][CH2:4][CH2:5]1>>[CH:1]1([OH:6])[CH2:2][CH2:3][CH2:4][CH2:5]1.Cl[C:19]([c:18]1[cH:17][cH:16][c:15]([O:14][CH2:7][c:8]2[cH:9][cH:10][cH:11][cH:12][cH:13]2)[cH:23][cH:22]1)=[O:20]",
                        "[C:1](=[O:2])([c:4]1[cH:5][cH:6][c:7]([O:8][CH2:9][c:10]2[cH:11][cH:12][cH:13][cH:14][cH:15]2)[cH:16][cH:17]1)[Cl:20]>>O[C:1](=[O:2])[c:4]1[cH:5][cH:6][c:7]([O:8][CH2:9][c:10]2[cH:11][cH:12][cH:13][cH:14][cH:15]2)[cH:16][cH:17]1.O=S(Cl)[Cl:20]"
                    ]
                ],
                "materials": [
                    "O=S(Cl)Cl",
                    "O=C(O)c1ccc(OCc2ccccc2)cc1",
                    "OC1CCCC1"
                ]
            },
            "2": {
                "retro_routes": [
                    [
                        "[CH:1]1([O:6][C:19]([c:18]2[cH:17][cH:16][c:15]([O:14][CH2:7][c:8]3[cH:9][cH:10][cH:11][cH:12][cH:13]3)[cH:23][cH:22]2)=[O:20])[CH2:2][CH2:3][CH2:4][CH2:5]1>>[CH:1]1([OH:6])[CH2:2][CH2:3][CH2:4][CH2:5]1.Cl[C:19]([c:18]1[cH:17][cH:16][c:15]([O:14][CH2:7][c:8]2[cH:9][cH:10][cH:11][cH:12][cH:13]2)[cH:23][cH:22]1)=[O:20]",
                        "[CH2:1]([c:2]1[cH:3][cH:4][cH:5][cH:6][cH:7]1)[O:8][c:9]1[cH:10][cH:11][c:12]([C:13](=[O:14])[Cl:21])[cH:16][cH:17]1>>O[C:13]([c:12]1[cH:11][cH:10][c:9]([O:8][CH2:1][c:2]2[cH:3][cH:4][cH:5][cH:6][cH:7]2)[cH:17][cH:16]1)=[O:14].O=C(Cl)C(=O)[Cl:21]"
                    ]
                ],
                "materials": [
                    "O=C(Cl)C(=O)Cl",
                    "O=C(O)c1ccc(OCc2ccccc2)cc1",
                    "OC1CCCC1"
                ]
            }
        }
    }

    product_smiles = "O=C(OC1CCCC1)c1ccc(OCc2ccccc2)cc1"
    product_info = data[product_smiles]

    build_route_tree_for_product_nx_with_imgs(
        product_smiles,
        product_info,
        out_path="route_tree_example_nx_with_imgs.png"
    )
