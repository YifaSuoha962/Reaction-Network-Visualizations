import os
from rdkit.Chem.Draw import IPythonConsole, rdMolDraw2D  # Needed to show molecules
from rdkit.Chem.Draw.MolDrawing import MolDrawing, DrawingOptions  # Only needed if modifying defaults
from IPython.display import SVG
from io import BytesIO
from PIL import Image
from cairosvg import svg2png

from rdkit.Chem import AllChem, rdChemReactions
from rdkit import Chem, DataStructs
from rdkit.DataStructs import TanimotoSimilarity

opts = DrawingOptions()
opts.includeAtomNumbers = True
opts.includeAtomNumbers = True
opts.bondLineWidth = 2.8

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import argparse
# 命令行
def add_result_args():
    parser = argparse.ArgumentParser("result")

    parser.add_argument(
        "--n_jobs",
        help="cpu workers",
        type=int,
        default=-1,
    )
    parser.add_argument(
        "--data_name",
        help="Dataset name",
        choices=["retro-bench", "Schneider-50k"],
        type=str,
        default="retro-bench",
    )
    parser.add_argument(
        "--data_mode",
        help="Dataset mode",
        choices=["train", "test"],
        type=str,
        default="test",
    )
    parser.add_argument(
        "--tsne_label",
        help="Granularity of reaction class label",
        choices=["pred_parent_class", "pred_rxn_class"],
        type=str,
        default="pred_parent_class",
    )
    parser.add_argument("--tsne_dim",
        help="t-sne spatial dimension",
        choices=[2, 3],
        type=int,
        default=2,)

    # heatmap 维度，是平面还是三维柱状图

    args = parser.parse_args()
    return args


# 绘制分子结构
def generate_image(mol, highlight_atoms=None, highlight_bonds=None, atomColors=None, bondColors=None, radii=None, size=None, output=None, isNumber=False):
    print("Highlight Atoms:", highlight_atoms)
    print("Highlight Bonds:", highlight_bonds)
    print("Atom Colors:", atomColors)
    print("Bond Colors:", bondColors)

    image_data = BytesIO()
    view = rdMolDraw2D.MolDraw2DSVG(size[0], size[1])
    tm = rdMolDraw2D.PrepareMolForDrawing(mol)

    option = view.drawOptions()
    if isNumber:
        for atom in mol.GetAtoms():
            option.atomLabels[atom.GetIdx()] = atom.GetSymbol() + str(atom.GetIdx() + 1)

    view.DrawMolecule(tm,
                      highlightAtoms=highlight_atoms,
                      highlightBonds=highlight_bonds,
                      highlightAtomColors={atom: atomColors for atom in highlight_atoms},
                      highlightBondColors={bond: bondColors for bond in highlight_bonds},
                      highlightAtomRadii={atom: radii for atom in highlight_atoms})

    view.FinishDrawing()
    svg = view.GetDrawingText()
    SVG(svg.replace('svg:', ''))
    svg2png(bytestring=svg, write_to=output)
    img = Image.open(output)
    img.save(image_data, format='PNG')

    return image_data


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


def smi_to_img(save_dir, smi):
    svg_draw = rdMolDraw2D.MolDraw2DSVG(300, 300)
    svg_draw.ClearDrawing()
    rdMolDraw2D.PrepareAndDrawMolecule(svg_draw, Chem.MolFromSmiles(smi))
    svg_draw.FinishDrawing()
    with open(os.path.join(save_dir, f'{smi}.svg'), 'w') as f:
        f.write(svg_draw.GetDrawingText())


# 获取反应指纹
def get_rxn_fingerprint(rxn: str):
    """最好是带atom-map的"""
    rxn_ent = AllChem.ReactionFromSmarts(rxn, useSmiles=True)
    rxn_fp = rdChemReactions.CreateStructuralFingerprintForReaction(rxn_ent)
    return rxn_fp


def highlight_subtructures(smi_p, smi_r, save_dir):
    """以产物 p 为基准，和反应物 r 比较"""
    # get mol
    demo_mol = Chem.MolFromSmiles(smi_p)
    mol_r = Chem.MolFromSmarts(smi_r)       #
    # common atoms
    comm_atoms = demo_mol.GetSubstructMatches(mol_r)
    # cannot find matched substructure
    if comm_atoms is not None:

        comm_atoms = comm_atoms[0]
        # comm_bonds
        comm_bonds = set()

        # 获取与共同原子相连的边
        for atom_idx in comm_atoms:
            for neighbor in demo_mol.GetAtomWithIdx(atom_idx).GetNeighbors():
                bond = demo_mol.GetBondBetweenAtoms(atom_idx, neighbor.GetIdx())
                comm_bonds.add(bond.GetIdx())

        # Prepare colors
        atom_color = (0.95, 0.6, 0.0)
        bond_color = (0.95, 0.6, 0.0)
        radius = 0.3

        _ = generate_image(demo_mol, list(comm_atoms), list(comm_bonds), atom_color, bond_color, radius,
                           (400, 400), f'{save_dir}/{smi_p}-vs-{smi_r}.png', False)


def canonicalize_smiles(smi: str, map_clear=True, cano_with_heavyatom=True) -> str:
    cano_smi = ''
    mol = Chem.MolFromSmiles(smi)

    if mol is None:
        cano_smi = ''
    else:
        if mol.GetNumHeavyAtoms() < 2 and cano_with_heavyatom:
            cano_smi = 'CC'
        elif map_clear:
            for a in mol.GetAtoms():
                a.ClearProp('molAtomMapNumber')
            cano_smi = Chem.MolToSmiles(mol, isomericSmiles=True)
        else:
            cano_smi = Chem.MolToSmiles(mol, isomericSmiles=True)
    return cano_smi


def mol_props(smi):
    mol = Chem.MolFromSmiles(smi)
    # 原子数量
    atoms_num = mol.GetNumAtoms()
    # 环数量
    ri = mol.GetRingInfo()
    rings_num = ri.NumRings()
    return atoms_num, rings_num



def plot_kde_comp(prop_train, prop_test, plot_dir, title):
        np.random.seed(42)  # 为了结果可复现
        prop_train = np.asarray(prop_train)
        prop_test = np.asarray(prop_test)

        # 计算均值
        train_mean = np.mean(prop_train)
        test_mean = np.mean(prop_test)

        plt.figure(figsize=(12, 8))
        sns.kdeplot(prop_train, label='train', fill=True, alpha=0.5, linewidth=2)  # , color='blue'
        sns.kdeplot(prop_test, label='test', fill=True, alpha=0.5, linewidth=2)  # , color='green'

        # 绘制均值竖线
        plt.axvline(x=train_mean, color='blue', linestyle='--', linewidth=2,
                    label=f'train mean: {train_mean:.3f}')
        plt.axvline(x=test_mean, color='green', linestyle='--', linewidth=2,
                    label=f'test mean: {test_mean:.3f}')

        plt.xlabel(title, fontsize=28)
        plt.ylabel('Density', fontsize=28)

        # 调整坐标轴刻度数字的字体大小
        plt.tick_params(axis='both', labelsize=24)

        plt.grid(True)

        # 显示图例
        plt.legend(fontsize=24, loc='upper right', ncol=1)
        # plt.legend(fontsize=24)

        plt.tight_layout()  # 自动调整子图参数，使之填充整个图像区域
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)
        else:
            plt.savefig(os.path.join(plot_dir, f'comparison_{title}.svg'))
            plt.savefig(os.path.join(plot_dir, f'comparison_{title}.pdf'))

        plt.show()
        plt.close()

