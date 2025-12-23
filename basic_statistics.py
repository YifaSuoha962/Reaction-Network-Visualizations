from mol_utils import mol_props, add_result_args, plot_kde_comp
import pandas as pd
import os

from joblib import Parallel, delayed


def load_split_dfs(data_name, split):
    """
    统一读取某个 split（train / test）的 mol_freq.csv 和 targets.csv

    返回：
        mol_info_df, route_info_df
    """
    base_dir = os.path.join("data", split)
    mol_csv = os.path.join(base_dir, "mol_freq.csv")
    route_csv = os.path.join(base_dir, "targets.csv")

    mol_info_df = pd.read_csv(mol_csv)
    route_info_df = pd.read_csv(route_csv)

    return mol_info_df, route_info_df

def save_split_dfs(data_name, split, df):
    """
    统一读取某个 split（train / test）的 mol_freq.csv 和 targets.csv

    返回：
        mol_info_df, route_info_df
    """
    base_dir = os.path.join("data", split)
    mol_csv = os.path.join(base_dir, "mol_freq.csv")
    df.to_csv(mol_csv, index=False)
    print("Files saved successfully")

def compute_mol_props_for_df(mol_info_df, smiles_col='Mol_SMILES', n_jobs=-1):
    """
    并行计算每个分子的 (Atoms_num, Rings_num)
    """
    mol_list = mol_info_df[smiles_col].tolist()

    results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(mol_props)(smi) for smi in mol_list
    )

    if len(results) > 0:
        atoms_list, rings_list = zip(*results)
        atoms_list, rings_list = list(atoms_list), list(rings_list)
    else:
        atoms_list, rings_list = [], []

    return atoms_list, rings_list


def get_mol_props_columns(mol_info_df, smiles_col='Mol_SMILES', n_jobs=-1):
    """
    确保 DataFrame 中存在 'Atoms_num' 和 'Rings_num' 两列：
        - 若列已存在，则保留原值
        - 若列不存在，则并行计算并补上
    """
    df = mol_info_df.copy()

    need_atoms = 'Atoms_num' not in df.columns
    need_rings = 'Rings_num' not in df.columns

    if need_atoms or need_rings:
        atoms_list, rings_list = compute_mol_props_for_df(
            df, smiles_col=smiles_col, n_jobs=n_jobs
        )
        if need_atoms:
            df['Atoms_num'] = atoms_list
        if need_rings:
            df['Rings_num'] = rings_list
    else:
        atoms_list = df['Atoms_num'].to_list()
        rings_list = df['Rings_num'].to_list()
    return atoms_list, rings_list, df



def add_mol_props_columns(mol_info_df, smiles_col='Mol_SMILES', n_jobs=-1):
    AN_list, RN_list = compute_mol_props_for_df(
        mol_info_df, smiles_col=smiles_col, n_jobs=n_jobs
    )
    df = mol_info_df.copy()
    df['Atoms_num'] = AN_list   # 分子尺寸分布
    df['Rings_num'] = RN_list   # 环数量分布
    return df


def get_syn_route_columns(mol_info_df, depth_col='route_depth', num_col='route_nums'):
    return mol_info_df[depth_col].to_list(), mol_info_df[num_col].to_list()


if __name__ == '__main__':
    args = add_result_args()

    mol_info_df_train, route_info_df_train = load_split_dfs(args.data_name, "train")
    mol_info_df_test, route_info_df_test = load_split_dfs(args.data_name, "test")

    # 用法
    an_train, rn_train, checked_df_train = get_mol_props_columns(mol_info_df_train, n_jobs=-1)
    an_test, rn_test, checked_df_test = get_mol_props_columns(mol_info_df_test, n_jobs=-1)

    save_split_dfs(args.data_name, "train", checked_df_train)
    save_split_dfs(args.data_name, "test", checked_df_test)

    # 合成路径长度分布
    depth_train, num_train = get_syn_route_columns(route_info_df_train)
    depth_test, num_test = get_syn_route_columns(route_info_df_test)

    save_dir = 'whole_statistics_plots'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    plot_kde_comp(an_train, an_test, plot_dir=save_dir, title='Atom Counts')
    plot_kde_comp(rn_train, rn_test, plot_dir=save_dir, title='Ring Counts')
    plot_kde_comp(depth_train, depth_test, plot_dir=save_dir, title='Route Depths')
    plot_kde_comp(num_train, num_test, plot_dir=save_dir, title='Route Numbers')