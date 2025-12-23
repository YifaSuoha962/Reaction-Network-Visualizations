from collections import Counter, defaultdict
from joblib import Parallel, delayed

import pandas as pd
import os
import json
from mol_utils import add_result_args, canonicalize_smiles
from tqdm import tqdm


def process_one_item_json(item):
    """
    单个 (product_smiles, info) 的处理逻辑：
    返回:
        - local_rows: List[str]  (每条重新排列后的反应)
        - local_freq: Counter    (分子 -> 计数)
    """
    product_smiles, info = item
    local_rows = []
    local_targ = defaultdict()
    local_freq = Counter()

    # 这里在最外层 info 中直接取 depth 和 num_reaction_trees
    depth = info.get("depth", None)
    num_rt = info.get("num_reaction_trees", None)

    for sub_key, tree_obj in info.items():
        # 只保留 "1" ~ "10" 这种“纯数字字符串”的 key
        if not isinstance(sub_key, str) or not sub_key.isdigit():
            continue
        retro_routes = tree_obj.get("retro_routes", [])
        for rxn_list in retro_routes:
            for i, rxn_smi in enumerate(rxn_list):
                prod, react = rxn_smi.split('>>')
                local_freq[prod] += 1
                all_reacts = react.split('.')
                for r in all_reacts:
                    local_freq[r] += 1

                reordered_rxn_smi = react + '>>' + prod
                local_rows.append(reordered_rxn_smi)

                if i == 0:
                    local_targ[prod] = {
                        "depth": depth,
                        "num_reaction_trees": num_rt
                    }

    # for rxn_list in info['retro_routes']:

    return local_rows, local_freq, local_targ


if __name__ == '__main__':
    args = add_result_args()

    save_dir = f'data/{args.data_mode}'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    rows = []               # 所有反应
    freq_mols = Counter()   # smi: occur_freq
    product_info = defaultdict()        # 所有 target mol

    if args.data_name == 'retro-bench':
        raw_data = f'../{args.data_mode}_dataset.json'
        with open(raw_data, 'r') as json_file:
            data = json.load(json_file)

        # ===== 2. 提取 retro_routes 中的所有字符串，保存为 csv =====

        # ===== 并行部分 =====
        items = list(data.items())  # 转成 list，方便多次遍历
        results = Parallel(n_jobs=-1, backend="loky")(
            delayed(process_one_item_json)(item)
            for item in tqdm(items, desc="Processing routes")
        )

        for local_rows, local_freq, local_targ in results:
            rows.extend(local_rows)
            freq_mols.update(local_freq)
            for prod, meta in local_targ.items():
                product_info[prod] = meta

        # product_list = list(data.keys())

        rows_for_df = []
        for prod, meta in product_info.items():
            rows_for_df.append({
                "Target_SMILES": prod,
                "route_depth": meta.get("depth"),
                "route_nums": meta.get("num_reaction_trees"),
            })
        product_df = pd.DataFrame(rows_for_df)

        output_products_csv = os.path.join(save_dir, "targets.csv")
        product_df.to_csv(output_products_csv, index=False)
        print(f"Saved {len(product_info)} products to {output_products_csv}")
        # with open(output_products_json, "w", encoding="utf-8") as f:
        #     # 存成一个字符串列表即可，每一项就是一个产物 SMILES
        #     json.dump(product_list, f, indent=2, ensure_ascii=False)

    else:
        ori_file = f"data/{args.data_name}/raw_{args.data_mode}.csv"
        data = pd.read_csv(ori_file)
        rows = data['reactants>reagents>production'].to_list()
        for rxn in rows:
            r, p = rxn.split('>>')
            cano_r = canonicalize_smiles(r)
            cano_p = canonicalize_smiles(p)
            freq_mols[cano_p] += 1
            freq_mols[cano_r] += 1

    # 转成 DataFrame 并写入 csv
    df = pd.DataFrame({'reactants>reagents>production': rows})
    output_csv_path = os.path.join(save_dir, "collected_rxns.csv")  # 想要的输出文件名
    df.to_csv(output_csv_path, index=False)

    print(f"Saved {len(df)} rows to {output_csv_path}")

    # ===== 4. 将所有分子及其出现频率，保存到另一个 json 文件 =====
    df_freq = pd.DataFrame.from_dict(freq_mols, orient="index", columns=["frequency"])
    df_freq.reset_index(inplace=True)
    df_freq.rename(columns={"index": "Mol_SMILES"}, inplace=True)
    mol_freq_csv = os.path.join(save_dir, "mol_freq.csv")
    df_freq.to_csv(mol_freq_csv, index=False)

    # mol_freq_json = os.path.join(save_dir, "mol_freq.json")
    # with open(mol_freq_json, "w", encoding="utf-8") as f:
    #     # 存成一个字符串列表即可，每一项就是一个产物 SMILES
    #     json.dump(freq_mols, f, indent=2, ensure_ascii=False)


