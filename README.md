# Final_Project 
Visualization of Chemical Reaction Network
## Tools preparation

Create a virtual environment.<br>
```
cd RxnNet
conda create -c conda-forge -n rdenv python=3.8 -y
conda install -c tmap tmap
pip install faerun
pip install rdkit
pip install networkx
pip install holoviews
pip install CairoSVG
```
`rdkit`包是解析分子结构必需的package。

## Data preparation
Raw data is acquired from [FusionRetro](https://github.com/songtaoliu0823/fusionretro), in the format of Json.   

Run `collect_reaction.py` to transform the records in the .csv file, and count the occurrence frequency of each molecule (in Json).   
```
data
├───train
│   ├───collected_rxn.csv
│   │───mol_freq.json
│   │───targets.json
│           
├───test
    ├───collected_rxn.csv
    │───mol_freq.json
    │───targets.json
```
**collected_rxn.csv 内容解释**

| reactants>reagents>production   | pred_rxn_class | pred_parent_class | pred_rxn_name     |
|---------------------------------|----------------|------------|-------------------|
| 单条反应记录，以 '>>'分割，左侧为反应物，右侧为产物    | 具体的反应类型代号 | 反应类型（大类）代号 | 反应类型名称  |   
| O=P(Cl)(Cl)[Cl:3].O[c:11]~[cH:12]1>>[Cl:3][c:11]1[c:10]~[cH:16][c:15]2[n:14][cH:13][cH:12]1   | 9.1.6 | 9   | Hydroxy to chloro |

**注： ‘>>’ 左侧的字符串中会有多个分子，以'.'符号分割，如果要做单个分子的表示时，需要用split('.')来切分。**

## Basic Operations
1. 从反应记录中获得单个分子的字符串以及具体结构表示、绘制保存分子结构图像：
```
import pandas as pd 
from rdkit import Chem

rxn_df = pd.read_csv('data/test/collected_rxns.csv')
rxn_list = rxn_df['reactants>reagents>production'].to_list()   # 每行为 'reactant SMILES >> product SMILES'
# 反应物、产物字符串
react_smi, product_smi = rxn_list[0].split('>>')
# 反应物、产物具体结构
react_mol = Chem.MolFromSmarts(react_smi)
product_mol = Chem.MolFromSmarts(product_smi)

# 获取单个分子
reacts = react_smi.split('.')

# 绘制分子结构图
from mol_utils import smi_to_img, smi_to_img_pil
# 直接保存文件
svg_react = smi_to_img(f'{reacts[0]}'.png, reacts[0])
img_react = smi_to_img_pil(reacts[0])
```
2. 获取反应指纹, 计算分子指纹相似度
```
# 续
from rdkit import DataStructs
from mol_utils import get_rxn_fingerprint
from template_extractor import extract_from_reaction

rxn_demo_1 = rxn_list[0]
rxn_demo_2 = rxn_list[1]

# 抽取反应模板
rxn_template_res_1 = extract_from_reaction(rxn_demo)
# 计算分子指纹
rxn_fp_1 = get_rxn_fingerprint(rxn_template_res_1["reaction_smarts"])

rxn_template_res_2 = extract_from_reaction(rxn_demo_2)
rxn_fp_2 = get_rxn_fingerprint(rxn_template_res_2["reaction_smarts"])

# 计算结构相似度
sim = DataStructs.BulkTanimotoSimilarity(rxn_fp_1, rxn_fp_2)
```


### Note: 下面的 task 只需要在 test 数据集上实现即可。考虑到原始数据集过大（10w+）,在样本数量选取时建议用下采样。

## Task1: Reaction Reaction Network
**需求：完善 `reaction_network_construction.py`，形成一张配图和文字描述。**
    demo 基于 networkx 实现，如果想做成ppt里说的那种hyper-edge形式，可以基于holoviews实现。

## Task2: Visualize Reaction Paths
Run `reaction_routes_vis.py` for demonstration.  
**需求：在 reaction_routes_vis.py 代码上拓展，分别形成两张配图和对结果的文字描述：**
 - 实现从 test/train_dataset.json 中采样一个反应树（以最外层的SMILES字符串为key）；
 - 在构造反应树时，从 collected_rxns.csv 索引中间反应对应的 pred_rxn_name 列值，显示在 edge 上。

## Task3: T-SNE of Reaction Classes
Run `tsne_rxn_classes.py --data_mode train/test` for demonstration.  
**需求：在 t-sne_rxn_class.py 代码上拓展如下两功能，分别形成两张配图和对结果的文字描述：**
 - 以pred_parent_class 为label， 或者使用umap等其他降维工具展示各大类反应类型下，的反应指纹的分布特性。
 - 随机选不同的 pred_rxn_class （小类）为label，展示在具体反应类型下，反应指纹的分布特性。
 
## Task4: Heatmap of Reaction Similarities
Run `rxn_fp_similarities.py --data_mode train/test` for demonstration.  
**需求：在 rxn_fp_similarities.py 代码上拓展如下两功能，分别形成两张配图和对结果的文字描述：**
 - 跨parent_class 对比：将同一大类的反应在位置上聚集在一起，最后展示的时候标记出所属的 pred_rxn_name；
 - 在同一parent_class下对比：不同子类类型下反应之间的指纹相似度。

