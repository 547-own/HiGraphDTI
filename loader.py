import os
import pdb
import torch
import pickle

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import BRICS
from torch_geometric.data import Data, Batch
from torch.utils.data import Dataset, DataLoader
import os.path as osp
# from torch_geometric.data import InMemoryDataset
from collections import defaultdict
# from torch_geometric.data import DataLoader
from tqdm import tqdm
from chemutils import get_mol, get_clique_mol

# allowable node and edge features
allowable_features = {
    'possible_atomic_num_list' : list(range(1, 119)),
    'possible_formal_charge_list' : [0, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5],
    'possible_chirality_list' : [
        Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
        Chem.rdchem.ChiralType.CHI_OTHER
    ],
    'possible_hybridization_list' : [
        Chem.rdchem.HybridizationType.UNSPECIFIED, Chem.rdchem.HybridizationType.S,
        Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2
    ],
    'possible_numH_list' : [0, 1, 2, 3, 4, 5, 6, 7, 8],
    'possible_implicit_valence_list' : [0, 1, 2, 3, 4, 5, 6],
    'possible_degree_list' : [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'possible_bonds' : [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC
    ],
    'possible_bond_dirs' : [ # only for double bond stereo information
        Chem.rdchem.BondDir.NONE,
        Chem.rdchem.BondDir.ENDUPRIGHT,
        Chem.rdchem.BondDir.ENDDOWNRIGHT
    ],
    'possible_bond_stereo': [
        Chem.rdchem.BondStereo.STEREONONE,
        Chem.rdchem.BondStereo.STEREOANY,
        Chem.rdchem.BondStereo.STEREOZ,
        Chem.rdchem.BondStereo.STEREOE,
        Chem.rdchem.BondStereo.STEREOCIS,
        Chem.rdchem.BondStereo.STEREOTRANS,
    ],
    'possible_bond_inring': [None, False, True]
}
max_degree = 0
sequence_word_dict = defaultdict(lambda: len(sequence_word_dict) + 1)

class MoleculeDataset(Dataset):
    def __init__(self, data, ngram=3):

        self.data = data 
        self.ngram = ngram
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        item = self.data.iloc[index]
        return self.process(item)

    def process(self, item):
        smiles, sequence, label = item['smiles'], item['sequence'],  item['interaction']
        ngram_words = item['ngram_words']
        rdkit_mol = AllChem.MolFromSmiles(smiles)
        # label = -1 if label == 0 else label 
        graph_data = mol_to_graph_data_obj_simple(rdkit_mol)
        ngram_words_mask = [1] * len(ngram_words)
        # sequence = split_sequence(sequence, self.ngram)
        return smiles, graph_data, sequence, ngram_words, ngram_words_mask, label
    
    @staticmethod
    def collate(batch):
        batch_smiles, batch_graph_data, batch_sequence, batch_labels = [], [], [], []
        batch_ngram_words, batch_ngram_words_mask = [], []
        for item in batch:
            smiles, graph_data, sequence, ngram_words, ngram_words_mask, label =  item
            batch_ngram_words.append(ngram_words)
            batch_ngram_words_mask.append(ngram_words_mask)
            batch_smiles.append(smiles)
            batch_graph_data.append(graph_data)
            batch_sequence.append(sequence)
            batch_labels.append(label)
        batch_labels = torch.tensor(batch_labels).long()
        batch_graph_data = Batch.from_data_list(batch_graph_data)
        batch_ngram_words = torch.tensor(sequence_padding(batch_ngram_words)).long()
        batch_ngram_words_mask = torch.tensor(sequence_padding(batch_ngram_words_mask)).float()

        return {
            "smiles": batch_smiles,
            "graph_data": batch_graph_data,
            "sequence" : batch_sequence,
            "ngram_words": batch_ngram_words,
            "ngram_words_mask": batch_ngram_words_mask,
            "labels" :batch_labels
        }
    
def sequence_padding(inputs, length=None, padding=0):
    """
    padding for the same length
    """
    if length is None:
        length = max([len(x) for x in inputs])

    pad_width = [(0, 0) for _ in np.shape(inputs[0])]
    outputs = []
    for x in inputs:
        x = x[:length]
        pad_width[0] = (0, length - len(x))
        x = np.pad(x, pad_width, 'constant', constant_values=padding)
        outputs.append(x)

    return np.array(outputs, dtype='int64')




def mol_to_graph_data_obj_simple(mol):
    """
    Converts rdkit mol object to graph Data object required by the pytorch
    geometric package. NB: Uses simplified atom and bond features, and represent
    as indices
    :param mol: rdkit mol object
    :return: graph data object with the attributes: x, edge_index, edge_attr
    """
    # atoms
    num_atom_features = 7   # atom type,  chirality tag
    atom_features_list = []
    for atom in mol.GetAtoms():

        atom_feature = [allowable_features['possible_atomic_num_list'].index(atom.GetAtomicNum())] + \
                [allowable_features['possible_degree_list'].index(atom.GetDegree())] + \
                [allowable_features['possible_formal_charge_list'].index(atom.GetFormalCharge())] + \
                [allowable_features['possible_chirality_list'].index(atom.GetChiralTag())] + \
                [allowable_features['possible_hybridization_list'].index(atom.GetHybridization())] + \
                [allowable_features['possible_numH_list'].index(atom.GetTotalNumHs())] + \
                [allowable_features['possible_implicit_valence_list'].index(atom.GetImplicitValence())]
        # [allowable_features['possible_degree_list'].index(atom.GetTotalDegree())]
#                 [1 if atom.GetIsAromatic() else 0]

        atom_features_list.append(atom_feature)
    x_nosuper = torch.tensor(np.array(atom_features_list), dtype=torch.long)

    # bonds
    num_bond_features = 4   # bond type, bond direction
    if len(mol.GetBonds()) > 0: # mol has bonds
        edges_list = []
        edge_features_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()

            edge_feature = [allowable_features['possible_bonds'].index(
                 bond.GetBondType())] + [allowable_features['possible_bond_inring'].index(
                 bond.IsInRing())] + [allowable_features['possible_bond_dirs'].index(
                bond.GetBondDir())] + [allowable_features['possible_bond_stereo'].index(
                bond.GetStereo())]
            
            edges_list.append((i, j))
            edge_features_list.append(edge_feature)
            edges_list.append((j, i))
            edge_features_list.append(edge_feature)

        # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
        edge_index_nosuper = torch.tensor(np.array(edges_list).T, dtype=torch.long)

        # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
        edge_attr_nosuper = torch.tensor(np.array(edge_features_list),
                                 dtype=torch.long)
    else:   # mol has no bonds
        edge_index_nosuper = torch.empty((2, 0), dtype=torch.long)
        edge_attr_nosuper = torch.empty((0, num_bond_features), dtype=torch.long)

    num_atoms = x_nosuper.size(0)
    super_x = torch.tensor([[119, 0, 0, 0, 0, 0, 0]]).to(x_nosuper.device)
    #add motif 
    cliques = motif_decomp(mol)
    num_motif = len(cliques)
    # motif_x = []
    if num_motif > 0:
        # for i in cliques:
        #     motif_x.append(torch.tensor([120, len(cliques)]))
        # motif_x = torch.stack(motif_x, dim=0).to(x_nosuper.device)
        motif_x = torch.tensor([[120, 0, 0, 0, 0, 0, 0]]).repeat_interleave(num_motif, dim=0).to(x_nosuper.device)
        # super_x[0][1] = num_motif
        x = torch.cat((x_nosuper, motif_x, super_x), dim=0)

        motif_edge_index = []
        for k, motif in enumerate(cliques):
            motif_edge_index = motif_edge_index + [[i, num_atoms+k] for i in motif]
        motif_edge_index = torch.tensor(np.array(motif_edge_index).T, dtype=torch.long).to(edge_index_nosuper.device)

        super_edge_index = [[num_atoms+i, num_atoms+num_motif] for i in range(num_motif)]
        super_edge_index = torch.tensor(np.array(super_edge_index).T, dtype=torch.long).to(edge_index_nosuper.device)
        edge_index = torch.cat((edge_index_nosuper, motif_edge_index, super_edge_index), dim=1)

        motif_edge_attr = torch.zeros(motif_edge_index.size()[1], num_bond_features)
        motif_edge_attr[:,0] = 6 
        motif_edge_attr = motif_edge_attr.to(edge_attr_nosuper.dtype).to(edge_attr_nosuper.device)

        super_edge_attr = torch.zeros(num_motif, num_bond_features)
        super_edge_attr[:,0] = 5 
        super_edge_attr = super_edge_attr.to(edge_attr_nosuper.dtype).to(edge_attr_nosuper.device)
        edge_attr = torch.cat((edge_attr_nosuper, motif_edge_attr, super_edge_attr), dim = 0)

        num_part = (num_atoms, num_motif, 1)

    else:
        x = torch.cat((x_nosuper, super_x), dim=0)

        super_edge_index = [[i, num_atoms] for i in range(num_atoms)]
        super_edge_index = torch.tensor(np.array(super_edge_index).T, dtype=torch.long).to(edge_index_nosuper.device)
        edge_index = torch.cat((edge_index_nosuper, super_edge_index), dim=1)

        super_edge_attr = torch.zeros(num_atoms, num_bond_features)
        super_edge_attr[:,0] = 5 #bond type for self-loop edge
        super_edge_attr = super_edge_attr.to(edge_attr_nosuper.dtype).to(edge_attr_nosuper.device)
        edge_attr = torch.cat((edge_attr_nosuper, super_edge_attr), dim = 0)

        num_part = (num_atoms, 0, 1)
    num_part = torch.tensor(num_part)
    
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, num_part = num_part)

    return data

def motif_decomp(mol):
    
    n_atoms = mol.GetNumAtoms()
    if n_atoms == 1:
        return []

    cliques = []  
    breaks = []
    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtom().GetIdx()
        a2 = bond.GetEndAtom().GetIdx()
        cliques.append([a1, a2])  

    res = list(BRICS.FindBRICSBonds(mol))  
    if len(res) != 0:
        for bond in res:
            if [bond[0][0], bond[0][1]] in cliques:
                cliques.remove([bond[0][0], bond[0][1]])
            else:
                cliques.remove([bond[0][1], bond[0][0]])
            cliques.append([bond[0][0]])
            cliques.append([bond[0][1]]) 

    for c in range(len(cliques) - 1):
        if c >= len(cliques):
            break
        for k in range(c + 1, len(cliques)):
            if k >= len(cliques):
                break
            if len(set(cliques[c]) & set(cliques[k])) > 0: 
                cliques[c] = list(set(cliques[c]) | set(cliques[k]))
                cliques[k] = []
        cliques = [c for c in cliques if len(c) > 0]
    cliques = [c for c in cliques if n_atoms> len(c) > 0]


    num_cli = len(cliques)
    ssr_mol = Chem.GetSymmSSSR(mol)
    for i in range(num_cli):
        c = cliques[i]
        cmol = get_clique_mol(mol, c)
        ssr = Chem.GetSymmSSSR(cmol)
        if len(ssr)>1: 
            for ring in ssr_mol:
                if len(set(list(ring)) & set(c)) == len(list(ring)):
                    cliques.append(list(ring))
            cliques[i]=[]
    
    cliques = [c for c in cliques if n_atoms> len(c) > 0]
    return cliques

if __name__ == '__main__':
    df = pd.read_csv('dataset/human/raw/human.csv')
    # df['words'] = df['sequence'].apply(split_sequence)
    train_dataset = MoleculeDataset(df, ngram=3)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=MoleculeDataset.collate)
    for step, batch in enumerate(tqdm(train_loader, desc="Iteration")):
        pdb.set_trace()

        