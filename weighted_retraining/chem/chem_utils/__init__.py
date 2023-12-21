""" Contains many chem utils codes """
import rdkit
from rdkit import Chem
from rdkit.Chem import Crippen
import networkx as nx
from rdkit.Chem import rdmolops

from rdkit.Chem import QED
from rdkit.Chem.rdMolDescriptors import CalcTPSA, CalcNumRotatableBonds

from rdkit.Chem import AllChem
import os
import pickle
import numpy as np
from sklearn import svm

# My imports
from weighted_retraining.chem.chem_utils.SA_Score import sascorer
from weighted_retraining.chem.chem_utils.NP_Score import npscorer
# print(os.path.dirname(os.path.realpath(__file__)))
with open(os.path.dirname(os.path.realpath(__file__))+'/clf.pkl', "rb") as f:
    activity_clf = pickle.load(f)

# Make rdkit be quiet
def rdkit_quiet():
    lg = rdkit.RDLogger.logger()
    lg.setLevel(rdkit.RDLogger.CRITICAL)

def get_mol(smiles_or_mol):                                                     
    '''                                                                                                                                       
    Loads SMILES/molecule into RDKit's object                                   
    '''                                                                                                                                       
    if isinstance(smiles_or_mol, str):                                          
        if len(smiles_or_mol) == 0:                                              
            return None                                                           
        mol = Chem.MolFromSmiles(smiles_or_mol)                                 
        if mol is None:                                                          
            return None                                                           
        try:                                                                    
            Chem.SanitizeMol(mol)                                                 
        except ValueError:                                                      
            return None                                                           
        return mol                                                              
    return smiles_or_mol

def standardize_smiles(smiles):
    """ Get standard smiles without stereo information """
    mol = get_mol(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, isomericSmiles=False)


def penalized_logP(smiles: str, min_score=-float("inf")) -> float:
    """ calculate penalized logP for a given smiles string """
    mol = Chem.MolFromSmiles(smiles)
    logp = Crippen.MolLogP(mol)
    sa = SA(mol)

    # Calculate cycle score
    cycle_length = _cycle_score(mol)

    """
    Calculate final adjusted score.
    These magic numbers are the empirical means and
    std devs of the dataset.

    I agree this is a weird way to calculate a score...
    but this is what previous papers did!
    """
    score = (
        (logp - 2.45777691) / 1.43341767
        + (-sa + 3.05352042) / 0.83460587
        + (-cycle_length - -0.04861121) / 0.28746695
    )
    return max(score, min_score)

def logP_func(smiles: str) -> float:
    """ calculate logP for a given smiles string """
    mol = Chem.MolFromSmiles(smiles)
    logp = Crippen.MolLogP(mol)
    
    return logp

def SAS_func(smiles: str) -> float:
    """ calculate SAS for a given smiles string """
    mol = Chem.MolFromSmiles(smiles)
    sa = SA(mol)

    return sa

def tpsa_func(smiles: str) -> float:
    """ calculate TPSA for a given smiles string """
    mol = Chem.MolFromSmiles(smiles)
    tpsa = CalcTPSA(mol)

    return tpsa

def num_rb_func(smiles: str) -> float:
    """ calculate number of rotatable bonds for a given smiles string """
    mol = Chem.MolFromSmiles(smiles)
    num_rb = CalcNumRotatableBonds(mol)

    return num_rb

# def qed_func(smiles: str) -> float:
#     """ calculate SAS for a given smiles string """
#     mol = Chem.MolFromSmiles(smiles)
#     qed = QED(mol)

#     return qed

def qed_func(smiles: str) -> float:
    """ calculate qed for a given smiles string """
    mol = Chem.MolFromSmiles(smiles)
    
    return QED.qed(mol)


def NP_score_func(smiles: str) -> float:
    """ calculate NP_score for a given smiles string """
    mol = Chem.MolFromSmiles(smiles)
    
    NP_score = npscorer.scoreMol(mol)
    
    return NP_score


def fingerprints_from_mol( mol):
    fp = AllChem.GetMorganFingerprint(mol, 3, useCounts=True, useFeatures=True)
    size = 2048
    nfp = np.zeros((1, size), np.int32)
    for idx,v in fp.GetNonzeroElements().items():
        nidx = idx%size
        nfp[0, nidx] += int(v)
    return nfp

def activity_func(smiles: str) -> float:
    """ predict probablity of being active against DRD2 for a given smiles string """
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        fp = fingerprints_from_mol(mol)
        score = activity_clf.predict_proba(fp)[:, 1]
        return float(score)
    return 0.0


def SA(mol):
    return sascorer.calculateScore(mol)


def _cycle_score(mol):
    cycle_list = nx.cycle_basis(nx.Graph(rdmolops.GetAdjacencyMatrix(mol)))
    if len(cycle_list) == 0:
        cycle_length = 0
    else:
        cycle_length = max([len(j) for j in cycle_list])
    if cycle_length <= 6:
        cycle_length = 0
    else:
        cycle_length = cycle_length - 6
    return cycle_length


# def QED(smiles: str) -> float:
#     mol = Chem.MolFromSmiles(smiles)
#     return Chem.QED.qed(mol)
