"""Product complexity scoring functions for molecules."""

import os
import sys

import rdkit
from openbabel import openbabel

openbabel.obErrorLog.SetOutputLevel(0)
from rdkit.Chem import SpacialScore

from icicle.utils.chem.complexity.boettcher import BottchScorer

sys.path.append(os.path.join(os.path.dirname(rdkit.__file__), "Contrib"))
from NP_Score import npscorer
from SA_Score import sascorer

fscore = npscorer.readNPModel()


def get_NPScore(mol):
    try:
        score = npscorer.scoreMol(mol, fscore)
        return score
    except Exception:
        return None


def get_SAScore(mol):
    try:
        score = sascorer.calculateScore(mol)
        return score
    except Exception:
        return None


def get_SPScore(mol):
    try:
        score = SpacialScore.SPS(mol, normalize=True)
        return score
    except Exception:
        return None


def get_BoettcherScore(smiles):
    """
    Adapted from https://gitlab.com/mlpds_mit/askcosv2/molecular_complexity.git

    return None if error
    """

    obmol = openbabel.OBMol()
    obConversion = openbabel.OBConversion()
    obConversion.SetInAndOutFormats("smi", "smi")
    obConversion.ReadString(obmol, smiles)

    scorer = BottchScorer(obmol, verbose=False)
    score = scorer.score(obmol)

    return score
