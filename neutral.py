#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 09:13:45 2021

@author: zhm
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 01:18:07 2020

@author: deepchem
"""

""" contribution from Hans de Winter """
# This code defines a function NeutraliseCharges that attempts to neutralize charges 
# in a molecule represented by its SMILES notation. 
# The neutralization is performed using a set of predefined reaction patterns stored 
# in the _InitialiseNeutralisationReactions function.

from rdkit import Chem
from rdkit.Chem import AllChem

# _InitialiseNeutralisationReactions function:
# Defines a set of reaction patterns (patts) for neutralizing charges in specific 
# functional groups.
# Each pattern consists of a SMARTS pattern for identifying the charged substructure 
# and a corresponding SMILES pattern for the neutralized product.
# Returns a list of tuples, where each tuple contains a SMARTS pattern 
# and the corresponding product SMILES.

def _InitialiseNeutralisationReactions():
    patts= (
    # Imidazoles
    ('[n+;H]','n'),
    # Amines
    ('[N+;!H0]','N'),
    # Carboxylic acids and alcohols
    ('[$([O-]);!$([O-][#7])]','O'),
    # Thiols
    ('[S-;X1]','S'),
    # Sulfonamides
    ('[$([N-;X2]S(=O)=O)]','N'),
    # Enamines
    ('[$([N-;X2][C,N]=C)]','N'),
    # Tetrazoles
    ('[n-]','[nH]'),
    # Sulfoxides
    ('[$([S-]=O)]','S'),
    # Amides
    ('[$([N-]C=O)]','N'),
    )
    return [(Chem.MolFromSmarts(x),Chem.MolFromSmiles(y,False)) for x,y in patts]
_reactions=None
# Takes a SMILES string (smiles) as input and attempts to neutralize charges 
# using a set of predefined reactions.
def NeutraliseCharges(smiles, reactions=None):
    global _reactions

    # If a custom set of reactions (reactions) is provided, 
    # it uses that; otherwise, it initializes and uses the default set of reactions 
    # from _InitialiseNeutralisationReactions.
    if reactions is None:
        if _reactions is None:
            _reactions=_InitialiseNeutralisationReactions()
        reactions=_reactions
    # Converts the input SMILES string to a RDKit molecule object (mol).
    mol = Chem.MolFromSmiles(smiles)
    replaced = False

    # Iterates through the list of reactions, attempting to replace charged substructures 
    # with neutralized products until no further replacements can be made.
    # If at least one replacement was made (replaced is True), 
    # returns the SMILES string of the modified molecule 
    # and a boolean flag indicating that modifications were made. 
    # Otherwise, returns the original SMILES string and False.
    for i,(reactant, product) in enumerate(reactions):
        while mol.HasSubstructMatch(reactant):
            replaced = True
            rms = AllChem.ReplaceSubstructs(mol, reactant, product)
            mol = rms[0]
    if replaced:
        return (Chem.MolToSmiles(mol,True), True)
    else:
        return (smiles, False)
