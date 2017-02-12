import re
import numpy as np
from matplotlib import pyplot as pp
import feature_extraction as extraction
pp.ion()

def read_data( data_path ):
    ff = open(data_path, 'r')
    lines = ff.readlines()
    ff.close()
    peptides = []
    for l in lines :
        values = re.split(r'\s+', l)
        if not '.' in values[0]:
            values[0] = 'A.' + values[0] + '.A'
        peptides.append(extraction.peptide(values[0], float(values[1])))
    return np.array(peptides)

def checked_duplicated(peptides):
    candiates = []
    rt = []
    for pp in peptides:
        if pp.sequence not in candiates:
            candiates.append(pp.sequence)
            rt.append(pp.rt)

    if len(candiates) == len(peptides):
        return "No duplicated peptide in this data set"
    else:
        for i in range(len(candiates)):
            for pp in peptides:
                if candiates[i] == pp.sequence and rt[i] != pp.rt:
                    return "Error: Same peptide has different RT"

    return "All same peptide has same RT"