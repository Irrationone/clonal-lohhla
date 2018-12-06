import numpy as np
import pandas as pd
import pymc3 as pm


def list_all_genotypes(max_cn=5, allow_zero=True):
    total_cns = range(max_cn + 1)
    if not allow_zero:
        total_cns = np.array(range(max_cn)) + 1

    genotype_list = []

    for cn in total_cns:
        cndf = pd.DataFrame({'alt_cn': range(cn + 1), 'total_cn': cn})
        genotype_list.append(cndf)

    all_genotypes = pd.concat(genotype_list, ignore_index=True)
    all_genotypes.set_index(np.array(range(all_genotypes.shape[0])))
    return all_genotypes