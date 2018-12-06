# Clonal HLA LOH inference

Workflow for basic probabilistic inference of clonal HLA LOH.

Refer to example input files under `data/`. 

## Requirements:

* Tree edges: Two-column table specifying the clonal phylogeny (0 = root)
* Clonal prevalences: Table of clonal prevalences
* Ploidy and cellularity (similar input format to LOHHLA)
* Read stats: Unique read counts from flagstat
* HLA loss table: Output of LOHHLA, but tab-separated
* CPN table: Output of LOHHLA, but tab-separated

## Dependencies

* numpy
* pandas
* pymc3
* theano
* matplotlib

## TODOs

* Account for branch length
* Add versions
* Properly document