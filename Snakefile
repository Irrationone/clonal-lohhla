configfile: "config/test.yaml"

rule all:
    input:
        '{outdir}/model_outputs'.format(outdir=config['outdir']),


rule run_model:
    input:
        tree_edges=config['clonal']['tree_edges'],
        prevalences=config['clonal']['prevalences'],
        hlaloss=config['lohhla']['hlaloss'],
        cpn=config['lohhla']['cpn'],
        ploidy_cellularity=config['ploidy_cellularity'],
        unique_read_counts=config['unique_read_counts'],
    params:
        name='run-model',
    log:
        '{logdir}/run_model.log'.format(logdir=config['logdir']),
    output:
        '{outdir}/model_outputs'.format(outdir=config['outdir']),
    shell:
        'python python/run_clonal_lohhla.py '
        '--clonal_prevalence_file {input.prevalences} '
        '--flagstat_file {input.unique_read_counts} '
        '--hlaloss_file {input.hlaloss} '
        '--integercpn_file {input.cpn} '
        '--tree_edges_file {input.tree_edges} '
        '--ploidy_cellularity_file {input.ploidy_cellularity} '
        '--anchor_model_type {params.anchor_model} '
        '--anchor_mode {params.anchor_mode} '
        '--outdir {output} '
        '>& {log}'