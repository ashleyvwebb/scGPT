#!bin/bash
#SBATCH --job-name=EDA
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4591
#SBATCH --time=00:00:10

SCRIPT_DIR="/springbrook/share/bioinf/csuxfw/scGPT/scgpt/research/data/cellxgene"

python inspect_downloaded_data.py \
    --h5ad-root "/springbrook/share/bioinf/csuxfw/scGPT/scgpt/research/data/cellxgene/h5ad" \
    --vocab-path "/springbrook/share/bioinf/csuxfw/scGPT/scgpt/tokenizer/default_gene_vocab.json" \
    --cancer-gene-path "/springbrook/share/bioinf/csuxfw/scGPT/scgpt/research/data/cancer_genes/cancer_gene_list.txt" \
    --queries "blood+lung+blood-cancer+lung-cancer" \
    --max-files 1 \
    --subset-n-cells 1