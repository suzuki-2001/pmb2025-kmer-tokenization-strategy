### RNA-seq Data Sources

The RNA-seq data used for splicing classification is derived from various Arabidopsis thaliana tissues, including shoot, leaf, root, and flower. 

| Tissue | Accession ID | Description | Links |
|--------|--------------|-------------|-------|
| Shoot  | ERR10903222 ERR10903221 ERR10903220 | Illumina NovaSeq 6000 paired-end sequencing; RNA-Seq analysis of splicing factor AtU2AF65a and AtU2AF65b against the wild type (Col). | [NCBI Link](https://www.ncbi.nlm.nih.gov/sra/ERX10345556) |
| Leaf   | SRR29195774 SRR29195773 SRR29195772 | Control plants at control condition rep 3; Arabidopsis thaliana; RNA-Seq. | [NCBI Link](https://www.ncbi.nlm.nih.gov/sra/SRX24715762) |
| Root   | SRR30793574 SRR30793573 SRR30793572 | RNA-seq of Arabidopsis thaliana root: wild type. | [NCBI Link](https://www.ncbi.nlm.nih.gov/sra/SRX26195046) |
| Flower | SRR22881467 SRR22881466 SRR22881465 | RNA-seq data of wild type (Col-0) control replicate 1; Arabidopsis thaliana; RNA-Seq. | [NCBI Link](https://www.ncbi.nlm.nih.gov/sra/?term=SRX18840028) |


### Data Processing Pipeline

```bash
# 1. install required tools
mamba install -c bioconda star hisat2 samtools bedtools ucsc-bedtogenepred ucsc-genepredtobed bioawk gffread -y
mamba install -c bioconda fastp trim-galore multiqc sra-tools -y
mamba install -c conda-forge numpy pandas biopython tqdm -y

# 2. download sra data
prefetch --output-directory rnaseq-data/ --option-file sra_download_list.txt

# 3. convert sra to fastq
find rnaseq-data/ -type f -name "*.sra" | while read sra_file; do
    echo "Processing $sra_file ..."
    fasterq-dump --split-files --threads 16 --outdir fastq/ "$sra_file"
done

# 4. construct STAR index
mkdir STAR_index
STAR --runThreadN 8 \
     --runMode genomeGenerate \
     --genomeDir STAR_index \
     --genomeFastaFiles reference-genome/Arabidopsis_thaliana.TAIR10.dna.toplevel.fa \
     --sjdbGTFfile reference-genome/Arabidopsis_thaliana.TAIR10.51.gff3 \
     --sjdbOverhang 149 \
     --genomeSAindexNbases 12

# 5. align reads to reference genome
bash star_alignment.sh

# 6. extract splice junctions and create splicing dataset
python make_hf_dataset.py
```
