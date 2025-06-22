#!/bin/bash
set -euo pipefail
shopt -s nullglob

RAW_FASTQ_DIR="fastq".        # Input directory for raw FASTQ files
REPAIRED_DIR="fastq_repaired" # Directory for repaired FASTQ files (if needed)
CLEAN_DIR="clean_fastq"       # Directory for fastp-cleaned FASTQ files
STAR_OUT_DIR="STAR_outputs"   # Directory for STAR alignment outputs
STAR_INDEX_DIR="STAR_index"   # Directory containing the STAR genome index
THREADS=16                    # Number of threads for parallel processing
STAR_SORT_RAM=240000000000    # RAM limit for STAR sorting (in bytes)


echo "Starting RNA-seq Alignment Pipeline..."
echo "[SETUP] Creating output directories..."
mkdir -p "$REPAIRED_DIR" "$CLEAN_DIR" "$STAR_OUT_DIR"

# Verify that the STAR index directory exists
if [ ! -d "$STAR_INDEX_DIR" ] || [ -z "$(ls -A "$STAR_INDEX_DIR")" ]; then
    echo "[ERROR] STAR index directory '$STAR_INDEX_DIR' is missing or empty." >&2
    exit 1
fi

# Find all forward read files (*_1.fastq) in the raw FASTQ directory
RAW_R1_FILES=("$RAW_FASTQ_DIR"/*_1.fastq)
if [ ${#RAW_R1_FILES[@]} -eq 0 ]; then
    echo "[ERROR] No files matching *_1.fastq found in '${RAW_FASTQ_DIR}'." >&2
    exit 1
fi

for R1_RAW in "${RAW_R1_FILES[@]}"; do
    # Extract the sample name from the R1 filename
    SAMPLE=$(basename "$R1_RAW" _1.fastq)
    R2_RAW="$RAW_FASTQ_DIR/${SAMPLE}_2.fastq"

    # Verify that the paired-end R2 file exists
    if [ ! -f "$R2_RAW" ]; then
        echo "[WARNING] Paired file not found: '$R2_RAW'. Skipping sample ${SAMPLE}."
        continue
    fi

    FASTP_INPUT_R1=""
    FASTP_INPUT_R2=""

    echo "[STEP 1/3] Checking read counts for ${SAMPLE}..."
    R1_LINES=$(wc -l < "$R1_RAW")
    R2_LINES=$(wc -l < "$R2_RAW")
    
    if [ "$R1_LINES" -eq "$R2_LINES" ]; then
        echo "  [INFO] Read counts match. Using original files."
        # Use the original raw files as input for the next step
        FASTP_INPUT_R1="$R1_RAW"
        FASTP_INPUT_R2="$R2_RAW"
    else
        echo "  [WARNING] Read counts mismatch! Running repair.sh..."
        echo "    - R1 reads: $(( R1_LINES / 4 ))"
        echo "    - R2 reads: $(( R2_LINES / 4 ))"
        
        REPAIRED_R1="$REPAIRED_DIR/${SAMPLE}_repaired_1.fastq"
        REPAIRED_R2="$REPAIRED_DIR/${SAMPLE}_repaired_2.fastq"
        
        # Run repair.sh from BBMap
        repair.sh in1="$R1_RAW" in2="$R2_RAW" out1="$REPAIRED_R1" out2="$REPAIRED_R2"
        echo "  [INFO] repair.sh finished."

        # Check if repair was successful before proceeding
        if [ -f "$REPAIRED_R1" ] && [ -f "$REPAIRED_R2" ]; then
            # Use the newly repaired files as input for the next step
            FASTP_INPUT_R1="$REPAIRED_R1"
            FASTP_INPUT_R2="$REPAIRED_R2"
        else
            echo "  [ERROR] repair.sh failed to create output files. Skipping sample ${SAMPLE}."
            continue
        fi
    fi

    echo "[STEP 2/3] Running quality control with fastp..."
    CLEAN_R1="${CLEAN_DIR}/${SAMPLE}_clean_1.fastq"
    CLEAN_R2="${CLEAN_DIR}/${SAMPLE}_clean_2.fastq"
    FASTP_REPORT_HTML="${CLEAN_DIR}/${SAMPLE}_fastp.html"
    FASTP_REPORT_JSON="${CLEAN_DIR}/${SAMPLE}_fastp.json"
    
    fastp \
      -i "$FASTP_INPUT_R1" -I "$FASTP_INPUT_R2" \
      -o "$CLEAN_R1" -O "$CLEAN_R2" \
      --thread "$THREADS" \
      --html "$FASTP_REPORT_HTML" --json "$FASTP_REPORT_JSON"
    echo "  [INFO] fastp finished."

    echo "[STEP 3/3] Aligning reads with STAR..."
    STAR \
      --runThreadN "$THREADS" \
      --genomeDir "$STAR_INDEX_DIR" \
      --readFilesIn "$CLEAN_R1" "$CLEAN_R2" \
      --outFileNamePrefix "${STAR_OUT_DIR}/${SAMPLE}_" \
      --outSAMtype BAM SortedByCoordinate \
      --limitBAMsortRAM "$STAR_SORT_RAM"
    echo "  [INFO] STAR alignment finished for ${SAMPLE}."

Done
