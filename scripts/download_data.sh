#!/bin/bash

# List of datasets
datasets=(
    "1027_yanjie_football"
    "0908_bidex_v5"
    "1027_charlie_pick_brick"
    "1022_charlie_pick_brick"
    "0908_bidex_v4"
    "0904_kickT_v2"
    "0904_bidex"
    "0904_kickT"
    "0906_bidex_v2"
    "0905_kickT_v3"
    "1022_charlie_fold"
)

# Base directories
BASE_DIR="/home/ubuntu/projects/TWIST2-Gr00T"
DATASETS_DIR="${BASE_DIR}/datasets"

# Loop through each dataset
for dataset in "${datasets[@]}"; do
    echo "=========================================="
    echo "Processing: ${dataset}"
    echo "=========================================="
    
    # Step 1: Download from Hugging Face
    echo "Downloading ${dataset} from Hugging Face..."
    hf download \
        --repo-type dataset \
        "yjze/${dataset}" \
        --include "*.tar.gz" \
        --local-dir "${DATASETS_DIR}/${dataset}"
    
    # Check if download succeeded
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to download ${dataset}, skipping..."
        continue
    fi
    
    # Step 2: Extract tar.gz files
    echo "Extracting ${dataset} archives..."
    cd "${DATASETS_DIR}/"
    
    # Find and extract all .tar.gz files
    tar_files=$(find . -name "*.tar.gz" -type f)
    if [ -z "$tar_files" ]; then
        echo "WARNING: No .tar.gz files found in ${dataset}"
    else
        for tarfile in $tar_files; do
            echo "Extracting: ${tarfile}"
            tar -xzf "${tarfile}"
            if [ $? -ne 0 ]; then
                echo "ERROR: Failed to extract ${tarfile}"
            else
                echo "Successfully extracted ${tarfile}"
                # Optionally remove the tar.gz after extraction to save space
                # rm "${tarfile}"
            fi
        done
    fi
    
    # Step 3: Convert to lerobot format
    echo "Converting ${dataset} to lerobot format..."
    cd "${BASE_DIR}"
    python scripts/convert_twist2_to_lerobot.py \
        --input_dir "./datasets/${dataset}" \
        --output_name "${dataset}_lerobot" \
        --output_dir "./datasets" \
        --fps 30 \
        --image_size 256 256
    
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to convert ${dataset}"
    else
        echo "Successfully processed ${dataset}"
    fi
    
    echo ""
done

echo "=========================================="
echo "All datasets processed!"
echo "=========================================="
