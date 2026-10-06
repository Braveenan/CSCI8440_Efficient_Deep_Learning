#!/bin/bash

mkdir -p combo_experiment/unstructured/yaml
mkdir -p combo_experiment/unstructured/result

combos=(
    "0 80 80 80 80 80 80 80 80 80"
    "0 70 70 80 70 70 90 90 60 90"
    "0 70 70 80 70 70 85 90 65 90"
    "0 70 70 80 70 70 80 90 70 90"
    "0 70 70 80 70 70 85 85 70 90"
    "0 70 70 80 70 70 85 90 70 85"
    "0 60 70 80 70 70 90 90 65 90"
    "0 70 80 80 60 70 90 90 65 90"
    "0 70 70 90 60 70 90 90 65 90"
    "0 60 80 80 70 70 85 90 65 90"
)

layers=(0 3 7 10 14 17 21 24 28 31)

total_start=$SECONDS

for i in "${!combos[@]}"
do
    combo_num=$((i + 1))
    read -ra ratios <<< "${combos[$i]}"

    yaml_file="combo_experiment/unstructured/yaml/combo_${combo_num}.yaml"

    cat > "$yaml_file" << EOF
prune_ratios:
EOF

    for j in "${!layers[@]}"
    do
        ratio_decimal=$(printf "0.%02d" "${ratios[$j]}")
        echo "  features.${layers[$j]}.weight: ${ratio_decimal}" >> "$yaml_file"
    done

    echo "Running Combo ${combo_num} OMP"

    CUDA_VISIBLE_DEVICES=0 python main.py \
        --sparsity-type unstructured \
        --sparsity-method omp \
        --yaml-path "$yaml_file" \
        2>&1 | grep "Test set:" > "combo_experiment/unstructured/result/combo_${combo_num}_omp.txt"
    
    echo "Completed Combo ${combo_num} OMP"
    
    echo "Running Combo ${combo_num} IMP"
    
    CUDA_VISIBLE_DEVICES=0 python main.py \
        --sparsity-type unstructured \
        --sparsity-method imp \
        --yaml-path "$yaml_file" \
        2>&1 | grep "Test set:" > "combo_experiment/unstructured/result/combo_${combo_num}_imp.txt"
    
    echo "Completed Combo ${combo_num} IMP"
    
done

total_time=$((SECONDS - total_start))

echo "All combination experiments completed in $((total_time / 3600))h $(((total_time % 3600) / 60))m $((total_time % 60))s"