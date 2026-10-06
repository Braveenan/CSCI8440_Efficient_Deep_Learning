#!/bin/bash

mkdir -p combo_experiment/filter/yaml
mkdir -p combo_experiment/filter/result

combos=(
    "0 40 40 40 40 40 40 40 40 40"
    "0 30 30 30 35 35 40 45 45 45"
    "0 30 35 35 35 40 40 45 45 45"
    "0 25 30 35 35 40 45 45 45 45"
    "0 25 30 30 35 40 45 45 50 45"
    "0 25 30 35 40 40 40 45 45 50"
    "0 20 30 35 35 40 45 50 45 45"
    "0 30 25 35 35 40 45 45 50 45"
    "0 25 35 30 40 35 45 50 45 45"
    "0 20 30 40 35 40 50 45 45 45"
)

layers=(0 3 7 10 14 17 21 24 28 31)

total_start=$SECONDS

for i in "${!combos[@]}"
do
    combo_num=$((i + 1))
    read -ra ratios <<< "${combos[$i]}"

    yaml_file="combo_experiment/filter/yaml/combo_${combo_num}.yaml"

    cat > "$yaml_file" << EOF
prune_ratios:
EOF

    for j in "${!layers[@]}"
    do
        ratio_decimal=$(printf "0.%02d" "${ratios[$j]}")
        echo "  features.${layers[$j]}.weight: ${ratio_decimal}" >> "$yaml_file"
    done

    echo "Running Combo ${combo_num} OMP"

    CUDA_VISIBLE_DEVICES=1 python main.py \
        --sparsity-type filter \
        --sparsity-method omp \
        --yaml-path "$yaml_file" \
        2>&1 | grep "Test set:" > "combo_experiment/filter/result/combo_${combo_num}_omp.txt"
    
    echo "Completed Combo ${combo_num} OMP"
    
    echo "Running Combo ${combo_num} IMP"
    
    CUDA_VISIBLE_DEVICES=1 python main.py \
        --sparsity-type filter \
        --sparsity-method imp \
        --yaml-path "$yaml_file" \
        2>&1 | grep "Test set:" > "combo_experiment/filter/result/combo_${combo_num}_imp.txt"
    
    echo "Completed Combo ${combo_num} IMP"

done

total_time=$((SECONDS - total_start))

echo "All filter combination experiments completed in $((total_time / 3600))h $(((total_time % 3600) / 60))m $((total_time % 60))s"