#!/bin/bash

mkdir -p sensitivity_experiment/unstructured/yaml
mkdir -p sensitivity_experiment/unstructured/result

layers=(3 7 10 14 17 21 24 28 31)
ratios=(10 20 30 40 50 60 70 80 90)

total_start=$SECONDS

for layer in "${layers[@]}"
do
    for ratio in "${ratios[@]}"
    do
        experiment_start=$SECONDS

        ratio_decimal=$(printf "0.%02d" "$ratio")

        yaml_file="sensitivity_experiment/unstructured/yaml/features_${layer}_${ratio}.yaml"
        result_file="sensitivity_experiment/unstructured/result/features_${layer}_${ratio}.txt"

        cat > "$yaml_file" << EOF
prune_ratios:
  features.0.weight: 0.0
  features.3.weight: 0.0
  features.7.weight: 0.0
  features.10.weight: 0.0
  features.14.weight: 0.0
  features.17.weight: 0.0
  features.21.weight: 0.0
  features.24.weight: 0.0
  features.28.weight: 0.0
  features.31.weight: 0.0
EOF

        sed -i "s/features.${layer}.weight: 0.0/features.${layer}.weight: ${ratio_decimal}/" "$yaml_file"

        echo "Running unstructured: features.${layer}.weight at ${ratio}%"

        CUDA_VISIBLE_DEVICES=0 python main.py --sparsity-type unstructured --sparsity-method omp --yaml-path "$yaml_file" 2>&1 | grep "Test set:" > "$result_file"

        experiment_time=$((SECONDS - experiment_start))

        echo "Completed unstructured: features.${layer}.weight at ${ratio}% in $((experiment_time / 60))m $((experiment_time % 60))s"
    done
done

total_time=$((SECONDS - total_start))

echo "All unstructured experiments completed in $((total_time / 3600))h $(((total_time % 3600) / 60))m $((total_time % 60))s"