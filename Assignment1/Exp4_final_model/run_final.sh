#!/bin/bash

# GPU 0: Unstructured pruning
(
    echo "Starting OMP Unstructured on GPU 0"
    START=$(date +%s)

    CUDA_VISIBLE_DEVICES=0 python main.py --sparsity-type unstructured --sparsity-method omp --yaml-path ./vgg13_unstructured.yaml > omp_unstructured.txt 2>&1

    END=$(date +%s)
    ELAPSED=$((END - START))
    echo "OMP Unstructured time: $((ELAPSED / 3600))h $(((ELAPSED % 3600) / 60))m $((ELAPSED % 60))s"


    echo "Starting IMP Unstructured on GPU 0"
    START=$(date +%s)

    CUDA_VISIBLE_DEVICES=0 python main.py --sparsity-type unstructured --sparsity-method imp --yaml-path ./vgg13_unstructured.yaml > imp_unstructured.txt 2>&1

    END=$(date +%s)
    ELAPSED=$((END - START))
    echo "IMP Unstructured time: $((ELAPSED / 3600))h $(((ELAPSED % 3600) / 60))m $((ELAPSED % 60))s"
) &


# GPU 1: Filter pruning
(
    echo "Starting OMP Filter on GPU 1"
    START=$(date +%s)

    CUDA_VISIBLE_DEVICES=1 python main.py --sparsity-type filter --sparsity-method omp --yaml-path ./vgg13_filter.yaml > omp_filter.txt 2>&1

    END=$(date +%s)
    ELAPSED=$((END - START))
    echo "OMP Filter time: $((ELAPSED / 3600))h $(((ELAPSED % 3600) / 60))m $((ELAPSED % 60))s"


    echo "Starting IMP Filter on GPU 1"
    START=$(date +%s)

    CUDA_VISIBLE_DEVICES=1 python main.py --sparsity-type filter --sparsity-method imp --yaml-path ./vgg13_filter.yaml > imp_filter.txt 2>&1

    END=$(date +%s)
    ELAPSED=$((END - START))
    echo "IMP Filter time: $((ELAPSED / 3600))h $(((ELAPSED % 3600) / 60))m $((ELAPSED % 60))s"
) &


wait

echo "All four experiments finished"