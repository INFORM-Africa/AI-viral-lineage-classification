#!/bin/bash

# python train.py --dataset SARS-CoV-2 --path KMER/remove/7 --n_workers 1 --criterion gini --class_weight balanced --test 0 --run_num 0
# python train.py --dataset SARS-CoV-2 --path RTD/remove/7 --n_workers 1 --criterion gini --class_weight balanced --test 0 --run_num 0
# python train.py --dataset SARS-CoV-2 --path RTD/remove/7 --n_workers 1 --criterion entropy --class_weight None --test 0 --run_num 0
python train.py --dataset SARS-CoV-2 --path RTD/remove/7 --n_workers 1 --criterion gini --class_weight balanced --test 0 --run_num 0

# python train.py --dataset SARS-CoV-2 --path RTD/remove/7 --n_workers 3 --criterion gini --class_weight balanced --test 1 --run_num 3
# python train.py --dataset SARS-CoV-2 --path RTD/remove/7 --n_workers 3 --criterion gini --class_weight balanced --test 1 --run_num 4
# python train.py --dataset SARS-CoV-2 --path RTD/remove/7 --n_workers 3 --criterion gini --class_weight balanced --test 1 --run_num 5
# python train.py --dataset SARS-CoV-2 --path RTD/remove/7 --n_workers 3 --criterion gini --class_weight balanced --test 1 --run_num 6
# python train.py --dataset SARS-CoV-2 --path RTD/remove/7 --n_workers 3 --criterion gini --class_weight balanced --test 1 --run_num 7
# python train.py --dataset SARS-CoV-2 --path FCGR/remove/128 --n_workers 2 --criterion gini --class_weight balanced --test 0 --run_num 0
# python train.py --dataset SARS-CoV-2 --path FCGR/remove/128 --n_workers 2 --criterion gini --class_weight balanced --test 1 --run_num 1
# python train.py --dataset SARS-CoV-2 --path FCGR/remove/128 --n_workers 2 --criterion gini --class_weight balanced --test 1 --run_num 2
# python train.py --dataset SARS-CoV-2 --path FCGR/remove/128 --n_workers 2 --criterion gini --class_weight balanced --test 1 --run_num 3
# python train.py --dataset SARS-CoV-2 --path FCGR/remove/128 --n_workers 2 --criterion gini --class_weight balanced --test 1 --run_num 4
# python train.py --dataset SARS-CoV-2 --path FCGR/remove/128 --n_workers 2 --criterion gini --class_weight balanced --test 1 --run_num 5
# python train.py --dataset SARS-CoV-2 --path FCGR/remove/128 --n_workers 2 --criterion gini --class_weight balanced --test 1 --run_num 6
# python train.py --dataset SARS-CoV-2 --path FCGR/remove/128 --n_workers 2 --criterion gini --class_weight balanced --test 1 --run_num 7
# python train.py --dataset SARS-CoV-2 --path FCGR/remove/128 --n_workers 2 --criterion gini --class_weight balanced --test 1 --run_num 8
# python train.py --dataset SARS-CoV-2 --path FCGR/remove/128 --n_workers 2 --criterion gini --class_weight balanced --test 1 --run_num 9
# python train.py --dataset SARS-CoV-2 --path FCGR/remove/128 --n_workers 2 --criterion gini --class_weight balanced --test 1 --run_num 10

# python train.py --dataset SARS-CoV-2 --path FCGR/replace/128 --n_workers 2 --criterion gini --class_weight balanced --test 1 --run_num 10
# python train.py --dataset SARS-CoV-2 --path GSP/remove/real --n_workers 15 --criterion gini --class_weight balanced --test 1 --run_num 10
# python train.py --dataset SARS-CoV-2 --path GSP/replace/real --n_workers 15 --criterion gini --class_weight balanced --test 1 --run_num 3
# python train.py --dataset SARS-CoV-2 --path GSP/replace/real --n_workers 15 --criterion gini --class_weight balanced --test 1 --run_num 4
# python train.py --dataset SARS-CoV-2 --path GSP/replace/real --n_workers 15 --criterion gini --class_weight balanced --test 1 --run_num 5
# python train.py --dataset SARS-CoV-2 --path GSP/replace/real --n_workers 15 --criterion gini --class_weight balanced --test 1 --run_num 6
# python train.py --dataset SARS-CoV-2 --path GSP/replace/real --n_workers 15 --criterion gini --class_weight balanced --test 1 --run_num 7
# python train.py --dataset SARS-CoV-2 --path GSP/replace/real --n_workers 15 --criterion gini --class_weight balanced --test 1 --run_num 8
# python train.py --dataset SARS-CoV-2 --path GSP/replace/real --n_workers 15 --criterion gini --class_weight balanced --test 1 --run_num 9

# python train.py --dataset SARS-CoV-2 --path MASH/replace/21/1000 --n_workers 1 --criterion gini --class_weight None --test 1 --run_num 1

# python train.py --dataset SARS-CoV-2 --path MASH/replace/21/1000 --n_workers 1 --criterion gini --class_weight None --test 1 --run_num 0
# python train.py --dataset SARS-CoV-2 --path MASH/replace/21/1000 --n_workers 1 --criterion gini --class_weight None --test 1 --run_num 1
# python train.py --dataset SARS-CoV-2 --path MASH/replace/21/1000 --n_workers 1 --criterion gini --class_weight None --test 1 --run_num 2
# python train.py --dataset SARS-CoV-2 --path MASH/replace/21/1000 --n_workers 1 --criterion gini --class_weight None --test 1 --run_num 3
# python train.py --dataset SARS-CoV-2 --path MASH/replace/21/1000 --n_workers 1 --criterion gini --class_weight None --test 1 --run_num 4

# python train.py --dataset SARS-CoV-2 --path MASH/replace/21/2000 --n_workers 1 --criterion gini --class_weight None --test 0 --run_num 0
# python train.py --dataset SARS-CoV-2 --path MASH/remove/21/1000 --n_workers 1 --criterion gini --class_weight balanced --test 0 --run_num 0
# python train.py --dataset SARS-CoV-2 --path MASH/remove/21/1000 --n_workers 1 --criterion entropy --class_weight None --test 0 --run_num 0
# python train.py --dataset SARS-CoV-2 --path MASH/remove/21/1000 --n_workers 1 --criterion entropy --class_weight balanced --test 0 --run_num 0