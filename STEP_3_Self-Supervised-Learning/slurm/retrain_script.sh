#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --nodelist master
#SBATCH --cpus-per-task=36
#SBATCH --mem-per-cpu=2000
#SBATCH --gpus=4
#SBATCH --qos interactive
#SBATCH --time 48:00:00
#SBATCH --mail-user=dhkdgmlghks@snu.ac.kr
#SBATCH --mail-type=BEGIN


cd  /scratch/connectome/dhkdgmlghks/3DCNN_test/simCLR_UKB
python3 run_simCLR.py --optim LAMB --lr 1e-3  --epoch 25 --model densenet3D121 --train_batch_size 88  --exp_name UKB_simCLR  --augmentation  RandRotate RandCrop RandFlip RandAdjustContrast RandGaussianSmooth RandGibbsNoise --resize 80 80 80 --accumulation_steps 20 --resume True --checkpoint_dir /scratch/connectome/dhkdgmlghks/3DCNN_test/simCLR_UKB/result/model/densenet3D121_UKB_simCLR_16f2e0.pth  --sbatch True
