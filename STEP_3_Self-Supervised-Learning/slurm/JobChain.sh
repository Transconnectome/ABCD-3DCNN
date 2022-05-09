#!/bin/bash
  
#initial script
START=$(sbatch --parsable UKB_simCLR.sh)

#retraining script
ID=$(sbatch --parsable --dependency=afterok:${START} UKB_simCLR_retrain.sh)

for i in $(seq $1)
do
        ID=$(sbatch --parsable --dependency=afterok:${ID} UKB_simCLR_retrain.sh)
done
         
