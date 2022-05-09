# slurm Job Chain 
**JobChain.sh** is a script for slurm Job Chain.  
To run this code you need **initial script** and **retrain script**.  
**Initial script** set ```--resume False``` and **retrain script** set ```--resume True``` option of simCLR python script.  

For example, run Job Chain script as follow.
```bash JobChain.sh 4 #the number of running retrain script```
