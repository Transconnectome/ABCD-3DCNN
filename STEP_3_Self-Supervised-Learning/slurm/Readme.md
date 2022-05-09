# slurm Job Chain 
**JobChain.sh** is a script for slurm Job Chain.  
To run this code you need **retrain script**.  
***Before you run Job Chain code, you should run **intitial script*****
**Initial script** set ```--resume False``` and **retrain script** set ```--resume True``` option of simCLR python script.  

For example, run Job Chain script as follow.
```
#the number indicates the number of running retrain script
bash JobChain.sh 4 
```
