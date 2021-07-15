#!/bin/bash
#int-or-string.sh


# updating repo
cd from_git
git fetch --all
git pull
cd ..
echo "Fetched latest files from git"

PARAMS_FILE="pr_paramSearch_params.csv"
LOG_DIR="/home/sidrees/scratch/RetinaPredictors/grid_scripts/pr_paramSearch/logs"

if [ ! -d "$LOG_DIR" ]; then 
	mkdir $LOG_DIR
fi


path_mdl_drive_all=( $(tail -n +2 $PARAMS_FILE | cut -d ',' -f1) )
model_dataset_all=( $(tail -n +2 $PARAMS_FILE | cut -d ',' -f2) )
path_excel_all=( $(tail -n +2 $PARAMS_FILE | cut -d ',' -f3) )
path_perFiles_all=( $(tail -n +2 $PARAMS_FILE | cut -d ',' -f4) )
r_sigma_all=( $(tail -n +2 $PARAMS_FILE | cut -d ',' -f5) )
r_phi_all=( $(tail -n +2 $PARAMS_FILE | cut -d ',' -f6) )
r_eta_all=( $(tail -n +2 $PARAMS_FILE | cut -d ',' -f7) )


numParams=${#r_sigma[@]}
echo "Number of parameter combinations: $numParams"



for ((i=0; i<$numParams; i++));
do
 path_mdl_drive=${path_mdl_drive_all[i]}
 model_dataset=${model_dataset_all[i]}
 path_excel=${path_excel_all[i]}
 path_perFiles=${path_perFiles_all[i]}
 
 r_sigma=${r_sigma_all[i]}
 r_phi=${r_phi_all[i]}
 r_eta=${r_eta_all[i]}
 
 
 

 echo "path_mdl_drive: $path_mdl_drive"
 echo "model_dataset: $model_dataset"
 echo "path_excel: $path_excel"
 echo "path_perFiles: $path_perFiles"
 echo "r_sigma: $r_sigma"
 echo "r_phi: $r_phi"
 echo "r_eta: $r_eta"


 JOB_ID=$(sbatch --export=LOG_DIR=$LOG_DIR,path_mdl_drive=$path_mdl_drive,model_dataset=$model_dataset,path_excel=$path_excel,path_perFiles=$path_perFiles,r_sigma=$r_sigma,r_phi=$r_phi,r_eta=$r_eta pr_paramSearch_launcher.sh)

echo $JOB_ID
JOB_ID=$(echo "$JOB_ID" | grep -Eo '[0-9]{1,6}')
 
printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t' $JOB_ID $expDate $path_mdl_drive $model_dataset $path_excel $path_perFiles $r_sigma $r_phi $r_eta | paste -sd '\t' >> job_list.csv
 
 
done
