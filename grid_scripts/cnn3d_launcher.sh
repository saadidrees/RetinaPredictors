#!/bin/bash
#int-or-string.sh
#SBATCH --job-name=CNN_3D
#SBATCH --account=def-joelzy
#SBATCH --time=30:00:00
#SBATCH --gres=gpu:1 --cpus-per-task=2
#SBATCH --mem-per-cpu=36000M
#SBATCH -o $LOG_DIR/%j-out.txt
#SBATCH -e $LOG_DIR/%j-error.txt



module load python/3.6 cuda cudnn

cd $SLURM_TMPDIR
git clone https://saadidrees:Trespasser987@github.com/saadidrees/RetinaPredictors.git
cd $SLURM_TMPDIR/RetinaPredictors

#cp /home/sidrees/scratch/dynamic_retina/models/cluster_run_model_cnn3d.py $SLURM_TMPDIR/
#cp -R /home/sidrees/scratch/dynamic_retina/models/model $SLURM_TMPDIR/

virtualenv --no-download  $SLURM_TMPDIR/tensorflow_env
source $SLURM_TMPDIR/tensorflow_env
pip install --no-index scipy matplotlib tensorflow_gpu



python run_model_cnn3d.py $expDate --runOnCluster=$runOnCluster --chan1_n=$chan1_n --filt1_size=$filt1_size --filt1_3rdDim=$filt1_3rdDim --chan2_n=$chan2_n --filt2_size=$filt2_size --filt2_3rdDim=$filt2_3rdDim --chan3_n=$chan3_n --filt3_size=$filt3_size --filt3_3rdDim=$filt3_3rdDim --nb_epochs=$nb_epochs --thresh_rr=$thresh_rr --temporal_width=$temporal_width --bz_ms=$bz_ms --BatchNorm=$BatchNorm --MaxPool=$MaxPool
