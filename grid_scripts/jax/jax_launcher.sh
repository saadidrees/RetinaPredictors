#!/bin/bash
#int-or-string.sh
#SBATCH --job-name=MAML
#SBATCH --account=rrg-tyrell-ab
#SBATCH --time=5:00:00
#SBATCH --gres=gpu:1 --cpus-per-task=1
#SBATCH --mem-per-cpu=32G
#SBATCH -o logs/%j-out.txt
#SBATCH -e logs/%j-error.txt





cd $SLURM_TMPDIR
echo "directory changed"
echo $PWD
mkdir RetinaPredictors
cp -r ~/scratch/RetinaPredictors/grid_scripts/from_git/ $SLURM_TMPDIR/
cd $SLURM_TMPDIR/from_git
echo "directory changed"
echo $PWD
ls -l
echo "files copied"

#module load python/3.6 cuda cudnn
module load python/3.11 cuda cudnn
export XLA_FLAGS=--xla_gpu_cuda_data_dir=$CUDA_HOME
virtualenv --no-download  $SLURM_TMPDIR/jax_env
source $SLURM_TMPDIR/jax_env
pip install --no-index scipy==1.13.1 matplotlib==3.9.0 tensorflow==2.16.1 jax==0.4.30 flax==0.8.5 optax==0.2.3 pillow==10.3.0 h5py==3.11 numpy==1.26.4 cloudpickle==3.0.0 torch==2.3.1
export XLA_FLAGS=--xla_gpu_cuda_data_dir=$CUDA_HOME
#export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/libtiff.so.5/:$LD_LIBRARY_PATH



python -u jax_run_models.py $expFold $mdl_name $path_model_save_base $fname_data_train_val_test --path_existing_mdl=$path_existing_mdl --runOnCluster=$runOnCluster --chan1_n=$chan1_n --filt1_size=$filt1_size --filt1_3rdDim=$filt1_3rdDim --chan2_n=$chan2_n --filt2_size=$filt2_size --filt2_3rdDim=$filt2_3rdDim --chan3_n=$chan3_n --filt3_size=$filt3_size --filt3_3rdDim=$filt3_3rdDim --chan4_n=$chan4_n --filt4_size=$filt4_size --filt4_3rdDim=$filt4_3rdDim --nb_epochs=$nb_epochs --thresh_rr=$thresh_rr --temporal_width=$temporal_width --pr_temporal_width=$pr_temporal_width --pr_params_name=$pr_params_name --bz_ms=$bz_ms --BatchNorm=$BatchNorm --MaxPool=$MaxPool --c_trial=$c_trial --USE_CHUNKER=$use_chunker --trainingSamps_dur=$TRSAMPS --validationSamps_dur=$VALSAMPS --lr=$lr --lrscheduler=$lrscheduler --idx_unitsToTake=$idx_unitsToTake --idxStart_fixedLayers=$idxStart_fixedLayers --idxEnd_fixedLayers=$idxEnd_fixedLayers --select_rgctype=$select_rgctype --CONTINUE_TRAINING=1 --path_dataset_base=/home/sidrees/scratch/RetinaPredictors/data/data_ej/
