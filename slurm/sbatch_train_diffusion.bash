#!/bin/bash -l
#SBATCH --partition=gpu
#SBATCH -N 1
#SBATCH -c 7             # Cores assigned to each tasks
#SBATCH --time=1-20:00:00
#SBATCH -o %x-%j.out # ./<jobname>-<jobid>.out


while [[ $# -gt 0 ]]
do
    key="$1"
    case $key in
        --config)
            CONFIG="$2"
            shift # past argument
            shift # past value
            ;;
        --batch-size)
            BATCHSIZE="$2"
            shift # past argument
            shift # past value
            ;;
        --num-gpus)
            NUM_GPUS="$2"
            shift # past argument
            shift # past value
            ;;
        *)    # unknown option
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Set the number of GPUs to use in the job
if [ -z "$NUM_GPUS" ]
then
    # Use the default value if NUM_GPUS is not provided
    NUM_GPUS=2
fi

if [ -n "$BATCHSIZE" ]
then
    # Use the default value if NUM_GPUS is not provided
    batchsize_arg="--batch-size $BATCHSIZE"
fi

# Use srun launch with ddp if number of GPUs is greater than 1
SRUN_ARGS=""
if [ $NUM_GPUS -gt 1 ]
then
    SRUN_ARGS="srun"
fi

print_error_and_exit() { echo "***ERROR*** $*"; exit 1; }
module purge || print_error_and_exit "No 'module' command"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export NCCL_DEBUG=INFO
# export PYTHONFAULTHANDLER=1


conda activate lunar_diffusion


$SRUN_ARGS python tools/train.py --config $CONFIG --mode $MODE --num-gpus $NUM_GPUS $batchsize_arg
