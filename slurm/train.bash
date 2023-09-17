#!/bin/bash

while [[ $# -gt 0 ]]
do
    key="$1"
    case $key in
        -c|--config)
            CONFIG_PATH="$2"
            shift # past argument
            shift # past value
            ;;
        -g|--num-gpus)
            NUM_GPUS="$2"
            shift # past argument
            shift # past value
            ;;
        -b|--batch-size)
            BATCHSIZE="$2"
            shift # past argument
            shift # past value
            ;;
        -t|--train-script)
            TRAIN_SCRIPT="$2"
            shift # past argument
            shift # past value
            ;;
        -h|--help)
            echo "Usage: bash $0 --config <relative/path/to/config> --batchsize <batch_size> [--num-gpus <num_gpus>] [--mode <mode>]"
            echo "Options:"
            echo "  -c, --config <relative/path/to/config>  Required relative path to the configuration file"
            echo "  -b, --batchsize <batch_size>            Required batch size"
            echo "  -g, --num-gpus <num_gpus>               Number of GPUs to use, default is 2"
            echo "  -m, --mode <mode>                       Mode to run, default is 'vae'"
            echo "  -h, --help                          Show this help message and exit"
            exit 0
            ;;
        *)    # unknown option
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done


# default values for optional parameters
NUM_GPUS=${NUM_GPUS:-2}


if [ -z "$CONFIG_PATH" ] || [ -z "$BATCHSIZE" ]; then
    echo "ERROR: --config and --batchsize are required parameters."
    exit 1
fi

CONFIG_PATH=$(readlink -f "$CONFIG_PATH")
EXP_NAME=$(basename "$CONFIG_PATH" .py)

OUTDIR="$(dirname "$0")/outs"

if [ ! -d "$OUTDIR" ]; then
    mkdir "$OUTDIR"
fi

# setting the outfile path
OUT_FILE="$OUTDIR/%x-%j.out"

if [ -z "$TRAIN_SCRIPT" ]
then
    sbatch -J $EXP_NAME --gpus $NUM_GPUS --ntasks-per-node $NUM_GPUS --output $OUT_FILE \
        sbatch_train_diffusion.bash --config $CONFIG_PATH --batch-size $BATCHSIZE --num-gpus $NUM_GPUS
else
    sbatch -J $EXP_NAME --gpus $NUM_GPUS --ntasks-per-node $NUM_GPUS --output $OUT_FILE \
        $TRAIN_SCRIPT --config $CONFIG_PATH --batch-size $BATCHSIZE --num-gpus $NUM_GPUS
fi
