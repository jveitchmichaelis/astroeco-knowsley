TIME=`date +"%y%m%d_%H%M"`
BASE_PATH=/home/josh/data/knowsley_final_split/trained_models/models/edgetpu_quant_330/
MODEL_DIR=$BASE_PATH/$TIME
mkdir -p $MODEL_DIR
PIPELINE_CONFIG_PATH=$BASE_PATH/pipeline.config

SAMPLE_1_OF_N_EVAL_EXAMPLES=5
TF_DIR=/home/josh/data/knowsley_final_split/models/research
export PYTHONPATH=$PYTHONPATH:$TF_DIR:$TF_DIR/slim
export TF_CPP_MIN_LOG_LEVEL="3"

# Backup pipeline
cp ${PIPELINE_CONFIG_PATH} $MODEL_DIR

python $TF_DIR/object_detection/model_main.py \
    --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
    --model_dir=$MODEL_DIR \
    --sample_1_of_n_eval_examples=$SAMPLE_1_OF_N_EVAL_EXAMPLES \
    --alsologtostderr
