MODEL_DIR=/home/josh/data/knowsley_labelled/tfrecord/models/mobilenet_ssd
PIPELINE_CONFIG_PATH=$MODEL_DIR/pipeline.config

SAMPLE_1_OF_N_EVAL_EXAMPLES=1
TF_DIR=/home/josh/data/knowsley_labelled/models/research/slim
export PYTHONPATH=$PYTHONPATH:$TF_DIR:$TF_DIR/slim
export PYTHONPATH=$PYTHONPATH:/home/josh/data/knowsley_labelled/models/research

export TF_CPP_MIN_LOG_LEVEL="3"

while true
do
python /home/josh/data/knowsley_labelled/models/research/object_detection/model_main.py \
    --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
    --model_dir=${MODEL_DIR} \
    --sample_1_of_n_eval_examples=$SAMPLE_1_OF_N_EVAL_EXAMPLES \
    --alsologtostderr

sleep 1
done
