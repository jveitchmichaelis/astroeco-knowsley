MODEL_DIR=/home/josh/data/knowsley_final_split/trained_models/models/faster_rcnn_resnet101
PIPELINE_CONFIG_PATH=$MODEL_DIR/pipeline.config
SAMPLE_1_OF_N_EVAL_EXAMPLES=1
TF_DIR=/home/josh/data/knowsley_final_split/models/research/slim
export PYTHONPATH=$PYTHONPATH:$TF_DIR:$TF_DIR/slim

while true
do
python /home/josh/data/knowsley_final_split/models/research/object_detection/model_main.py \
    --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
    --model_dir=${MODEL_DIR} \
    --sample_1_of_n_eval_examples=$SAMPLE_1_OF_N_EVAL_EXAMPLES \
    --alsologtostderr

sleep 1
done
