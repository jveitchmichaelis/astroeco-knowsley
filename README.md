## Training models with Darknet

To train darknet models you need the following:

* Folders with train, validation and test images + labels in suitable format

  * For Darknet this means an image and a `.txt` file with the same base name containing annotations for that image
  * Actually you don't need separate folders, but it helps

* Image lists - text files containing paths to images in each training set

  * One for each of train/test/val
  * They don't need to be images in the same folder, but as above, it helps a bit for organisation.

* A `.names` file containing the class names

* A `.data` file which contains something like:

  * ```
    classes= 4
    train  = G:/data/knowsley_final_split/train_clean_10.txt
    valid  = G:/data/knowsley_final_split/val.txt
    names = G:/data/knowsley_final_split/obj.names
    backup = G:/data/knowsley_final_split/trained_models/models/yolov4
    eval=coco
    ```

  * This tells Darknet where to find the train/validation images, class names, and a backup folder where weights will be stored.

### Evaluating your model

* Darknet has the ability to calculate various mAP metrics
* Make a folder called `/results` where you run `darknet` - this is somewhat awkward. This isn't the folder where `darknet.exe` is, it's where you call the exe _from_. 
  * Someone should really PR this and fix it so it's a config file parameter
* Then run `darknet.exe detector map [data] [cfg] [weights]`
  * There is also a validation option which will do something similar, but it's mostly geared towards evaluation of MS COCO. A `.json` file will be created in the results folder
* 

## Training models with Ultralytics (YOLOV5 etc)

* Clone the repository:

  * ```
    $ git clone https://github.com/ultralytics/yolov5  # clone repo
    $ cd yolov5
    $ pip install -r requirements.txt  # install dependencies
    ```

  * You might need to use Conda to install pytorch

  *  `python C:/Users/Josh/code/yolov5/train.py --img 640 --batch 16 --epochs 5 --data .\knowsley_final_ultralytics.yaml --weights C:\Users\Josh\code\yolov5\yolov5s.pt --hyp .\hyp.scratch.yaml`

## Training models with Tensorflow

1. Make a new conda environment and install:

   1. `pip install tensorflow-gpu==1.5`
   2. `conda install -c anaconda cudnn`

2. Setup folder structure:

   1. Clone tensorflow/models into the current folder: `git clone https://github.com/tensorflow/models.git`
   2. Follow the install instructions [here](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md)
   3. Make a folder for your training results (e.g. `mkdir trained_models`)
      1. Make two subfolders for `data` (storing your records) and `models` (to store each model)

3. Create three files containing images paths for train/val/test splits. This guide assumes you've exported your images with darknet-style labels (e.g. /path/to/image.jpg + /path/to/image.txt)

4. Run `make_tfrecord.py` for each of the text files, to create `train.tfrecord` etc

   1. This code will add 1 to each of your class IDs, since darknet starts with zero
   2. The code also contains checks for things that Tensorflow hates: objects with bounding boxes that extend outside the image, bounding boxes with zero area, etc.
   3. Copy these files into the `./trained_models/data folder`

5. Ensure that your `labels.pbtxt` matches up with your names file. **If you have any mismatch with class ID and the name in the tfrecord, Tensorflow will fail with a NaN loss**. Remember that Tensorflow IDs are indexed from 1, not 0.

6. Check the pipeline config for you model:

   1. Adjust the train and evaluation paths, as well as the label.pbtxt path
   2. Check that the **batch size** is not enormous. Google trains a lot of the "zoo" models on enormous machines with a batch size of e.g. 512. This will slaughter your RAM. It's best to start out with 1 and if it works, double until you hit problems. For Faster-RCNN on a 1080 you can probably only do 1 anyway. For edge models, you might be able to get away with higher.
   3. Double the number of training steps if you're training from a model zoo checkpoint, otherwise the API will assume you're already finished (oops).
   4. The number of evaluations is default 10, this is the number of images that get shown to you in tensorboard. If you set shuffle to true (default) then you'll get a new batch of images each time. I find this more useful than seeing the same batch get better, but do as you prefer.

7. Modify a `train_<model>.sh` example appropriately:

   1. Set the path to the model folder, which should have pre-trained weights in (grab from the Tensorflow model zoo)

   2. For example:

      ```
      MODEL_DIR=/home/josh/data/knowsley_split/trained_models/models/edgetpu_quant_330
      PIPELINE_CONFIG_PATH=$MODEL_DIR/pipeline.config
      
      SAMPLE_1_OF_N_EVAL_EXAMPLES=1
      TF_DIR=/home/josh/data/knowsley_final_split/models/research
      export PYTHONPATH=$PYTHONPATH:$TF_DIR:$TF_DIR/slim
      export TF_CPP_MIN_LOG_LEVEL="3"
      
      python $TF_DIR/research/object_detection/model_main.py \
          --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
          --model_dir=${MODEL_DIR} \
          --sample_1_of_n_eval_examples=$SAMPLE_1_OF_N_EVAL_EXAMPLES \
          --alsologtostderr
      
      ```

   3. Double check that the tensorflow directory is correct, as this is used to setup the python path.

8. Run!

9. Fire up something like `htop` and `watch nvidia-smi` to check what's going on. Also load tensorboard and check that things look normal.

   1. After each epoch, you should get an image output which compares detections and ground truth. This is extremely useful to check if you imported your labels properly!
   2. Sit back and wait while everything trains. Performance will be awful to start with, but should improve fairly after an epoch or two.

#### Creating intermediate files and trimming datasets

* List all files in a folder and exclude text (label files) `find ./train -type f -maxdepth 1 -not -name "*.txt" > train.txt` . This is particularly good as using wildcard globbing with `ls` , eg. `ls *.jpg > file` doesn't work in huge directories.
* Taking 1 in 10 images - recommended for movies: `cat train.txt | sed '0~10!d' > train_10.txt` . Sed to the rescue.
* Shuffling lists: `shuf` e.g. `cat train.txt | shuf > train_shuffled.txt`

## Sanity checks

* You should see ground truth bounding boxes in the eval stage. If you don't (and these aren't negative examples), something has gone wrong and you should check your dataset.

* For an object detection model using transfer learning, unless your input data is wildly different, you should start seeing reasonable bounding boxes after the first epoch.

  If you don't, the first thing to try is **lowering your learning rate** because you may simply be jumping out of the minimum in the "loss landscape". Try lowering it to e.g. 2 orders of magnitude below the default pipeline configuration.

## Troubleshooting

The object detection API _should_ just work, but often it takes some effort to get things running. Mostly errors you have will be related to your dataset, not the workings of Tensorflow. Unfortunately Tensorflow is renowned for it's extremely opaque error messages. It doesn't help that the Object Detection API works only with TF1 and not TF2 (yet).

### Evaluation takes forever

This can happen if you have an enormous test dataset. There is some kind of bug where, if your evaluation takes too long, you end up in an infinite loop where the API is continually trying to evaluate your model. You can fix it by adjusting:

`SAMPLE_1_OF_N_EVAL_EXAMPLES` in the launch script, to e.g. 5 which will use only 20% of the evaluation set.

You can adjust `num_readers` in the `eval_input_reader` to be e.g. 8 which should speed up the process generally.

As a last resort, you can follow [this](https://github.com/tensorflow/models/issues/6106#issuecomment-471130489) guide and add a throttle parameter to the evaluator so it will only run e.g. once an hour. The suggestion of 18000 secs may be a bit long. Try 3600 to start with.

### Errors related to CuDNN

Check that CuDNN and CUDA are installed and are the right versions for each other.

### Training ends immediately

This is probably because you used a pre-trained checkpoint and you didn't adjust the number of steps that you're training the model for (and Tensorflow thinks you're finished).

### My computer stutters to a halt shortly after starting to train

Probably this is an Out Of Memory (OOM) problem. Check the batch size in the pipeline configuration file and set it to 1 initially, then slowly increase until you hit your RAM limit.

### I get big/diverging/NaN losses

This is usually due to one of two things.

##### Shortly after starting:

If you see NaNs immediately or very shortly after training, check your dataset. Tensorflow is **extremely** picky about what you feed into it. This can be something as simple as:

* You have a bounding box which has:
  * Negative coordinates
  * Zero area
  * Other coordinates outside the range of [0, 1]
* An image is somehow corrupted - e.g. maybe you interrupted tf record generation.
* Your class name/ID doesn't match your labels.pbtxt file. This is a weird one, but it can skewer you.
* You have an invalid class name - for example if there's a _space_, I believe this can cause issues sometimes.

##### During training:

Once you've ruled out dataset issues, it's likely to be a learning rate problem. Try halving it, or decreasing by an order of magnitude. If your loss doesn't explode this time, then try increasing it slightly.

When you're fine-tuning, you're often using pipeline configurations that were used to train from scratch. The base learning rate is very large for a fine-tuning step because at this stage the model should be very good and the learning rate (at 400k steps) is probably an order of magnitude or two lower. For example, if you try training a Mobilenet EdgeTPU model with the base rate of 0.8, try decreasing it to 0.1 with a warm up of 0.01.

You can adjust things like:

* The base learning rate (try halving it)
* Adjusting the cycle length, if using a cosine decay (`total_steps`)

You should also check what checkpoint you start training from. You can use pre-trained weights, but if you don't have the pre-trained _checkpoint_ in your training folder, then you'll start from step 0. Otherwise you'll start from the step at which your pre-trained model stopped. This can have a profound difference on the current learning rate.

Finally, try training from scratch as a baseline.

### Classification loss increases

This is likely due to a learning rate issue, try lowering it by an order of magnitude.

Next, try using a different weighting on your loss function.