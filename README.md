# Whole-slide CNN Training Pipeline

This repository provides scripts to reproduce the results in the paper "An annotation-free whole-slide training approach to pathological classification of lung cancer types by deep learning", including model training, inference, visualization, and statistics calculation, etc.
Also, the pipeline is seamlessly adaptable to other pathological cases by simply creating new configuration files. 

<img src="https://user-images.githubusercontent.com/6285919/122543694-a180ad00-d05e-11eb-8ad0-e8a4b22d29a7.png" width="800" />

## Publication

Chen, CL., Chen, CC., Yu, WH. *et al.* An annotation-free whole-slide training approach to pathological classification of lung cancer types using deep learning. *Nat Commun* **12,** 1193 (2021). https://doi.org/10.1038/s41467-021-21467-y

Chuang, WY., Chen, CC., Yu, WH. *et al.* Identification of nodal micrometastasis in colorectal cancer using deep learning on annotation-free whole-slide images. *Mod Pathol* (2021). https://doi.org/10.1038/s41379-021-00838-2

## License

Copyright (C) 2021 aetherAI Co., Ltd.
All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).

## TCGA Pre-trained Model

A referenced pre-trained weight for lung cancer type classification is now available at https://drive.google.com/file/d/1XuONWICAzJ-cUKjC7uHLS0YLJhbLRoo1/view?usp=sharing.

The model was trained by TCGA-LUAD and TCGA-LUSC diagnostic slides specified in `data_configs/pure_tcga/train_pure_tcga.csv` using the config `train_configs/pure_tcga/config_pure_tcga_wholeslide_4x.yaml`.
Since no normal lung slides were provided in these data sets, the model predicts a slide as either adenocarcinoma (class_id=1) or squamous cell carcinoma (class_id=2).
The prediction scores for normal (class_id=0) should be ignored.

Validation results (*n* = 192) on `data_configs/pure_tcga/val_pure_tcga.csv` are listed as follow.

- AUC (LUAD vs LUSC) = **0.9794** (95% CI: 0.9635-0.9953)
- Accuracy (LUAD vs LUSC) = **0.9323** (95% CI: 0.8876-0.9600, @threshold = 0.7 for class1, 0.3 for class2)

<img src="https://user-images.githubusercontent.com/6285919/122541978-cd029800-d05c-11eb-932c-3cc0c517101e.png" width="400" />

### Error occurred when the pretrained model was loaded.

The number of the layers of the provided model was different to the Resnet 50 or Resnet 32. <br>
Thus, it is NOT compatible with any models that can be selected in model.py.


## Requirements

### Hardware Requirements

Make sure the system contains adequate amount of main memory space (minimal: 256 GB, recommended: 512 GB) to prevent out-of-memory error.
For ones who would like to have a try with less concern about model accuracy, setting a lower resizing ratio and image size in configuration can drastically reduce memory consumption, friendly for limited computing resources.

### Packages

The codes are tested on the environment with Ubuntu 18.04 / CentOS 7.5, Python 3.7.3, cuda 10.0, cudnn 7.6 and Open MPI 4.0.1.

Note) Install a set of CUDA related stuffs compatible with tensorflow version 1.15.

> $conda install cudatoolkit=10.0

> $conda install cudnn=7.6.5

Some Python packages should be installed before running the scripts, including

- Tensorflow v1.x (tensorflow-gpu==1.15.3)
- Horovod (horovod==0.19.0)
- MPI for Python (mpi4py==3.0.3)
- OpenSlide 3.4.1 (https://github.com/openslide/openslide/releases/tag/v3.4.1)
- OpenSlide Python (openslide-python=1.1.1)
- (optional) R 4.0.2 (https://www.r-project.org/)
- Tensorflow Huge Model Support (our package)

Note) The above packages are installed under the folder tensorflow-huge-model-support. 

> $pip install .

Refer to poetry.lock under whole_slide_cnn folder for the full list.
The installation of these packages should take few minutes.

## Usage

### 1. Define Datasets

To initiate a training task, several CSV files, e.g. train.csv, val.csv and test.csv, should be prepared to define training, validation and testing datasets.

These CSV files should follow the format:
```
[slide_name_1],[class_id_1]
[slide_name_2],[class_id_2]
...
```
, where [slide_name_\*] specify the filename **without extension** of a slide image and [class_id_\*] is an integer indicating a slide-level label (e.g. 0 for normal, 1 for cancerous). 

The configuration files for our experiments are placed at data_configs/.

### 2. Set Up Training Configurations

Model hyper-parameters are set up in a YAML file. 

For convenience, you can copy one from train_configs/ (e.g. train_configs/config_wholeslide_2x.yaml) and make modifications for your own recipe.

The following table describes each field in a train_config.
| Field                      | Description
| -------------------------- | ---------------------------------------------------------------------------------------------
| RESULT_DIR                 | Directory to store output stuffs, including model weights, testing results, etc.
| MODEL_PATH                 | Path to store the model weight. (default: `${RESULT_DIR}/model.h5`)
| LOAD_MODEL_BEFORE_TRAIN    | Whether to load the model weight before training. (default: `False`)
| CONFIG_RECORD_PATH         | Path to back up this config file. (default: `${RESULT_DIR}/config.yaml`)
| USE_MIXED_PRECISION        | Whether to enable mixed precision training.
| USE_HMS                    | Whether to enable whole-slide training by optimized unified memory.
| USE_MIL                    | Whether to use MIL for training.
| TRAIN_CSV_PATH             | CSV file defining the training dataset.
| VAL_CSV_PATH               | CSV file defining the validation dataset.
| TEST_CSV_PATH              | CSV file defining the testing dataset.
| SLIDE_DIR                  | Directory containing all the slide image files (can be soft links).
| SLIDE_FILE_EXTENSION       | File extension. (e.g. ".ndpi", ".svs")
| SLIDE_READER               | Library to read slides. (default: `openslide`)
| RESIZE_RATIO               | Resize ratio for downsampling slide images.
| INPUT_SIZE                 | Size of model inputs in [height, width, channels]. Resized images are padded or cropped to the size. Try decreasing this field when main memory are limited.
| MODEL                      | Model architecture to use. One of `fixup_resnet50`, `fixup_resnet34`, `resnet34`, and `frozenbn_resnet50`.
| NUM_CLASSES                | Number of classes.
| BATCH_SIZE                 | Number of slides processed in each training iteration for each MPI worker. (default: 1)
| EPOCHS                     | Maximal number of training epochs.
| NUM_UPDATES_PER_EPOCH      | Number of interations in an epoch.
| INIT_LEARNING_RATE         | Initial learning rate for Adam optimizer.
| POOL_USE                   | Global pooling method in ResNet. One of `gmp` and `gap`.
| REDUCE_LR_FACTOR           | The learning rate will be decreased by this factor upon no validation loss improvement in consequent epochs.
| REDUCE_LR_PATIENCE         | Number of consequent epochs to reduce learning rate.
| TIME_RECORD_PATH           | Path to store a CSV file recording per-iteration training time.
| TEST_TIME_RECORD_PATH      | Path to store a CSV file recording per-iteration inference time.
| TEST_RESULT_PATH           | Path to store the model predictions after testing in a JSON format. (default: `${RESULT_DIR}/test_result.json`)
| USE_TCGA_VAHADANE          | Whether to enable color normalization on TCGA images to TMUH color style. (default: `False`)
| ENABLE_VIZ                 | Whether to draw prediction maps when testing. (default: `False`)
| VIZ_SIZE                   | Size of the output prediction maps in [height, width].
| VIZ_FOLDER                 | Folder to store prediction maps. (default: `${RESULT_DIR}/viz`)

The following fields are valid only when `USE_MIL: True`.
| Field                      | Description
| -------------------------- | ---------------------------------------------------------------------------------------------
| MIL_PATCH_SIZE             | Patch size of the MIL model in [height, width].
| MIL_INFER_BATCH_SIZE       | Batch size for MIL finding representative patches.
| MIL_USE_EM                 | Whether to use EM-MIL.
| MIL_K                      | Number of representative patches. (default: 1)
| MIL_SKIP_WHITE             | Whether to skip white patches. (default: `True`)
| POST_TRAIN_METHOD          | Patch aggregation method to use. One of `svm`, `lr`, `maxfeat_rf`, `milrnn` and `""` (disable).
| POST_TRAIN_MIL_PATCH_SIZE  | (The same as above, for patch aggregation method training process.)
| POST_TRAIN_INIT_LEARNING_RATE | (The same as above, for patch aggregation method training process.)
| POST_TRAIN_REDUCE_LR_FACTOR | (The same as above, for patch aggregation method training process.)
| POST_TRAIN_REDUCE_LR_PATIENCE | (The same as above, for patch aggregation method training process.)
| POST_TRAIN_EPOCHS          | (The same as above, for patch aggregation method training process.)
| POST_TRAIN_NUM_UPDATES_PER_EPOCH | (The same as above, for patch aggregation method training process.)
| POST_TRAIN_MODEL_PATH      | Path to store patch aggregation model weights.

### 3. Train a Model

To train a model, simply run
```
python -m whole_slide_cnn.train --config YOUR_TRAIN_CONFIG.YAML [--continue_mode]
```
, where `--continue_mode` is optional that makes the training process begin after loading the model weights.

To enable multi-node, multi-GPU distributed training, simply add `mpirun` in front of the above command, e.g.
```
mpirun -np 4 -x CUDA_VISIBLE_DEVICES="0,1,2,3" python -m whole_slide_cnn.train --config YOUR_TRAIN_CONFIG.YAML
```

Note that you should `cd` to the root folder of this repo before calling the above commands.

Typically, this step takes days to complete, depending on the computing power, while you can trace the progress in real time from program output.

### Backbones of the model (Note)

In model.py, a dictionary variable called graph_mapping is defined as below.  One of noticeable things is that fixup_resnet50 and frozenbn_resnet50 taking up the weights of the imagenet as initial weights.  The other two are not so their weights are initiated randomly.  

graph_mapping = {

    "resnet34": lambda *args, **kwargs: ResNet34(
        *args, 
        norm_use="bn", 
        weights=None,
        use_fixup=False,
        data_format="channels_last",
        **kwargs
    ),
    
    "fixup_resnet34": lambda *args, **kwargs: ResNet34(
        *args, 
        norm_use="", 
        weights=None,
        use_fixup=True,
        data_format="channels_last",
        **kwargs
    ),
    
    "fixup_resnet50": lambda *args, **kwargs: ResNet50(
        *args, 
        norm_use="", 
        weights="imagenet",
        use_fixup=True, 
        data_format="channels_last",
        **kwargs
    ),
    
    "frozenbn_resnet50": lambda *args, **kwargs: ResNet50(
        *args,
        norm_use="frozen_bn",
        weights="imagenet",
        use_fixup=False,
        data_format="channels_last",
        to_caffe_preproc=True,
        **kwargs
    ),
    
The use of arguments such as norm_use, use_fixup, and to_caffe_preproc seems to be referred by the authors to:

> Reference papers

- [Deep Residual Learning for Image Recognition]
  (https://arxiv.org/abs/1512.03385) (CVPR 2016)
  
- [Fixup Initialization: Residual Learning Without Normalization]
  (https://arxiv.org/abs/1901.09321) (ICLR 2019)

> Reference implementations

- [ResNet]
  (https://github.com/keras-team/keras-applications/blob/master/keras_applications/resnet_common.py)



### 4. (Optional) Post-train Patch Aggregation Model for MIL

EM-MIL-SVM, EM-MIL-LR, MIL-RNN and CNN-MaxFeat-based RF involve training a second patch aggregation model, requiring users to run another script to initiate patch aggregation model training.
Just like the command above, simply call
```
[mpirun ...] python -m whole_slide_cnn.post_train --config YOUR_TRAIN_CONFIG.YAML 
```

### 5. Evaluate the Model

To evaluate the model or optionally generate prediction heatmap, call
```
[mpirun ...] python -m whole_slide_cnn.test --config YOUR_TRAIN_CONFIG.YAML
```

This command will generate a JSON file in the result directory named `test_result.json` by default.
The file contains the model predictions for each testing slide. 

To statistically analyze the results, some scripts are provided in tools/.
See the following table for the usage of each tool.
| Tool                  | Description                                     | Example
| --------------------- | ----------------------------------------------- | ---------------------------------------------
| tools/calc_auc.R      | Calculate AUC and CI.                           | tools/calc_auc.R RESULT_DIR/test_result.json
| tools/compare_auc.R   | Testing significance of the AUCs of two models. | tools/compare_auc.R RESULT_DIR_1/test_result.json RESULT_DIR_2/test_result.json
| tools/draw_roc.py     | Draw the ROC diagram.                           | python tools/draw_roc.py test_result.json:MODEL_NAME:#FF0000
| tools/gen_bootstrap_aucs.R | Generate 100 AUCs by bootstrapping.        | tools/gen_bootstrap_aucs.R RESULT_DIR/test_result.json

Note that these tools are currently profiled for lung cancer maintype classification and should be modified when applying to your own tasks.

## Data Availability

The slide data from TMUH, WFH and SHH are not publicly available due to patient privacy constraints, but are available uponon reasonable request from the corresponding author Chao-Yuan Yeh or Cheng-Yu Chen. 
The slide data supporting the cross-site generalization capability in this study are obtained from TCGA via the Genomic Data Commons Data Portal (https://gdc.cancer.gov).

A dataset consists of several slides from TCGA-LUAD and TCGA-LUSC is suitable for testing our pipeline in small scale, with some proper modifications of configuration files described above.

## Results

### Problem encountered when the hardware requirement was not met

We tried the training the Resnet50, initialized by the weights obtained by training with Imagenet.  The image size was set to be at 5500 x 5500 with the Resize factor of 0.05, representing the magnification at 1x.

The loss and accuracy curves were plotted upon training the model through 100 epochs.  The validation accuracy reached about 0.80 with the validation loss of 0.41.

![image](https://user-images.githubusercontent.com/64822593/198936803-2a2fb8d3-d3b2-4009-b9d9-e54b24d96e79.png)

To improve the accuracy, we tried the training with 2x images upon loading the model.h5 provided by the authors.  Although it was trained at 4x but utilized for our training at 2x.  The validation accuracy reached about 0.89 with the validation loss of 0.25 within 100 epochs, that was significantly increased from the initial training at 1x.  Out hardware spec. remained the same as for the 1x training (CPU RAM : 128 GB).

![image](https://user-images.githubusercontent.com/64822593/201279646-b3c4170d-2cc1-4f87-b32d-6486e306f473.png)

The model was evaluated visually by grad-CAM that depicts the likelihood of the tumor in the tissue. The image below highlights where the lung cancer cells are likely located (LUAD class).  The middle image was generated using the ResNet50 model trained from the weights of Imagenet and shows somewhat large tissue area of false positive.  The right image was generated using the Resnet model trained in a continous mode after loading the previously trained model (at 4x) and shows much tighter marking of the tumor tissue.  
![TCGA-49-4494-01Z-00-DX5 1e9e22d6-a4c9-40d1-aedb-b6cd404fe16f svs](https://user-images.githubusercontent.com/64822593/199023104-e39cd3af-4445-4ee2-86fa-cf4700b08b52.png)

