# Whole slide image (WSI) training CNN Pipeline

### Short Introduction
The WSI is the digital image representing the entire histologic slide captured by a high-throughput scanner such as Leica's Aperio scanner.  The pixel size of a WSI is often larger than 100,000 by 100,000 reaching upto 10 GB. Each individual WSI can be easily labeled by using the corresponding clinical information readily available at the deposit site such as the NCI's GDC data portal, for example.

It is widely known that training a deep neural network with a large WSI dataset is more practical since it does not require a time consuming pixel level annotation. However, to achieve a high accuracy, the WSI input should be provided at least 5x or even higher, requiring a computing resource with an adequately large memory close to 512 GB, for example.

![image](https://user-images.githubusercontent.com/64822593/208895067-51db8300-c49d-489a-b19a-d90a16bfc849.png)
<p align="center">
Whole Slide Image(WSI) is a tiled TIFF.  Captured from TCGA-49-4494-01Z-00-DX5.1e9e22d6-a4c9-40d1-aedb-b6cd404fe16f.svs
</p> 

The computing resource even with multiple GPUs often lacks this much memory and ends up with the out of memory(OOM) error.  To solve this problem, IBM developed Large Model Support(LMS) that takes a user-defined computational graph and swaps tensors from GPUs to the host memory and vice versa [For more, see (https://www.ibm.com/docs/en/wmlce/1.6.0?topic=gsmf-getting-started-tensorflow-large-model-support-tflms-v2)].  

The authors who set up this depository originally had also developed a huge model support(HMS) function, probably similar to the IBM's LMS in a nutshell [1][2].  The scripts provided by the authors let us to reproduce their results by WSI input DL training, inference, visualization, and statistics calculation, etc[1].

The exactly same pipeline should be utilized to other human cancers such as prostate cancers or breast cancers by simply creating new configuration files (and label files in addition which is a piece of cake).

![image](https://user-images.githubusercontent.com/64822593/208655851-6986f3e5-84c0-46ab-9194-b35887bf83a0.png)

## Requirements

### Hardware Requirements

The computing resource should provide an adequate amount of main memory space (minimal: 256 GB, recommended: 512 GB) to prevent out-of-memory(OOM) error and fully explore the utility of the huge memory support(HMS) function.  

However, a trial training could be performed by adjusting the resizing ratio at 0.05 and the size of an input image to 5000 x 5000, for example, in the configuration file(0.1 and 11000 x 11000 also worked for the server with 128 GB memory).  It would result in the lower accuracy values ranging from 80 to 90%.

### Packages

The codes were tested on the environment with Ubuntu 18.04, Python 3.7.3, cuda 10.0, cudnn 7.6 and Open MPI 4.0.1.

More specifically, a set of CUDA related stuffs compatible with tensorflow version 1.15 were installed as below
```
conda install cudatoolkit=10.0

conda install cudnn=7.6.5
```
Python packages were installed before running the scripts, including

- Tensorflow v1.x (tensorflow-gpu==1.15.3)
- Horovod (horovod==0.19.0)
- MPI for Python (mpi4py==3.0.3)
- OpenSlide 3.4.1 (https://github.com/openslide/openslide/releases/tag/v3.4.1)
- OpenSlide Python (openslide-python=1.1.1)
- (optional) R 4.0.2 (https://www.r-project.org/)
- Tensorflow Huge Model Support (our package)

The above packages were installed under the folder tensorflow-huge-model-support by running the command shown below.
```
pip install .
```
Refer to poetry.lock under whole_slide_cnn folder for the full list.

## Methods
### 1. Datasets and Configurations
The .csv files under data_configs folder were used as they were without any modification. A detailed description was available by the authors as in the Appendix below.  Hyperparameters were set up in a YAML file (config_wholeslide_2x.yaml) under train_configs folder.  
In the YAML file, the parameters of RESIZE_RATIO and INPUT_SIZE were set appropriately to avoid the OOM error.  One example is shown below:
```
RESIZE_RATIO: 0.05
INPUT_SIZE: [5500, 5500, 3]
NUM_UPDATES_PER_EPOCH: 80
```
### 2. Train a Model

To train a model, the following script was run:
```
python -m whole_slide_cnn.train --config config_wholeslide_2x.yaml [--continue_mode]
```
, where `--continue_mode` is optional that makes the training process begin after loading the model weights.

To enable multi-node, multi-GPU distributed training, simply add `mpirun` in front of the above command, e.g.
```
mpirun -np 4 -x CUDA_VISIBLE_DEVICES="0,1,2,3" python -m whole_slide_cnn.train --config config_wholeslide_2x.yaml
```

Note) You should be at the root folder of this repository when calling the above commands.

### 3. Evaluate the Model

The model was evaluated by calling the command as below and optionally a prediction heatmap was also generated.
```
python -m whole_slide_cnn.test --config config_wholeslide_2x.yaml
```
This command generated a JSON file in the result directory named `test_result.json` by default.
The file contained the model predictions for each testing slide. To further generate the AUC values and their graphs, more tools were available, as explained in the Appendix.

Note) These tools are currently profiled for lung cancer maintype classification and should be modified when applying to your own tasks.

## Results (Draft)

### 1. Performance of Resnet 34 with the WSI at 1x magnification
We first tried the training with the Resnet34, initialized by the weights obtained by training with Imagenet.  The image size was set to be at 5500 x 5500 with the resize factor of 0.05, representing the magnification at 1x.  The loss and accuracy curves were plotted upon training the model through 100 epochs.  The validation accuracy reached about 0.68 with the validation loss of 0.64.
![image](https://user-images.githubusercontent.com/64822593/201547097-89a4b7f7-9218-4250-964d-1f564bb60266.png)

### 2. Performance of Resnet 50 with the WSI at 1x and 2x magnification
We then tried the training with the Resnet50, initialized by the weights obtained by training with Imagenet, at the same magnification. The validation accuracy reached about 0.80 with the validation loss of 0.41.

![image](https://user-images.githubusercontent.com/64822593/198936803-2a2fb8d3-d3b2-4009-b9d9-e54b24d96e79.png)

To improve the accuracy, we tried the training with 2x images upon loading the model.h5 provided by the authors.  Although it was trained at 4x, it could be utilized for the training at 2x.  The validation accuracy reached about 0.89 with the validation loss of 0.25 within 100 epochs, that was significantly increased from the initial training at 1x.  Out hardware spec. remained the same as for the 1x training (CPU RAM : 128 GB).

![image](https://user-images.githubusercontent.com/64822593/201279646-b3c4170d-2cc1-4f87-b32d-6486e306f473.png)

### 3. Visualization of grad-CAM
The model was evaluated visually by grad-CAM that depicts the likelihood of the tumor in the tissue(panel A in the figure). The image below highlights where the lung cancer cells are likely located in an LUAD case.  

The second image was generated using the ResNet50 model trained at the 1x magnification from the weights of Imagenet and shows somewhat large tissue area of false positive(panel B).  

The third image was generated using the Resnet model trained at the same magnification in a continous mode after loading the previously trained model (at 4x) and seemed to show much tighter marking of the tumor tissue(panel C).  

Finally, the fourth image was generated using the Resnet model trained at the 2x in the continous mode and seems to show much larger marking area of the tumor tissue(panel D).  At the moment, the marking has not been validated by an expert.
![image](https://user-images.githubusercontent.com/64822593/201546476-51062b10-2bd1-4e96-a929-65078ae32f0b.png)

### 4. Problems encountered when the hardware requirement was not met
The computing server consisted of an NVIDIA Quadro RTX 6000 GPU that had 24 GB memory capable of runnning at 14~16 TFlops in single precision.  The CPU memory was 128 GB.  

Due to the memory requirement of the algorithm, only images at the magnification at the 2x were suitable for the training at the server.  Thus, higher resolution images at 5x or above could not be used for the training. 

When the magnification of the input images was 1x, the accuracy values were 0.68 and 0.80 for Resnet 34 and Resnet 50, respectively.  When the magnification was increased to 2x, the accuracy was increased to 0.89 in Resnet 50.  This suggested that the higher resolution of the input images helped to improve the performance. However, we could not improve the accuracy further due to the limitation of the memory in the server for the 4x images.

### 5. Computing time
The computing time at the 1x was about 4.3 sec per batch and the batch was set to 80.  Total computing time was 115 mins when the number of the total batch for the training of ResNet50 was 1,600.  The computing time at the 2x was about 46 sec per batch.  Total computing time was about 102 hours when the number of the batch to process was 8,000.

## Acknowledgement
The computing server was kindly provided by the National Internet Promotion Agency(NIPA) of south Korea.

## References

[1] Chi-Long Chen, Chi-Chung Chen, Wei-Hsiang Yu, Szu-Hua Chen, Yu-Chan Chang, Tai-I Hsu, Michael Hsiao, Chao-Yuan Yeh$ & Cheng-Yu Chen$  An annotation-free whole-slide training approach to pathological classification of lung cancer types using deep learning. *Nat Commun* **12,** 1193 (2021). https://doi.org/10.1038/s41467-021-21467-y

[2] Wen-Yu Chuang, Chi-Chung Chen, Wei-Hsiang Yu, Chi-Ju Yeh, Shang-Hung Chang, Shir-Hwa Ueng, Tong-Hong Wang, Chuen Hsueh, Chang-Fu Kuo & Chao-Yuan Yeh$ Identification of nodal micrometastasis in colorectal cancer using deep learning on annotation-free whole-slide images. *Mod Pathol* (2021). https://doi.org/10.1038/s41379-021-00838-2

## Appendix
mostly taken from the original README.MD and 
some more stuffs learning while trying this and that...

## License

Copyright (C) 2021 aetherAI Co., Ltd.
All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).

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

### 3. Generating AUC and ROC Graphs
To statistically analyze the results, some scripts were provided in tools/.
See the following table for the usage of each tool.
| Tool                  | Description                                     | Example
| --------------------- | ----------------------------------------------- | ---------------------------------------------
| tools/calc_auc.R      | Calculate AUC and CI.                           | tools/calc_auc.R RESULT_DIR/test_result.json
| tools/compare_auc.R   | Testing significance of the AUCs of two models. | tools/compare_auc.R RESULT_DIR_1/test_result.json RESULT_DIR_2/test_result.json
| tools/draw_roc.py     | Draw the ROC diagram.                           | python tools/draw_roc.py test_result.json:MODEL_NAME:#FF0000
| tools/gen_bootstrap_aucs.R | Generate 100 AUCs by bootstrapping.        | tools/gen_bootstrap_aucs.R RESULT_DIR/test_result.json
| tools/print_scores.py | Print scores from test_result.json              | python tools/print_scores.py RESULT_DIR/test_result.json --column adeno (is_adeno, is_squamous, or squamous can be given)

### 4. Useful pre-trained Model

A pre-trained weight was obtained from https://drive.google.com/file/d/1XuONWICAzJ-cUKjC7uHLS0YLJhbLRoo1/view?usp=sharing kindly provided by the authors.

The model was trained by TCGA-LUAD and TCGA-LUSC diagnostic slides specified in `data_configs/pure_tcga/train_pure_tcga.csv` using the config `train_configs/pure_tcga/config_pure_tcga_wholeslide_4x.yaml`.

Since no normal lung slides were provided in these data sets, the model predicts a slide as either adenocarcinoma (class_id=1) or squamous cell carcinoma (class_id=2).

The prediction scores for normal (class_id=0) should be ignored.

Validation results (*n* = 192) on `data_configs/pure_tcga/val_pure_tcga.csv` are listed as follow.

- AUC (LUAD vs LUSC) = **0.9794** (95% CI: 0.9635-0.9953)
- Accuracy (LUAD vs LUSC) = **0.9323** (95% CI: 0.8876-0.9600, @threshold = 0.7 for class1, 0.3 for class2)

<img src="https://user-images.githubusercontent.com/6285919/122541978-cd029800-d05c-11eb-932c-3cc0c517101e.png" width="400" />

### 5. Data Availability

The slide data supporting the cross-site generalization capability in this study are obtained from TCGA via the Genomic Data Commons Data Portal (https://gdc.cancer.gov).

A dataset consists of several slides from TCGA-LUAD and TCGA-LUSC is suitable for testing our pipeline in small scale, with some proper modifications of configuration files described above.

### 6. Error occurred when the pretrained model was loaded.

The number of the layers of the provided model was different to the Resnet 50 or Resnet 32. <br>
Thus, it is NOT compatible with any models that can be selected in model.py.

### 7. Backbones of the model 

In model.py, a dictionary variable called graph_mapping was defined as below.  'fixup_resnet50' and 'frozenbn_resnet50' were taking up the weights of the imagenet as initial weights.  The other two, 'resnet34' and 'fixup_resnet34', were initialized randomly.  

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
    
The use of arguments such as norm_use, use_fixup, and to_caffe_preproc seemed to be referred by the authors to:

> Reference papers

- [Deep Residual Learning for Image Recognition]
  (https://arxiv.org/abs/1512.03385) (CVPR 2016)
  
- [Fixup Initialization: Residual Learning Without Normalization]
  (https://arxiv.org/abs/1901.09321) (ICLR 2019)

> Reference implementations

- [ResNet]
  (https://github.com/keras-team/keras-applications/blob/master/keras_applications/resnet_common.py)
