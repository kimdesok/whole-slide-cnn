# Whole slide image (WSI) training CNN Pipeline

### Introduction
The whole slide image(WSI) is the digital image representing the entire histologic slide captured by a high-throughput digital scanner such as Leica's Aperio scanner.  The image size of a WSI is often larger than 100,000 by 100,000 pixels reaching upto 10 GB. Each individual WSI can be easily labeled according to the corresponding clinical information readily available at the data deposit site such as the NCI's GDC data portal, for example.

| ![image](https://user-images.githubusercontent.com/64822593/218235019-81fd8cbe-00ce-482c-8d32-4d926ae80e2e.png) |
|:--:| 
| *Whole Slide Image(WSI) is basically a tiled TIFF.  The size is 19,442 by 39,089 with three color channels, equivalent to 2.3GB.* |


It is widely known that training a deep neural network with a large WSI dataset is more practical since it does not require a time consuming pixel level annotation. However, to achieve a high classification accuracy, the magnification of WSI images should be at least 4x[1].  This requirement, in turn, limits the use of a computing resource with the memory smaller than 512 GB, for example.

The computing resource even with multiple GPUs often lacks this much memory and ends up with the out of memory(OOM) error while training.  To solve this problem, IBM developed Large Model Support(LMS) that takes a user-defined computational graph and swaps tensors from GPUs to the host memory and vice versa (For more information, visit https://www.ibm.com/docs/en/wmlce/1.6.0?topic=gsmf-getting-started-tensorflow-large-model-support-tflms-v2).  

The authors who set up this repository originally had also developed a huge model support(HMS) package, probably similar to the IBM's LMS in a nutshell [1][2].  The scripts provided by the authors let us to reproduce their results by WSI dataset preparation, DL training, inference, visualization, and statistics calculation, etc [1] (See the figure below).

![image](https://user-images.githubusercontent.com/64822593/208655851-6986f3e5-84c0-46ab-9194-b35887bf83a0.png)

Once this pipeline is explored for lung cancer datasets, it is believed to be utilized to other important human cancers such as prostate cancers or breast cancers by simply creating new configuration files (and label files in addition which is a piece of cake).

Thus, here lies the purpose of reproducing the original works of the authors: We can learn about software & hardware requirements for the application of the pipeline to other cancers.  In addition, we can identify numerous tasks that would enable us to apply this sort of the DL pipeline to the real world problems.  

List of tasks:
1. Migration to TF2.
2. Reducing the hardware requirement (or find out how to increase the computing speed). 

## Requirements

### Hardware Requirements

The computing resource should provide an adequate amount of main memory space (minimal: 256 GB, recommended: 512 GB) to prevent out-of-memory(OOM) error and fully explore the utility of the huge memory support(HMS) function.  

However, a trial training could be performed by reducing the size of the WSI training image, the size of batch, and the complexity of the model basically.  However, it would result in the lower accuracy values ranging from 80 to 90%.  For the specific setup, please refer to the method section below.

### Packages

The codes were tested in the environment with Ubuntu 18.04, Python 3.7.3, cuda 10.0, cudnn 7.6 and Open MPI 4.0.1.

More specifically, a set of CUDA related libraries compatible with Tensorflow version 1.15.5 should be installed as below
```
conda install cudatoolkit=10.0
conda install cudnn=7.6.5
```

Python packages were installed via a dependency management and packaging tool, poetry.  
```
poetry install
```
Some of packages include: 
- Tensorflow v1.x (tensorflow-gpu==1.15.5)
- Horovod (horovod==0.19.0)
- MPI for Python (mpi4py==3.0.3)
- OpenSlide 3.4.1 (https://github.com/openslide/openslide/releases/tag/v3.4.1)
- OpenSlide Python (openslide-python=1.1.1)
- (optional) R 4.0.2 (https://www.r-project.org/)
- Tensorflow Huge Model Support (package provided in the author's repository called 'tensorflow-huge-memory-support') <br>

For a full list, see poetry.lock in the same folder.

## Methods
### 1. Datasets and Configurations
The dataset consists of WSIs reresenting LUAD(Lung adenocarcinoma) and LUSC(Lung squamous cell carcinoma).  
These WSIs were downloaded from GDC's TCGA-PRAD project (See detailed method on how to download WSIs [here](https://andrewjanowczyk.com/download-tcga-digital-pathology-images-ffpe/).  
Train, validation, and test sets were divided by the ratio of 7:1.5:1.5, respectively.

The .csv files under data_configs folder were used without any modification. 
A detailed description was available by the authors as in the Appendix below.  

Hyperparameters were set in a YAML file under train_configs folder.  
In the YAML file, the parameters of RESIZE_RATIO, INPUT_SIZE, and  NUM_UPDATES_PER_EPOCH were set appropriately to avoid the OOM error at each magnification.  

Some examples are shown below.
```
#Training @1x
    RESIZE_RATIO: 0.05
    INPUT_SIZE: [5500, 5500, 3]
    NUM_UPDATES_PER_EPOCH: 80

#Training @2x
    RESIZE_RATIO: 0.1
    INPUT_SIZE: [10000, 10000, 3]
```

In order to utilize the pretrained model provided by the authors, the LOAD_MODEL_BEFORE_TRAIN and MODEL argumens were reset as below:
    
```
#Training @2x with the model pretrained @ 4X
    LOAD_MODEL_BEFORE_TRAIN: True
    MODEL: "frozenbn_resnet50"    
```
### 2. Train a Model

To train a model, the following script was run:
```
python -m whole_slide_cnn.train --config config_wholeslide_1x.yaml [--continue_mode]
```
where `--continue_mode` is optional that makes the training process begin after loading the model weights.

To enable multi-node, multi-GPU distributed training, add `mpirun` in front of the above command, e.g.
```
[CUDA_VISIBLE_DEVICES="0,1,2,3";]mpirun -np 4 python -m whole_slide_cnn.train --config config_wholeslide_2x.yaml
```

Note) 
1) Explicit export command was not required to activate all the GPUs that the node has.  You may selectively activate the GPUs for the job but somehow it resulted in unknown errors (no error notice but the program just stopped).  
2) At the first run, 2x input images have been successfully trained but images with higher resolutions such as 4x were resulted in the OOM errors.  Note that the current node consists of four Tesla T4 GPUs with 64 GB memory and extra 350 GB in the server utilized by adopting the UM technique that the authors developed.  
3) At the moment, resnet-34 was chosen as the model and the magnification was being adjusted at 3x to avoid the OOM error(resize ratio of 0.15).  The input image size was set at (10000,10000,3). This setting occasionally produce a deadlock among GPUs.  No solutions are available at the moment.  

```
[2023-02-20 15:42:31.168674: W horovod/common/stall_inspector.cc:105] One or more tensors were submitted to be reduced, gathered or broadcasted 
by subset of ranks and are waiting for remainder of ranks for more than 60 seconds. 
This may indicate that different ranks are trying to submit different tensors or that only subset of ranks is submitting tensors, 
which will cause deadlock. 
```

### 3. Evaluate the Model

The model was evaluated by calling the command as below and optionally a prediction heatmap was also generated.
```
python -m whole_slide_cnn.test --config config_wholeslide_1x.yaml
```
This command generated a JSON file named `test_result.json` by default in the result directory.
The file contained the model predictions for each testing slide. To further generate the AUC values and their graphs, more tools were available, as explained in the Appendix.

Note) These tools are currently profiled for lung cancer classification and should be modified when applying to your own tasks.

## Results & Discussion

### 1. Performance of Resnet 34 with the WSI at 1x magnification
We first tried the training with the Resnet34, initialized by the weights obtained by training with Imagenet.  The image size was set to be at 5500 x 5500 with the resize factor of 0.05, representing the magnification at 1x.  The loss and accuracy curves were plotted upon training the model through 100 epochs.  The validation accuracy reached about 0.68 with the validation loss of 0.64.  The test result is not shown (no data).
![image](https://user-images.githubusercontent.com/64822593/201547097-89a4b7f7-9218-4250-964d-1f564bb60266.png)

### 2. Performance of Resnet 50 with the WSI at 1x and 2x magnification
We then tried the training with the Resnet50, that would show the effect of increasing the complexity of the model. The ResNet50 model was initialized by the weights of the pretrained Imagenet model. The validation accuracy reached about 0.80 with the validation loss of 0.41.  The AUC was 0.8063 when the test dataset was evaluated with the model. 

![image](https://user-images.githubusercontent.com/64822593/198936803-2a2fb8d3-d3b2-4009-b9d9-e54b24d96e79.png)

![image](https://user-images.githubusercontent.com/64822593/209490657-9591d29d-89b4-4f8d-bf0e-5f21a1259795.png)
<br>Receiver Operating Characteristic(ROC) Curves for LUAD and LUSC at 1x by Resnet 50 model<br>

To further improve the accuracy, we tried the training with 2x images.  The AUC slightly increased to 0.8261.

![image](https://user-images.githubusercontent.com/64822593/211115004-cc7fdb16-2029-4c0a-8239-d32afd1c199a.png)
<br>Receiver Operating Characteristic(ROC) Curves for LUAD and LUSC at 2x by Resnet 50 model<br>

### 3. Performance of Resnet 50 with the WSI at 2x magnification with the model pretrained at 4x

Although the model provided by the authors was trained at 4x, it could be utilized for the training at 2x.  The validation accuracy reached about 0.89 with the validation loss of 0.25 upon completing 100 epochs.  The AUC also dramatically increased to 0.9731, 17.9% boost compared the previous training at 2x without the 4x pretrained model.  Our hardware spec. remained the same as for the 1x training (CPU RAM : 128 GB).

![image](https://user-images.githubusercontent.com/64822593/201279646-b3c4170d-2cc1-4f87-b32d-6486e306f473.png)

![image](https://user-images.githubusercontent.com/64822593/211124003-166dd949-94b4-4f32-9c02-3bc8452383e7.png)
<br>Receiver Operating Characteristic(ROC) Curves for LUAD and LUSC at 2x by Resnet 50 model initialized by the model pretrained at 4x<br>

### 4. Visualization of grad-CAM
The model was evaluated visually by grad-CAM that depicts the likelihood of the tumor in the tissue(panel A in the figure). The image below highlights where the lung cancer cells are likely located in an LUAD case.  

The second image was generated using the ResNet50 model trained at the 1x magnification from the weights of Imagenet and shows somewhat large tissue area of false positive(panel B).  

The third image was generated using the Resnet model trained at the same magnification in a continous mode after loading the previously trained model (at 4x) and seemed to show much tighter marking of the tumor tissue(panel C).  

Finally, the fourth image was generated using the Resnet model trained at the 2x in the continous mode and seems to show much larger marking area of the tumor tissue(panel D).  At the moment, the marking has not been validated by an expert.
![image](https://user-images.githubusercontent.com/64822593/201546476-51062b10-2bd1-4e96-a929-65078ae32f0b.png)

### 5. Problems encountered when the hardware requirement was not met
The computing server consisted of an NVIDIA Quadro RTX 6000 GPU that had 24 GB memory capable of runnning at 14~16 TFlops in single precision.  The CPU memory was 128 GB.  

Due to the memory requirement of the training algorithm, only images at the magnification of 2x or lower were suitable for the training at the server.  Thus, the images with the magnification higher than 2x could not be used for the training. 

When the magnification of the training images was 1x, the accuracy was 0.8063 for the Resnet 50 model.  When the magnification was increased to 2x, the accuracy was also increased to 0.8261(See the figure above).  This suggested that the training images of higher magnification should improve the accuracy further. 

For trying it out with the 4x images, for example, the memory size is expected to be larger than 512 GB to accomodate the deep layers of the Resnet 50, the batch of 1, and the image of 22,000 x 22,000 pixels or larger, while avoiding the OOM error according to the authors.  Without manipulating the CPU memory and floating point operations, the memory requirement reaches up to 892 GB due the huge size of feature variables in the Resnet 50 model while the parameters in the model takes about 0.1 GB (obtained from Calc_GPU_memory_requirement.ipynb, Author: Dr. Sreenivas Bhattiprolu).

When this algorithm was executed in a high computing resource with many GPUs but with small memory, for example, four GPUs with each 16 GB memory, the Resnet 50 model causes a GPU stall.  The input image size was set at (10000, 10000, 3) for the 2x magnification.  For this setting, the memory requirement was 276 GB.  Since the server memory size was 390 GB and the total memory reached 454 GB, the Resnet 50 model was thought to be working without a problem.   

### 6. Computing time
The computing time at 1x was about 4.3 sec per batch and the size of batch was set to 80.  Total computing time was 115 mins when the number of the total batch for the training of ResNet50 was 1,600.  The computing time at 2x was about 46 sec per batch.  Total computing time was about 102 hours when the number of the batch to process was 8,000.

## Acknowledgement
The computing server was kindly provided by the National Internet Promotion Agency(NIPA) of s. Korea.

## References

[1] Chi-Long Chen, Chi-Chung Chen, Wei-Hsiang Yu, Szu-Hua Chen, Yu-Chan Chang, Tai-I Hsu, Michael Hsiao, Chao-Yuan Yeh$ & Cheng-Yu Chen$  An annotation-free whole-slide training approach to pathological classification of lung cancer types using deep learning. *Nat Commun* **12,** 1193 (2021). https://doi.org/10.1038/s41467-021-21467-y

[2] Wen-Yu Chuang, Chi-Chung Chen, Wei-Hsiang Yu, Chi-Ju Yeh, Shang-Hung Chang, Shir-Hwa Ueng, Tong-Hong Wang, Chuen Hsueh, Chang-Fu Kuo & Chao-Yuan Yeh$ Identification of nodal micrometastasis in colorectal cancer using deep learning on annotation-free whole-slide images. *Mod Pathol* (2021). https://doi.org/10.1038/s41379-021-00838-2

## Appendix
Mostly taken from the original README.MD and some additional stuffs worth of mentioning were added.

### License

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

The model was trained by TCGA-LUAD and TCGA-LUSC diagnostic slides specified in `data_configs/pure_tcga/train_pure_tcga.csv` using the configuration file, `train_configs/pure_tcga/config_pure_tcga_wholeslide_4x.yaml`.

Since no normal lung slides were provided in these data sets, the model predicts a slide as either adenocarcinoma (class_id=1) or squamous cell carcinoma (class_id=2).

Thus, the prediction scores for normal (class_id=0) should be ignored.

Validation results (*n* = 192) on `data_configs/pure_tcga/val_pure_tcga.csv` are listed as follow.

- AUC (LUAD vs LUSC) = **0.9794** (95% CI: 0.9635-0.9953)
- Accuracy (LUAD vs LUSC) = **0.9323** (95% CI: 0.8876-0.9600, @threshold = 0.7 for class1, 0.3 for class2)

<img src="https://user-images.githubusercontent.com/6285919/122541978-cd029800-d05c-11eb-932c-3cc0c517101e.png" width="400" />

#### (Note) Error in loading the pretrained model

The number of the layers of the provided model was different to the ones of Resnet 50 or Resnet 32. <br>
It was compatible only with the 'frozenbn_resnet50', whereas the 'fixup_resnet50' also resulted in an error while initializing the model.
In fact, the 'config_pure_tcga_wholeslide_4x.yaml' had the MODEL argument defined as 'frozenbn_resnet50'.

### 5. Data Availability

The slide data supporting the cross-site generalization capability in this study are obtained from TCGA via the Genomic Data Commons Data Portal (https://gdc.cancer.gov).

A dataset consists of several hundred slides from TCGA-LUAD and TCGA-LUSC is suitable for testing our pipeline in small scale, with some proper modifications of configuration files described above.

### 6. Backbones of the model 

In model.py, a dictionary variable called graph_mapping was defined as below.  The model types, 'fixup_resnet50' and 'frozenbn_resnet50', were initialized with the imagenet as weights.  The other two, 'resnet34' and 'fixup_resnet34', were initialized randomly.  

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
