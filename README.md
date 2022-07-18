This is the code repository for paper, Class-Specific Attention (CSA) for Time-Series Classification, that was submitted to ICDE 2023.

This paper uses 40 datasets, 28 multivariate time series and 12 univariate time series datasets. This repository has one dataset as a running example. Other datasets can be found: http://www.timeseriesclassification.com/dataset.php 

1. Prerequirments:
    The project is writen using Python 3.7.
    The following packages are required to run this project:
    1.1 pytorch-1.10.0
    1.2 scikit-learn 0.22.0
    1.3 numpy 1.17.3

2. Generate results in Table 2 and Figure 3
    2.1 Script:
        # python main.py <DATA_NAME> <MODEL_NAME> <ATTENTION_TYPE> <DATA_PRE_KEY>

    2.2 Parameters:
        <DATA_NAME>: dataset names. Case sensitive. 
        This parameter is also used to identify the parameter file.
        For example, for "ArticularyWordRecognition" dataset, the parameter file is 
        mts_parameters/all_feature_classification_ArticularyWordRecognition.txt. 

        <MODEL_NAME>: model names.
        'fcn' means FCN (Fully-Convolutional Network without any attention)
        'cnn' means CNN-ATN (Cross-Attention Stablized Convolutional Neural Network [4])
        'mlstm' means MLSTM model (multivariate Long Short Term Memory)
        'fcn-mlstm' means FCN-MLSTM model [5][6]
        'tapnet' means TapNet model [9]

        <ATTENTION_TYPE>: the method we can test. 
        4 possible values: -1, 0, 1, 2.
        1 means the proposed CSA module.
        2 means the Cross Attention in a reference paper [4].
        3 means the Cross Attention with the proposed CSA moduel
        4 means no attention
        5 means the proposed CSA moduel without the CD component

        <DATA_PRE_KEY>: either "mts" or "utc"
        "mts" means for multivariate time seires datasets
        "uts" means for unvariate time seires datasets

        For example, the command "python main.py ArticularyWordRecognition fcn 1 mts"

    2.3 Outputs:
        The log file of the training stage locates at log_pytorch_epoch400/<DATASET_NAME>/fcn_group_attn_batchfalse/

        The testing accuracy, training time, and testing time can be found at the last three rows of the output log file.