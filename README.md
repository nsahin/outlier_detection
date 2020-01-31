# Outlier detection for single cell images
This repository contains the code and example datasets for the paper:  

"A comparison of automated outlier detection approaches for single cell images"

Nil Sahin, Mojca Mattiazzi Usaj, Matej Usaj, Audrina Zhou, Erin Styles, Charles Boone, Brenda J. Andrews and Quaid Morris

This script identify single cells with abnormal phenotypes from genetic or chemical perturbation screens.
The wild-type (negative) morphology is modelled using one of the four methods and any feature set.

Perturbed cells are scored under this model and cells with high scores are identified as outliers.
The score threshold is chosen with two methods:
1. The maximum difference between the cumulative score distributions of wild-type and perturbed alleles are calculated independently for each perturbed allele, an idea similar to Kolmogorov-Smirnov statistic.
2. Hard thresholding method where all cells from every perturbed allele share the same score threshold. A good option is to find the score around 90th percentile of wild-type (neg) score.

We provide the source code that implemented the dynamic thresholding and hard thresholding with 70th, 75th, 80th, 85th, 90th, 95th and 99th percentile of wild-type (neg) scores.
Penetrance as percent outlier cells are calculated for every perturbed population.
Performance in terms of AUPR and AUROC are calculated if positive controls are provided.

We also provide our script to extract deep learning (DL) features from pre-trained networks.



## Packages

Python 3.6+: http://www.python.org/getit/
   
Tensorflow 1.15+: https://www.tensorflow.org/install
   
Keras 2.2+: https://keras.io/#installation

You can use the following command to install all the required packages:

    conda create --name <env> --file environment.yml


## Full Datasets

The datasets are too large to store in the repository.

The datasets are available at:
<http://data_link.com>


## Running outlier detection

Please use Argument Parser, for example:

    python outlier_detection.py
    --input_file input/screen1_features.csv
    --controls_file input/screen1_controls.csv
    --method GMM


### Input Options

Example output files at _input/_

**--input_file** (-i): A .csv file of single cell features.
Each row should be a single cell, with the first column being a unique cell identifier (sample_id) and the second column being the group identifier (group_id).
The following columns should be single cell features that are extracted from any method but they should at least be two.
Considering this to be used in perturbation analyses, each cell should come from a population of cells, therefore the group_id is used to calculate the penetrance of the perturbation.
Example file at _input/screen1_features.csv_

**--controls_file** (-c): This sheet contains the group_ids of negative and positive control strains.
Negative controls are the normal (WT, unperturbed) strains and at least one negative control group is required for the outlier detection analysis to work.
Positive controls are the mutant phenotype classes known prior and is only used to report the performance of outlier detection.
Example file at _input/screen1_controls.csv_  

**--method** (-m): The method is used to model the normal morphology.
The options are Gaussian Mixture Models (GMM), Mahalanobis distance (MH) and OneClassSVM (SVM).
The default method is GMM.

**--kernel** (-k): One-class SVM kernel is required only when SVM method is chosen. 

**--mapping_file** (-f): This sheet contains group information for each population in the screen.
This is not a required file, it is only used to combine the penetrance information with group information.
It should have a group_id column that is matched with the input file.
Example file at _input/screen1_mapping.csv_

**--output_file** (-o): A path to save the result files.
Example output files at _output/_

**--od_thresholds** : A list of thresholds for the hard-thresholding method for the scoring on outlier detection.
Default thresholds are [1, 5, 10, 15, 20, 25, 30] that translates to 99th, 95th, 90th, 85th, 80th, 75th and 70th percentile of wild-type scores.

**--preprocess_data** (-p): The features are standardized and PCA is applied.
Default is no preprocessing applied.

**--variance** (-v): The percentage of variance explained by PCA.
It is only used it --preprocess_data is used.
Default is 0.8.


### Output Files

Example output files at _output/_

**OD_results.csv**: Penetrance calculation in terms of percentage of outlier cells for every perturbed population is provided in this file for all thresholding approaches separately.
The "penetrance" column is calculated using the maximum difference method and the other penetrance columns are the separate hard thresholds.
For penetrance using the maximum difference method, the wild-type (neg) percentile at each of the threshold scores to identify cells as outliers is also provided separately for each perturbed population.
 
**OD_results_scores.csv**: For every cell, the sample_id and the score under the wild-type model is provided in this file.
The outlier decision is made using the maximum difference method and hard thresholding methods for each threshold separately.
If the cell is called an outlier with a method, it will be written as False with the respective "is_inlier" column.

**log.txt**: The log file is generated to combine information about the outlier detection analysis.
The method and wall time of outlier detection, sample count of the whole screen and negative (wild-type) samples are provided.
The mean of the wild-type (neg) percentile at each of the threshold scores is also provided.
If the controls_file contains positive controls, the performance is calculated separately for different thresholding methods in terms of AUPR and AUROC.


## Running Deep Learning feature extraction

Please use Argument Parser, for example:

    python extract_DL_features.py
    --input_file input/screen1_image_paths.csv
    --model input/cnn_weights.h5


### Input Options

Example output files at _input/_

**--input_file** (-i): A .csv file of single cell coordinates from full (population) images.
Each row should be a single cell, with the first column being a unique cell identifier (sample_id)
The group identifier (group_id) can also be provided for mapping but not required.
The location of full images and the xy coordinates of single cells should be provided with the column names as 'image_path', 'center_x' and 'center_y'.
Example file at _input/screen1_image_paths.csv_

**--mapping_file** (-f): This sheet contains group information for each population in the screen.
This is not a required file, it is only used to combine the feature information with group information.
It should have a group_id column that is matched with the input file.
Example file at _input/screen1_mapping.csv_

**--model** (-m): The pretrained network should be provided as an .h5 file to extract the features from.
This script is written to extract features from the last 2 fully connected layers of a CNN with both layers having 512 units each.
The script should be edited in "extract_features" and "initialize_data" functions to fit the inputted network if the architecture is different than the original architecture used as in Kraus et al 2017 (doi: 10.15252/msb.20177551).

**--crop_size** (-c): The crop size to put a bounding box around single cells.
Default is 64 to fit the original CNN architecture.

**--num_channels** (-n): Number of channels for the model and images, implemented only for 1 or 2 channels.


### Output Files

Example output files at _output/_

**DL_features_samples.csv**: The feature file is provided with sample_ids and features extracted as activations of the last two fully connected layers.
 
**DL_features_groups.csv**: The group information is appended to the feature file.