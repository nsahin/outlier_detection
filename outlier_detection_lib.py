import time
import pathlib
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats, spatial
from sklearn import svm, metrics, mixture
from sklearn.decomposition import PCA


def create_output_filenames(output_folder, input_file, method, kernel):
    """ Read data with features and create dataframe with information.

        Args:
            output_folder (str): The main output folder
            input_file (str): Input data with features
            method (str): OD method
            kernel (str): Kernel for one-class SVM method

        Returns:
            output (dict): Output filenames
        """

    # Create screen name
    screen_name = '%s_%s' % (input_file.split('/')[-1].replace('_input.csv', ''), method)
    if method == 'SVM':
        screen_name += '-%s' % kernel

    # Create output folder
    output_path = pathlib.Path(output_folder) / screen_name
    output_path.mkdir(parents=True, exist_ok=True)

    # Create output filenames
    output = {'od_results': '%s/%s_OD_results.csv' % (str(output_path), screen_name),
              'score_cells': '%s/%s_OD_results_score.csv' % (str(output_path), screen_name),
              'pca_cells': '%s/%s_PCA_cells' % (str(output_path), screen_name),
              'neg_percentile': '%s/%s_neg_percentile' % (str(output_path), screen_name),
              'log': '%s/%s_log.txt' % (str(output_path), screen_name)
              }

    return output
    
      
def log_write(log_f, text):
    """ Open and write to the log file

        Args:
            log_f (str): Log file
            text (str): Something to write to log file
        """
    
    f = open(log_f, 'a')
    f.write(text)
    f.close()
    

def read_input_data(input_f, controls_f, output, var, preprocess=False):
    """ Read data with features and create dataframe with information.

        Args:
            input_f (str): Input data with features
            controls_f (str): File that contains group_id for negative and/or positive controls
            output (dict): Output filenames
            var (float): Percentage of explained variance
            preprocess (bool): Scale and apply PCA if needed

        Returns:
            df (pd.DataFrame): Single cell data and information
            pos (np.array): Positive control group_ids
            neg (np.array): Negative control group_ids
        """

    # Read input data
    log = open(output['log'], 'w')
    log.write('Reading input files\n')
    log.close()

    # Read input data and drop cells that have missing features
    df = pd.read_csv(input_f)
    df = df.dropna(axis=0, how='any').reset_index(drop=True)

    # Read controls file
    df_controls = pd.read_csv(controls_f)
    neg = df_controls[df_controls['control'] == 'negative']['group_id'].values
    pos = df_controls[df_controls['control'] == 'positive']['group_id'].values

    # Preprocess data
    if preprocess:
        # Scale features
        data_all = df.iloc[:, 2:].values
        data_neg = df[df['group_id'].isin(neg)].iloc[:, 2:].values
        wt_mean = np.nanmean(data_neg, axis=0)
        wt_variance = np.nanstd(data_neg, axis=0)
        data_all = (data_all - wt_mean) / wt_variance

        # Apply PCA
        pca = PCA(n_components=data_all.shape[1])
        pca.fit(data_all)
        exp_var = []
        num_pc = 0
        total_var = 0
        if sum(pca.explained_variance_ratio_[:2]) == 1:
            num_pc = 2
        else:
            for i in range(len(pca.explained_variance_ratio_)):
                total_var += pca.explained_variance_ratio_[i]
                exp_var.append(total_var)
                if total_var > var:
                    num_pc = i
                    break
        pca = PCA(n_components=num_pc)
        data_all = pca.fit_transform(data_all)

        # Create new dataframe
        df = df.iloc[:, :2]
        data = pd.DataFrame(columns=['PC%d' % pc for pc in range(num_pc)], data=data_all)
        # MERGE THE TWO UP
        df = pd.concat([df, data], axis=1)

    return df, pos, neg


def oneclass_svm_method(df, neg, output, kernel):
    """ Outlier Detection with One-Class SVM Method.

        Args:
            df (pd.DataFrame): Single cell data and information
            neg (np.array): Negative control group ids
            output (dict): Output filenames
            kernel (str): Kernel for one-class SVM

        Returns:
            df (pd.DataFrame): Single cell data and information with scores added
        """

    # Detect outliers
    log_write(output['log'], 'Outlier detection...\n')
    start_time = time.time()

    # Create a subset with only neg cells and fit the model
    ocsvm = svm.OneClassSVM(kernel=kernel)
    data_neg = df[df['group_id'].isin(neg)].iloc[:, 2:].values
    size = 10000
    if data_neg.shape[0] < size:
        size = data_neg.shape[0]
    data_neg = data_neg[np.random.choice(data_neg.shape[0], size=size, replace=False), :]
    ocsvm.fit(data_neg)
    df['score'] = - ocsvm.decision_function(df.iloc[:, 2:].values).ravel()

    # Print OD wall time
    text = 'Outlier detection method: One-Class SVM\n'
    text += 'Outlier detection wall time: %.2f minutes\n' % ((time.time() - start_time) / 60.0)
    text += 'Number of samples: %d\n' % df.shape[0]
    text += 'Number of negative samples: %d\n' % df[df['group_id'].isin(neg)].shape[0]
    log_write(output['log'], text)

    return df


def mahalanobis_distance_method(df, neg, output):
    """ Outlier Detection with Mahalanobis Distance Method.

        Args:
            df (pd.DataFrame): Single cell data and information
            neg (np.array): Negative control group ids
            output (dict): Output filenames

        Returns:
            df (pd.DataFrame): Single cell data and information with scores added
        """

    # Detect outliers
    log_write(output['log'], 'Outlier detection...\n')
    start_time = time.time()

    # Create a subset with only neg cells and calculate mean and covariance matrix
    data_neg = df[df['group_id'].isin(neg)].iloc[:, 2:].values
    mean = np.mean(data_neg, axis=0)
    inverse_cov = np.linalg.inv(np.cov(np.transpose(data_neg)))

    # Calculate distances and threshold
    def mahalanobis_distance(x):
        return spatial.distance.mahalanobis(x, mean, inverse_cov)

    df['score'] = np.apply_along_axis(mahalanobis_distance, axis=1, arr=df.iloc[:, 2:].values)

    # Print OD wall time
    text = 'Outlier detection method: Mahalanobis Distance\n'
    text += 'Outlier detection wall time: %.2f minutes\n' % ((time.time() - start_time) / 60.0)
    text += 'Number of samples: %d\n' % df.shape[0]
    text += 'Number of negative samples: %d\n' % df[df['group_id'].isin(neg)].shape[0]
    log_write(output['log'], text)

    return df


def gmm_method(df, neg, output):
    """ Outlier Detection with Mahalanobis Distance Method.

        Args:
            df (pd.DataFrame): Single cell data and information
            neg (np.array): Negative control group ids
            output (dict): Output filenames

        Returns:
            df (pd.DataFrame): Single cell data and information with scores added
        """

    # Detect outliers
    log_write(output['log'], 'Outlier detection...\n')
    start_time = time.time()

    # Create a subset with only neg cells and fit the model
    data_neg = df[df['group_id'].isin(neg)].iloc[:, 2:].values
    gmm = mixture.GaussianMixture(n_components=3, covariance_type='full')
    gmm.fit(data_neg)
    df['score'] = - gmm.score_samples(df.iloc[:, 2:].values).ravel()

    # Print OD wall time
    text = 'Outlier detection method: GMM 3-components with full covariance\n'
    text += 'Outlier detection wall time: %.2f minutes\n' % ((time.time() - start_time) / 60.0)
    text += 'Number of samples: %d\n' % df.shape[0]
    text += 'Number of negative samples: %d\n' % df[df['group_id'].isin(neg)].shape[0]
    log_write(output['log'], text)

    return df


def identify_outliers(df, neg, output, od_thresholds):
    """ Outlier Detection with different OD thresholds.

        Args:
            df (pd.DataFrame): Single cell data and information
            neg (np.array): Negative control group ids
            output (dict): Output filenames
            od_thresholds (list): Threshold on the right tail to decide on outlier boundary

        Returns:
            main_dict (dict): With in-outlier information
        """

    # Subset relevant data
    neg_scores = df[df['group_id'].isin(neg)]['score'].values
    data_all = df.iloc[:, 2:-1].values

    is_inlier_columns = []
    for ODT in od_thresholds:
        # Is_inlier column
        is_inlier_col = 'is_inlier_%d' % ODT
        is_inlier_columns.append(is_inlier_col)

        # Threshold and plot data
        threshold = stats.scoreatpercentile(neg_scores, 100 - ODT)
        df[is_inlier_col] = df['score'] <= threshold

    # Remove data from the dataframe
    df = df[['sample_id', 'group_id', 'score'] + is_inlier_columns]

    return df


def max_diff_stat(neg, mut):
    """ Calculate the maximum difference in percentage between mutant and neg populations
        on cumulative distribution of score

        Args:
            neg (np.array): Neg scores
            mut (np.array): Mutant scores
        Returns:
            maxx (float): Maximum difference
        """

    # Find the maximum difference between two CDFs
    maxx = 0
    for i in range(len(neg)):
        diff = neg[i]-mut[i]
        if diff > maxx:
            maxx = diff

    return maxx * 100


def prepare_output_results(df, neg, output, od_thresholds, mapping_file):
    """ Prepare the output file with group_id information
        Calculate penetrance and p-value

        Args:
            df (pd.DataFrame): Single cell data and information
            neg (np.array): Neg group_id names
            output (dict): Output filenames
            od_thresholds (list): Threshold on the right tail to decide on outlier boundary
            mapping_file (str): Combine results file with mapping sheet

        Returns:
            df_output:  Combined outlier detection results
        """

    # Negative sample scores at each percentile for maximum difference calculation
    neg_s = np.sort(df[df['group_id'].isin(neg)].score.values)
    neg_scores = [stats.scoreatpercentile(neg_s, p) for p in range(1, 101)]
    y_neg_scores = np.arange(len(neg_scores)) / float(len(neg_scores) - 1)

    # Initialize output folder
    columns = ['group_id', 'num_cells', 'penetrance', 'neg_percentile_at_threshold']
    for ODT in od_thresholds:
        columns.append('penetrance_t%d' % ODT)
    df_output = pd.DataFrame(columns=columns)

    # Initialize
    this_row = 0
    sample_ids_all = np.array([])
    is_inlier_all = np.array([])

    # Regroup this dataframes by group_id
    for group_name in df['group_id'].unique():
        # Gather group name and group cell count
        df_group = df[df['group_id'] == group_name]
        line = [group_name]
        num_cells = df_group.shape[0]
        line.append(num_cells)

        # Calculate penetrance with maximum difference
        mut_scores = df_group['score'].values
        y_mut_scores = [stats.percentileofscore(mut_scores, s) / 100 for s in neg_scores]
        max_diff_pene = max_diff_stat(y_neg_scores, y_mut_scores)
        line.append(max_diff_pene)

        # Calculate neg percentile at the score of maximum difference
        threshold = stats.scoreatpercentile(mut_scores, 100 - max_diff_pene)
        neg_percentile = stats.percentileofscore(neg_s, threshold)
        line.append(neg_percentile)

        # Single cell identification from maximum difference
        is_inlier = df_group['score'].values < threshold
        sample_ids_all = np.append(sample_ids_all, df_group['sample_id'])
        is_inlier_all = np.append(is_inlier_all, is_inlier)

        # Calculate penetrances with different thresholds
        for ODT in od_thresholds:
            is_inlier_col = 'is_inlier_%d' % ODT
            pene = float(sum(np.asarray(df_group[is_inlier_col]) == 0)) / num_cells * 100
            line.append(pene)

        # Append results for this well
        df_output.loc[this_row, ] = line
        this_row += 1

    # Save into a dataframe
    df_output = df_output.sort_values('penetrance', ascending=False)
    if mapping_file:
        df_output = pd.read_csv(mapping_file).merge(df_output, on=['group_id'])
    df_output = df_output.reset_index(drop=True)
    df_output.to_csv(path_or_buf=output['od_results'], index=False)

    # Merge genotype information and scores file
    df_scores = pd.DataFrame(columns=['sample_id', 'is_inlier_max_diff'])
    df_scores['sample_id'] = sample_ids_all
    df_scores['is_inlier_max_diff'] = np.array(is_inlier_all, dtype=bool)
    df = df_scores.merge(df, how='inner', on=['sample_id'])
    df.to_csv(output['score_cells'], index=False)

    # Plot negative samples percentile distribution
    plt.figure(figsize=(6, 6))
    sns.set(font_scale=1.25)
    sns.set_style('white')
    sns.kdeplot(df_output.neg_percentile_at_threshold.values, color='mediumblue', shade=True)
    plt.xlabel('neg percentile at the score of maximum difference')
    mean_percentile = np.mean(df_output.neg_percentile_at_threshold.values)
    plt.title('Mean percentile: %.4f' % mean_percentile)
    fig = plt.gcf()
    fig.savefig('%s.png' % output['neg_percentile'], dpi=150, bbox_inches='tight')
    fig.clf()
    plt.close(fig)
    mean_neg_percentile = np.mean(df_output['neg_percentile_at_threshold'])
    log_write(output['log'], 'Mean neg percentile at thresholds: %.2f\n' % mean_neg_percentile)

    return df_output


def calculate_auc(df, neg, pos, value_col):
    """ Plot ROC and PR curves, penetrance agreement and confusion matrices if the positive control file is available

        Args:
            df (pd.DataFrame): OD results dataframe grouped by group_id
            neg (np.array): Negative control group_ids
            pos (np.array): Positive control group_ids
            value_col (str): Value column to calculate performance
        """

    df = df[df.num_cells >= 15].reset_index(drop=True)
    nc = df[df.group_id.isin(neg)][value_col].values
    pc = df[df.group_id.isin(pos)][value_col].values

    y_score = np.append(nc, pc)
    y_true = np.append(np.repeat(0, len(nc)), np.repeat(1, len(pc)))
    sample_weights = np.append(np.repeat(float(len(pc)) / len(nc), len(nc)), np.repeat(1, len(pc)))
    aupr = metrics.average_precision_score(y_true, y_score, sample_weight=sample_weights)
    auroc = metrics.roc_auc_score(y_true, y_score)

    return aupr, auroc


def calculate_performance(df, od_thresholds, neg, pos, output):
    """ Calculate outlier detection performance on all metrics

        Args:
            df (pd.DataFrame): OD results dataframe grouped by group_id
            od_thresholds (list): Outlier detection threshold
            neg (np.array): Negative control group_ids
            pos (np.array): Positive control group_ids
            output (dict): Output filenames
        """

    log_write(output['log'], '\nOutlier detection threshold performances:\nOD_Threshold,AUPR,AUROC\n')

    # Calculate performance with maximum difference
    aupr, auroc = calculate_auc(df, neg, pos, 'penetrance')
    log_write(output['log'], 'Max difference,%.4f,%.4f\n' % (aupr, auroc))

    # Calculate performance with difference OD thresholds
    for ODT in od_thresholds:
        value_col = 'penetrance_t%d' % ODT
        aupr, auroc = calculate_auc(df, neg, pos, value_col)
        log_write(output['log'], '%d,%.4f,%.4f\n' % (ODT, aupr, auroc))
