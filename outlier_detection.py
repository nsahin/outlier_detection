from outlier_detection_lib import *

if __name__ == '__main__':
    # Arguments
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_file', default='',
                        help='Input data, every row should have unique sample_id')
    parser.add_argument('-c', '--controls_file', default='',
                        help='Controls file that matches group_id to the input data')
    parser.add_argument('-m', '--method', default='GMM',
                        help='OD method: Mahalanobis (MH) - OneClassSVM (SVM) - GMM')
    parser.add_argument('-k', '--kernel', default='rbf',
                        help='one-class SVM kernel: rbf, linear')
    parser.add_argument('-f', '--mapping_file', default='',
                        help='Mapping sheet that matches group_id to the input data')
    parser.add_argument('-o', '--output_folder', default='',
                        help='Output folder to save the results')
    parser.add_argument('-t', '--od_thresholds', nargs='+', type=int, default=[1, 5, 10, 15, 20, 25, 30],
                        help='Hard thresholding method for the scoring on outlier detection')
    parser.add_argument('-p', '--preprocess_data', action='store_true', default=False,
                        help='Scale and apply PCA to the features')
    parser.add_argument('-v', '--variance', default=0.8, type=float,
                        help='The percentage of explained variance if preprocess_data is true')
    args = parser.parse_args()

    # Read input data
    output = create_output_filenames(args.output_folder, args.input_file, args.method, args.kernel)
    df, pos_controls, neg_controls = read_input_data(args.input_file, args.controls_file,
                                                     output, args.variance, args.preprocess_data)

    # Score samples by modelling negative samples
    if args.method == 'SVM':
        df = oneclass_svm_method(df, neg_controls, output, args.kernel)
    elif args.method == 'MH':
        df = mahalanobis_distance_method(df, neg_controls, output)
    elif args.method == 'GMM':
        df = gmm_method(df, neg_controls, output)

    # Threshold scores for identifying outliers
    df = identify_outliers(df, neg_controls, output, args.od_thresholds)

    # Prepare penetrance files
    df_output = prepare_output_results(df, neg_controls, output,
                                       args.od_thresholds, args.mapping_file)

    # Plot performance results
    if len(pos_controls):
        calculate_performance(df_output, args.od_thresholds, neg_controls, pos_controls, output)
