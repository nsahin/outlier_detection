import logging
import numpy as np
import pandas as pd
from datetime import datetime
from keras.layers import K
from keras.models import load_model
from skimage.io import imread

logging.basicConfig(level=logging.DEBUG, format='%(name)s:%(levelname)s:%(message)s')


def not_on_border(im, loc_left, loc_upper, loc_right, loc_lower):
    return loc_left >= 0 and loc_right <= im.shape[1] and loc_upper >= 0 and loc_lower <= im.shape[0]


def crop_cells(x, y, c, im):
    loc_left = int(x - c / 2)
    loc_upper = int(y - c / 2)
    loc_right = int(x + c / 2)
    loc_lower = int(y + c / 2)

    if not_on_border(im, loc_left, loc_upper, loc_right, loc_lower):
        return im[loc_upper:loc_lower, loc_left:loc_right]


def extract_features(functor, im, n, c):
    acts = functor([im.reshape(-1, c, c, n), 1.])
    return np.append(acts[-3][0], acts[-2][0])


def reshape_image(im, n, p):
    return np.rollaxis(im.reshape((-1, n, p, p)), 1, 4)


class ExtractFeatures:
    functor = None
    df = None
    multi_field_image = False
    out = None

    def __init__(self, input_file, mapping_file, model, channels, single_channel_only, crop):
        self.input_file = input_file
        self.mapping_file = mapping_file
        self.num_channels = channels
        self.model_name = model
        self.single_channel = single_channel_only
        self.crop_size = crop
        logging.info('Processing images with %d channels', self.num_channels)

    def load_trained_model(self):
        logging.info('Loading model')
        logging.info('Model name: %s' % self.model_name)
        model = load_model(self.model_name)
        self.functor = K.function([model.input, K.learning_phase()], [layer.output for layer in model.layers])

    def initialize_data(self):
        logging.info('Initializing data')
        # Load pixel data locations file
        df = pd.read_csv(self.input_file)
        df = df.astype({'center_x': int, 'center_y': int})

        # Pre-allocate memory for the results
        new_column_names = []
        for i in range(1024):
            new_column_names.append('FC%d' % (i+1))
        self.out = pd.DataFrame(columns=new_column_names, index=df.index, dtype=np.float32)
        self.df = df

    # noinspection PyUnresolvedReferences
    def _process_single_image(self, path, image_dataframe):
        image = imread(path)
        channels = []

        if self.num_channels == 1:
            if self.single_channel:
                channels.append(image[0::2])
            else:
                channels.append(image)
        else:
            channels.append(image[0::2])
            channels.append(image[1::2])

        cell_times = []
        border_cells = 0

        for rid, row in image_dataframe.iterrows():
            cell_start = datetime.now()
            row_channels = channels
            if self.multi_field_image:
                row_channels = [c[row.field] for c in row_channels]

            cropped = [crop_cells(row.center_x, row.center_y, self.crop_size, image) for image in row_channels]
            if cropped[0] is None:
                border_cells += 1
                continue

            #cell_image = np.dstack(cropped).reshape((self.num_channels, self.crop_size, self.crop_size))
            cell_image = reshape_image(np.dstack(cropped), self.num_channels, self.crop_size)

            np.copyto(self.out.loc[rid].values,
                      extract_features(self.functor, cell_image, self.num_channels, self.crop_size),
                      casting='no')

            cell_times.append(datetime.now() - cell_start)

        logging.info('Processed %s cells=%d, border_cells=%d, cell_avg_time=%s', path, len(cell_times), border_cells,
                     np.mean(cell_times))

    def process_images(self):
        logging.info('Processing images')
        images_start = datetime.now()
        # Crop cells and extract DL features
        for path, imdf in self.df.reindex(columns=('center_x', 'center_y', 'image_path')).groupby('image_path'):
            self._process_single_image(path, imdf)

        # drop rows with all NaN... that was a skipped cell
        self.out = self.out.dropna(how='all')

        # Adding common data back to results, joins by common index
        cols = [0]
        if 'group_id' in self.df.columns:
            cols.append(1)
        self.out = self.df.iloc[:, cols].join(self.out, how='right')

        # Save activations
        logging.info('Saving activations...')
        self.out.to_csv(self.input_file.replace('.csv', '_DL_features_samples.csv'), index=False)

        logging.info('Processed images in %s', datetime.now() - images_start)

        # Map with mapping sheet
        if self.mapping_file:
            self.out_group = pd.read_csv(self.mapping_file).merge(self.out, on=['group_id'])
            self.out_group.to_csv(self.input_file.replace('.csv', '_DL_features_groups.csv'), index=False)


if __name__ == '__main__':
    # Arguments
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_file', default='',
                        help='Input data, every row should have unique sample_id')
    parser.add_argument('-f', '--mapping_file', default='',
                        help='Mapping sheet that matches group_id to the input data')
    parser.add_argument('-m', '--model', default='',
                        help='Trained model')
    parser.add_argument('-c', '--crop_size', type=int, default=64,
                        help='Single cell bounding box size')
    parser.add_argument('-n', '--num_channels', type=int, default=1, choices=[1, 2],
                        help='Number of channels for the model and images, implemented only for 1 or 2 channels')
    parser.add_argument('-s', '--single_channel', action='store_false', default=True,
                        help='Extracts features from the first channel if the image is multi-channel')
    args = parser.parse_args()

    # Start logging
    logfile = logging.FileHandler(args.input_file.replace('.csv', '.log'))
    logfile.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        fmt='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logfile.setFormatter(formatter)
    logging.getLogger('').addHandler(logfile)

    # Extract features
    extract = ExtractFeatures(args.input_file, args.mapping_file, args.model,
                              args.num_channels, args.single_channel, args.crop_size)
    extract.load_trained_model()
    extract.initialize_data()
    extract.process_images()
