# Copyright 2012-2018 (C) Butterfly Network, Inc.

import logging
import os
import shutil
import tarfile
import urllib.request

import click
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim.nets import inception
from tqdm import tqdm

# Create top level logger
log = logging.getLogger()
log.setLevel(logging.INFO)

# Add console handler using our custom ColoredFormatter
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
log.addHandler(ch)

# Global parameters

# The image labels.
LABEL_NAMES = np.array([
    'morisons_pouch',
    'bladder',
    'plax',
    '4ch',
    '2ch',
    'ivc',
    'carotid',
    'lungs',
    'thyroid',
])
NUM_CLASSES = len(LABEL_NAMES)

# The size of the raw ultrasound images.
IMAGE_WIDTH = 436
IMAGE_HEIGHT = 512

# The default image size as required by the inception v1 model
TARGET_IMAGE_WIDTH = TARGET_IMAGE_HEIGHT = 224

LARGE_DATASET_URLS = [
    'https://github.com/ButterflyNetwork/MITGrandHack2018/releases/download'
    '/v.0.0.1/butterfly_dataset_test.tar.gz',
    'https://github.com/ButterflyNetwork/MITGrandHack2018/releases/download'
    '/v.0.0.1/butterfly_dataset_training1.tar.gz',
    'https://github.com/ButterflyNetwork/MITGrandHack2018/releases/download'
    '/v.0.0.1/butterfly_dataset_training2.tar.gz',
]

MINI_DATASET_URLS = [
    'https://github.com/ButterflyNetwork/MITGrandHack2018'
    '/releases/download/v.0.0.1/butterfly_mini_dataset.tar.gz'
]


@click.group()
def cli():
    pass


@click.option(
    '--dest_dir',
    required=True,
    default=os.getcwd(),
    type=click.Path(exists=True, dir_okay=True)
)
@click.option('--large', default=False, is_flag=True)
@cli.command()
def download_dataset(dest_dir, large):
    """ Download and extract the mini dataset.

    :param dest_dir: The directory where the dataset will be extracted.
    :params large: Indicate whether to download the large dataset.

    Example:
    python hackathon_example.py download_dataset

    To download the large dataset use:
    python hackathon_example.py download_dataset --large
    """
    urls = MINI_DATASET_URLS
    dataset_name = 'butterfly_mini_dataset'
    if large:
        urls = LARGE_DATASET_URLS
        dataset_name = 'butterfly_dataset'

    downloaded_files = []

    for url in urls:
        filename = os.path.basename(url)
        downloaded_files.append(filename)
        filepath = os.path.join(dest_dir, filename)
        if not os.path.exists(filepath):
            class TqdmUpTo(tqdm):
                def update_to(self, b=1, bsize=1, tsize=None):
                    if tsize is not None:
                        self.total = tsize
                    self.update(b * bsize - self.n)

            with TqdmUpTo(unit='B', unit_scale=True, miniters=1,
                          desc=os.path.basename(filename)) as progress_bar:
                urllib.request.urlretrieve(url, filename=filepath,
                                           reporthook=progress_bar.update_to)
                statinfo = os.stat(filepath)
                log.info('Successfully downloaded {} {} bytes.'.format(
                    filename, statinfo.st_size))
        else:
            log.info('Data segment is already available here: {}'.format(
                dest_dir))

    extracted_dir_path = os.path.join(
        dest_dir,
        dataset_name
    )

    if os.path.exists(extracted_dir_path):
        shutil.rmtree(extracted_dir_path)

    for downloaded_file in downloaded_files:
        log.info('Extracting the data from {}'.format(downloaded_file))
        tarfile.open(downloaded_file, 'r:gz').extractall(
            extracted_dir_path
        )

    # If large dataset, combined the training folders into a single folder.
    if large:
        os.chdir(extracted_dir_path)
        os.rename('training1', 'training')
        # Copy content of trianig2 into training
        for src in os.listdir("training2"):
            if os.path.isdir(src):
                dst = os.path.join('training', os.path.basename(src))
                shutil.copytree(src, dst)
            else:
                dst = 'training'
                shutil.move(os.path.join('training2', src), dst)
        os.rmdir("training2")

    log.info('The dataset {} is now available here {}.'.format(
        dataset_name, extracted_dir_path))


@cli.command()
@click.option(
    '--input_file',
    required=True,
    type=click.Path(exists=True, dir_okay=False)
)
@click.option('--batch_size', required=False, type=click.INT, default=36)
@click.option(
    '--export_dir',
    required=True,
    type=click.Path(exists=True, dir_okay=True)
)
def evaluate(input_file, batch_size, export_dir):
    """ Evaluates the given dataset

    :param input_file: the csv file containing the training set.
    :param batch_size: the batch size used for training
    :param export_dir: the checkpoint directory from which the model should
    be restored.

    Example:
    python hackathon_example.py evaluate
    --input_file=butterfly_mini_dataset/test/test.csv
    --export_dir=models
    """

    dataset_image_paths, dataset_labels = load_data_from_csv(input_file)

    with tf.Graph().as_default() as graph:

        # Define the data iterator.
        image_paths = tf.placeholder(tf.string, [None])
        label_data = tf.placeholder(tf.int32, [None])

        data_iterator = create_dataset_iterator(
            image_paths,
            label_data,
            batch_size,
        )
        next_test_batch = data_iterator.get_next()

        saver = tf.train.import_meta_graph(
            os.path.join(export_dir, 'butterfly-model.meta')
        )
        images = graph.get_tensor_by_name("images:0")
        labels = graph.get_tensor_by_name("labels:0")
        predictions = graph.get_tensor_by_name("predictions:0")
        # Add an accuracy node.
        accuracy_to_value, accuracy_update_op = tf.metrics.accuracy(
            predictions,
            labels,
        )
        local_init_op = tf.local_variables_initializer()

        # Load from check-point
        with tf.Session() as session:

            # Restore model.
            saver.restore(session, tf.train.latest_checkpoint(export_dir))

            # Initialize the iterator.
            session.run([local_init_op, data_iterator.initializer],
                        feed_dict={
                            image_paths: dataset_image_paths,
                            label_data: dataset_labels,
                        })

            while True:
                try:
                    # Read the next batch.
                    batch_images, batch_labels = session.run(next_test_batch)

                    # Evaluating the model.
                    session.run(accuracy_update_op,
                                feed_dict={
                                    images: batch_images,
                                    labels: batch_labels,
                                })
                except tf.errors.OutOfRangeError:
                    break

            accuracy = session.run(accuracy_to_value)
            log.info('test accuracy: {:.1%}'.format(accuracy))


@cli.command()
@click.option(
    '--input_file',
    required=True,
    type=click.Path(exists=True, dir_okay=False)
)
@click.option('--batch_size', required=False, type=click.INT, default=36)
@click.option('--number_of_epochs', required=False, type=click.INT, default=10)
@click.option(
    '--export_dir',
    required=True,
    type=click.Path(exists=False, dir_okay=True)
)
def train(input_file, batch_size, number_of_epochs, export_dir):
    """
    :param input_file: the csv file containing the training set.
    :param batch_size: the batch size used for training.
    :param number_of_epochs: the number of times the model will be trained
    on all the dataset.
    :param export_dir: The directory where the model will be saved.

    Example:
    python hackathon_example.py train
    --input_file=butterfly_mini_dataset/training/training.csv
    --export_dir=models
    """

    train_image_paths, train_labels, valid_image_paths, valid_labels = \
        load_data_from_csv(input_file, split=True)

    # Define the data iterators.
    image_paths = tf.placeholder(tf.string, [None])
    label_data = tf.placeholder(tf.int32, [None])

    training_iterator = create_dataset_iterator(
        image_paths,
        label_data,
        batch_size,
    )
    next_train_batch = training_iterator.get_next()

    valid_iterator = create_dataset_iterator(
        image_paths,
        label_data,
        batch_size,
    )
    next_valid_batch = valid_iterator.get_next()

    # Define input and output to the inception v1 model.
    images = tf.placeholder(
        tf.float32,
        [None, TARGET_IMAGE_WIDTH, TARGET_IMAGE_HEIGHT, 3],
        name="images"
    )
    labels = tf.placeholder(tf.int64, [None], name="labels")

    # Define inception v1 and return ops to load pre-trained model (trained on
    # ImageNet).

    restore_op, feed_dict, train_op, metrics_to_values, metrics_to_updates \
        = create_model(images, labels)

    init_local_op = tf.local_variables_initializer()
    init_op = tf.group(tf.global_variables_initializer(), init_local_op)

    # Start the training validation loop.
    with tf.Session() as session:

        session.run(init_op)
        session.run(restore_op, feed_dict=feed_dict)

        # Define a ModelSaver
        saver = tf.train.Saver()

        best_validation_accuracy = None

        # Running training loop.
        for _ in range(number_of_epochs):

            session.run([init_local_op, training_iterator.initializer],
                        feed_dict={
                            image_paths: train_image_paths,
                            label_data: train_labels,
                        }
                        )

            while True:
                try:
                    # Read the next batch.
                    batch_images, batch_labels = session.run(next_train_batch)
                    # Train the model.
                    session.run([metrics_to_updates, train_op],
                                feed_dict={
                                    images: batch_images,
                                    labels: batch_labels,
                                })
                except tf.errors.OutOfRangeError:
                    break

            metrics_values = session.run(metrics_to_values)

            accuracy = metrics_values['accuracy']
            mean_loss = metrics_values['mean_loss']
            log.info('training accuracy: {:.1%}, '
                     'training mean loss: {}'.format(accuracy, mean_loss))

            # Running validation loop.

            session.run(
                [init_local_op, valid_iterator.initializer],
                feed_dict={
                    image_paths: valid_image_paths,
                    label_data: valid_labels,
                })

            while True:
                try:
                    # Read the next batch.
                    batch_images, batch_labels = session.run(next_valid_batch)

                    # Train the model.
                    session.run(
                        metrics_to_updates,
                        feed_dict={
                            images: batch_images,
                            labels: batch_labels,
                        })
                except tf.errors.OutOfRangeError:
                    break

            metrics_values = session.run(metrics_to_values)
            accuracy = metrics_values['accuracy']
            mean_loss = metrics_values['mean_loss']
            log.info(
                'validation accuracy: {:.1%}, validation mean loss: {}'.format(
                    accuracy,
                    mean_loss))
            # Save model if accuracy improved.
            if (
                    (not best_validation_accuracy) or
                    best_validation_accuracy < accuracy
            ):
                best_validation_accuracy = accuracy
                saver.save(session, os.path.join(export_dir,
                                                 'butterfly-model'))


def create_model(images, labels):
    """
    This methods initialize the inception v1 model with weights generated
    from training on the ImageNet dataset for all layers expect the last.
    The last layer is adjusted to output only 9 classes (instead of the
    1000 required for ImageNet). Note also that the methods set the model
    for fine-tuning meaning that during training only the last layer's
    weights can change.

    :param images: A tensor containing the images.
    :param labels: A tensor representing the correct labels for the images.

    :return restore_op: The operation used to restore the weights of the model.
    :return feed_dict: The feed_dict used for restoring the model.
    :return train_op: The train_op used to train the model.
    :return metrics_to_values: The metrics collected when training.
    :return metrics_to_updates: The metrics update op used when training.
    """
    with slim.arg_scope(inception.inception_v1_arg_scope()):
        #  Load the deep learning model.
        logits, end_points = inception.inception_v1(
            images,
            num_classes=NUM_CLASSES,
            is_training=False
        )

        # We are going to train only the last layer of the model.
        trainable_layer = 'InceptionV1/Logits/Conv2d_0c_1x1'

        variables_to_restore = slim.get_variables_to_restore(
            exclude=[trainable_layer]
        )
        variables_to_train = slim.get_variables_by_suffix('', trainable_layer)

        # Transform the labels into one hot encoding.
        one_hot_labels = tf.one_hot(
            labels,
            NUM_CLASSES,
        )

        # Define the loss function.
        loss = tf.losses.softmax_cross_entropy(
            one_hot_labels,
            end_points['Logits'],
        )

        # Select the optimizer.
        optimizer = tf.train.AdamOptimizer(1e-4)

        # Create a train op.
        train_op = tf.contrib.training.create_train_op(
            loss,
            optimizer,
            variables_to_train=variables_to_train,
        )

        predictions = tf.argmax(
            end_points['Predictions'], 1, name="predictions"
        )
        metrics_to_values, metrics_to_updates = \
            slim.metrics.aggregate_metric_map({
                'accuracy': tf.metrics.accuracy(predictions, labels),
                'mean_loss': tf.metrics.mean(loss),
            })

        # Define load predefined model operation.
        restore_op, feed_dict = slim.assign_from_checkpoint(
            'inception_v1.ckpt',
            variables_to_restore
        )

        return (
            restore_op,
            feed_dict,
            train_op,
            metrics_to_values,
            metrics_to_updates,
        )


def create_dataset_iterator(
        image_placeholder,
        label_placeholder,
        batch_size,
):
    """

    :param image_placeholder: A placeholder for the images.
    :param label_placeholder: A placeholder for the labels.
    :param batch_size: The batch size used by the iterator.
    :return: A tensorflow iterator that can be used to iterate over the
    dataset.
    """
    dataset = tf.data.Dataset.from_tensor_slices(
        (image_placeholder, label_placeholder)
    )
    dataset = dataset.map(load_image)
    dataset = dataset.cache()
    dataset = dataset.batch(batch_size)
    return dataset.make_initializable_iterator()


def load_data_from_csv(filename, split=False, split_percentage=0.8):
    """
    :param filename: The path to the file to be loaded.
    :param split: whether to split the data into train and validation.
    :param split_percentage: The percentage that will be retained as
    :return: A tuple containing 2 lists in case there is no split: one with
    the image paths and one with the corresponding labels. If split is true
    the method returns 4 lists (2 for training and 2 for validation).
    """
    df = pd.read_csv(filename)
    df = df.sample(frac=1).reset_index(drop=True)
    if split:
        mask = np.random.rand(len(df)) < split_percentage
        training_set = df[mask]
        validation_set = df[~mask]
        return (
            training_set['image_file_path'].tolist(),
            training_set['label'].tolist(),
            validation_set['image_file_path'].tolist(),
            validation_set['label'].tolist(),
        )
    else:
        return (
            df['image_file_path'].tolist(),
            df['label'].tolist(),
        )


def load_image(filepath, label):
    """

    :param filepath: A tensor representing the filepath of the image
    :param label: The label for the image.
    :return: A tensor representing the image ready to be used in the inception
    model and its label.
    """

    image_string = tf.read_file(filepath)
    image_decoded = tf.image.decode_image(
        image_string,
        channels=1
    )
    image_resized = tf.image.resize_image_with_crop_or_pad(
        image_decoded,
        IMAGE_WIDTH,
        IMAGE_HEIGHT
    )
    image_resized = tf.image.resize_images(
        image_resized,
        (
            TARGET_IMAGE_WIDTH,
            TARGET_IMAGE_HEIGHT
        )
    )
    # Normalize the image.
    image_normalized = image_resized / 255
    image = tf.reshape(
        tf.cast(image_normalized, tf.float32),
        shape=(TARGET_IMAGE_WIDTH, TARGET_IMAGE_HEIGHT)
    )
    # Stack the image 3 times since the pre-trained inception model
    # required a 3 channel image. This can be optimized by instantiating
    # inception with 1 channel and retrain the first layer from scratch.
    return tf.stack([image, image, image], axis=2), label


# This setup the script so it can be used with different command groups from
# command line.
if __name__ == '__main__':
    cli()
