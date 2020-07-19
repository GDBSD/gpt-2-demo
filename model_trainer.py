# -*- coding: utf-8 -*-
import pandas as pd
import torch

from simpletransformers.language_modeling import LanguageModelingModel

import logging
logging.basicConfig(level=logging.INFO)

transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

DATA_FOLDER = 'data'
MODEL = 'gpt-2-medium'


def get_training_args():
    train_args = {
        "reprocess_input_data": True,
        "overwrite_output_dir": True,
        "max_length": 1000,
        "block_size:": 250,
        "train_batch_size": 32,
        "num_train_epochs": 3,
        "mlm": False
    }
    return train_args


def prep_data():
    """ Create the training and test text files

    :return: tuple with paths to the training and test text files
    """
    data = f'{DATA_FOLDER}/merged_books.txt'
    df = pd.read_csv(data, sep="\t", header=None)
    df.columns = ['sentence']

    # Create a list of sentences
    sentences = df['sentence'].tolist()

    # Create the paths for the training and test data files
    train_data_filepath = f'{DATA_FOLDER}/train.txt'
    test_data_filepath = f'{DATA_FOLDER}/test.txt'

    # Create the training and test text files
    with open(train_data_filepath, 'w') as f:
        for sentence in sentences[:-10]:
            f.writelines(sentence + "\n")
    with open(test_data_filepath, 'w') as f:
        for sentence in sentences[-10:]:
            f.writelines(sentence + "\n")

    # Return the paths to the files
    return train_data_filepath, test_data_filepath


def train_model(train_data, test_data, train_args):
    """ Train the model and save it to the outputs folder

    :param test_data:
    :param train_data:
    :param train_args: dict - arguments to be passed into the train function
    :return:
    """
    use_cuda = torch.cuda.is_available()
    logging.info(f'Using CUDA: {use_cuda}')
    model = LanguageModelingModel('gpt2', MODEL, args=train_args, use_cuda=use_cuda)
    model.train_model(train_data, eval_file=test_data)
    model.eval_model(test_data)


if __name__ == '__main__':
    data_files = prep_data()
    training_args = get_training_args()
    train_model(data_files[0], data_files[1], training_args)
