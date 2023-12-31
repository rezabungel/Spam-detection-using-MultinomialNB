import multiprocessing
import time
from functools import wraps

import pandas as pd

import TextProcessor


class TrainMultinomialNB:
    """
    TO DO
    """

    __processor = TextProcessor.TextProcessor()
    __min_word_frequency = 15

    def __init__(self, path_to_email_dataset: str) -> None:
        """
        TO DO
        """

        try:
            self.__email_dataset = pd.read_csv(path_to_email_dataset, index_col=0)

        except FileNotFoundError:
            raise FileNotFoundError(
                f'\n\tError: The DataFrame file of emails and labels is not found at "{path_to_email_dataset}".\n'
                f"\tThe DataFrame must have the following structure:\n"
                f"\t\tColumns should be labeled as follows: text, label.\n"
                f"\tNote:\n"
                f"\t\tlabel = 1 is spam, while label = 0 is ham.\n\n"
                f"\tExample:\n"
                f"\t\t,text,label\n"
                f"\t\t0,Some text of some letter...,1\n"
                f"\t\t1,Some text of some letter...,1\n"
                f"\t\t2,Some text of some letter...,0\n"
                f"\t\t3,Some text of some letter...,1\n"
            )

        if not(['text', 'label'] == list(self.__email_dataset.columns) or ['label', 'text'] == list(self.__email_dataset.columns)):
            raise AttributeError(
                f'\n\tError: The DataFrame file of emails and labels has an incorrect structure.\n'
                f"\tThe DataFrame must have the following structure:\n"
                f"\t\tColumns should be labeled as follows: text, label.\n"
                f"\tNote:\n"
                f"\t\tlabel = 1 is spam, while label = 0 is ham.\n\n"
                f"\tExample:\n"
                f"\t\t,text,label\n"
                f"\t\t0,Some text of some letter...,1\n"
                f"\t\t1,Some text of some letter...,1\n"
                f"\t\t2,Some text of some letter...,0\n"
                f"\t\t3,Some text of some letter...,1\n"
            )

        self.__word_frequencies = {}

        self.__dataframe_email_ratios = pd.DataFrame
        self.__dataframe_word_frequencies = pd.DataFrame

    def __calculate_execution_time(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if func.__name__ == 'train':
                print(f"The beginning of training.")
                start_time = time.time()
                func(*args, **kwargs)
                print(f"The end of training. The training was completed in {'%.3f' % (time.time() - start_time)} seconds.")
            elif func.__name__ in ['__pipeline_prepare_dataframe_email_ratios', '__pipeline_prepare_dataframe_word_frequencies']:
                print(f"\t{'*' * 60}")
                print(f'\tThe beginning of {func.__name__}.')
                start_time = time.time()
                func(*args)
                print(f"\tThe end of {func.__name__}. Time spent: {'%.3f' % (time.time() - start_time)} seconds.")
                print(f"\t{'*' * 60}")
            else:
                print(f"\t\t{'-' * 80}")
                print(f"\t\tThe beginning of {func.__name__}.")
                start_time = time.time()
                func(*args)
                print(f"\t\tThe end of {func.__name__}. Time spent: {'%.3f' % (time.time() - start_time)} seconds.")
                print(f"\t\t{'-' * 80}")

        return wrapper

    @__calculate_execution_time
    def train(self, path_to_save_dataset_word_frequencies: str = None, path_to_save_dataset_email_ratios: str = None) -> None:
        """
        TO DO
        """

        self.__pipeline_prepare_dataframe_email_ratios(path_to_save_dataset_email_ratios)
        self.__pipeline_prepare_dataframe_word_frequencies(path_to_save_dataset_word_frequencies)
        print(f'\tThe dataset_word_frequencies has been saved to "{path_to_save_dataset_word_frequencies}".')
        print(f'\tThe dataset_email_ratios has been saved to "{path_to_save_dataset_email_ratios}".')

    @__calculate_execution_time
    def __pipeline_prepare_dataframe_email_ratios(self, path_to_save_dataset_email_ratios: str) -> None:
        """
        TO DO
        """

        self.__initialize_dataframe_email_ratios()
        self.__save_dataframe(self.__dataframe_email_ratios, path_to_save_dataset_email_ratios)

    @__calculate_execution_time
    def __pipeline_prepare_dataframe_word_frequencies(self, path_to_save_dataset_word_frequencies: str) -> None:
        """
        TO DO
        """

        self.__merge_text_category()

        print(f"\t\tThe beginning of __prepare_text.")
        start_time = time.time()
        self.__email_dataset['text'] = self.__email_dataset['text'].apply(self.__prepare_text)
        print(f"\t\tThe end of prepare_text. Time spent: {'%.3f' % (time.time() - start_time)} seconds.")

        self.__initialize_dict_word_frequencies()
        self.__count_word_frequencies()
        self.__remove_words_with_low_frequency_and_sort_word_frequencies()
        self.__initialize_dataframe_word_frequencies()
        self.__save_dataframe(self.__dataframe_word_frequencies, path_to_save_dataset_word_frequencies)

    @__calculate_execution_time
    def __merge_text_category(self) -> None:
        """
        TO DO
        """

        self.__email_dataset = self.__email_dataset.groupby('label')['text'].apply(lambda x: ' '.join(x)).reset_index()
        self.__email_dataset = self.__email_dataset[['text', 'label']]

    def __prepare_text(self, text: str) -> list[str]:
        """
        TO DO
        """

        words = text.split()
        text = []

        step = int(len(words) / (multiprocessing.cpu_count()))
        
        for i in range(multiprocessing.cpu_count()-1):
            text.append(' '.join(words[step*i:step*(i+1)]))
        else:
            text.append(' '.join(words[step*(multiprocessing.cpu_count()-1):len(words)]))

        with multiprocessing.Pool(multiprocessing.cpu_count()) as p:
            processed_text = p.map(self.__processor.pipeline, text)

        return [word for sublist in processed_text for word in sublist]

    @__calculate_execution_time
    def __initialize_dict_word_frequencies(self) -> None:
        """
        TO DO
        """

        unique_words = set.union(set(self.__email_dataset['text'][0]), set(self.__email_dataset['text'][1]))
        self.__word_frequencies = {key: [0, 0, 0] for key in unique_words}

    @__calculate_execution_time
    def __count_word_frequencies(self):
        """
        TO DO
        """

        for word in self.__email_dataset['text'][0]:
            self.__word_frequencies[word][0] += 1
            self.__word_frequencies[word][1] += 1

        for word in self.__email_dataset['text'][1]:
            self.__word_frequencies[word][0] += 1
            self.__word_frequencies[word][2] += 1

    @__calculate_execution_time
    def __remove_words_with_low_frequency_and_sort_word_frequencies(self) -> None:
        """
        TO DO
        """

        self.__word_frequencies = dict(sorted(filter(lambda item: item[1][0] >= self.__min_word_frequency, self.__word_frequencies.items()), key=lambda item: item[1][0], reverse=True))

    @__calculate_execution_time
    def __initialize_dataframe_word_frequencies(self):
        """
        TO DO
        """

        self.__dataframe_word_frequencies = pd.DataFrame.from_dict(self.__word_frequencies, orient='index', columns=['frequency', 'frequency_ham', 'frequency_spam'])

    @__calculate_execution_time
    def __save_dataframe(self, dataframe: pd.DataFrame, path_to_save: str) -> None:
        """
        TO DO
        """

        dataframe.to_csv(path_to_save)

    @__calculate_execution_time
    def __initialize_dataframe_email_ratios(self) -> None:
        """
        TO DO
        """

        count_ham = len(self.__email_dataset[self.__email_dataset['label'] == 0])
        count_spam = len(self.__email_dataset[self.__email_dataset['label'] == 1])

        self.__dataframe_email_ratios = pd.DataFrame({'ham': [count_ham/(count_ham + count_spam)], 'spam': [count_spam/(count_ham + count_spam)]}, index=['ratios-to-total-emails'])

    @classmethod
    def set_min_word_frequency(cls, min_word_frequency: int) -> None:
        """
        TO DO
        """

        if min_word_frequency >= 0:
            cls.min_word_frequency = min_word_frequency
        else:
            raise ValueError(f"Error: min_word_frequency should be greater than or equal to 0. Got: {min_word_frequency}")


if __name__ == "__main__":

    path_to_email_dataset = '../dataset/email_dataset_1/model_data/spam_ham_dataset_ready_to_train.csv'

    path_to_save_dataset_word_frequencies = '../dataset/email_dataset_1/model_data/word_frequencies.csv'
    path_to_save_dataset_email_ratios = '../dataset/email_dataset_1/model_data/email_ratios.csv'

    test = TrainMultinomialNB(path_to_email_dataset)
    test.train(path_to_save_dataset_word_frequencies, path_to_save_dataset_email_ratios)