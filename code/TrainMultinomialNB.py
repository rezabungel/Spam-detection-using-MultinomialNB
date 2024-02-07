import multiprocessing
import time
import os
from functools import wraps

import pandas as pd

import TextProcessor


class TrainMultinomialNB:
    """
    A class for training a Multinomial Naive Bayes classifier for email classification.

    Class Attributes:
        __processor (TextProcessor): An instance of TextProcessor is used for text processing and tokenization.
            Default minimum word length is 2. Should be greater than or equal to 0.
            Default includes stop words from the nltk and wordcloud libraries.
        __min_word_frequency (int): The minimum word frequency threshold.
            Default is 15. Should be greater than or equal to 0.

    Attributes:
        __email_dataset (pd.DataFrame): The DataFrame containing email data for training.
        __path_to_email_dataset (str): The file path to the DataFrame file of emails and labels.
        __word_frequencies (dict): The dictionary containing word frequencies; derived from the DataFrame with email data for training.
        __dataframe_email_ratios (pd.DataFrame): The DataFrame containing ham and spam email ratios to total emails; derived from the DataFrame with email data for training.
        __dataframe_word_frequencies (pd.DataFrame): The DataFrame containing word frequencies; derived from the DataFrame with email data for training.
    """

    __processor = TextProcessor.TextProcessor()
    __min_word_frequency = 15

    def __init__(self, path_to_email_dataset: str) -> None:
        """
        Initializes a TrainMultinomialNB object.

        Parameters:
            path_to_email_dataset (str): The file path to the DataFrame file of emails and labels.
                The DataFrame must have the following structure:
                    Columns should be labeled as follows: text, label.
                        Note: label = 1 is spam, while label = 0 is ham.
                    Example:
                        ,text,label
                        0,Some text of some letter...,1
                        1,Some text of some letter...,1
                        2,Some text of some letter...,0
                        3,Some text of some letter...,1
        
        Raises:
            FileNotFoundError: If the specified DataFrame file of emails and labels is not found.
            AttributeError: If the DataFrame file of emails and labels has an incorrect structure.

        Returns:
            None
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

        self.__path_to_email_dataset = path_to_email_dataset
        self.__word_frequencies = {}
        self.__dataframe_email_ratios = pd.DataFrame
        self.__dataframe_word_frequencies = pd.DataFrame

    def __calculate_execution_time(method: callable) -> callable:
        """
        Decorator to calculate and print the execution time of the decorated method.

        Parameters:
            method (callable): The method to be decorated.

        Returns:
            callable: The decorated method.
        """
        @wraps(method)
        def wrapper(*args, **kwargs) -> None:
            """
            Wrapper function that prints the beginning and end of the decorated method's execution, along with the time spent on execution.

            Parameters:
                *args: Variable-length argument list.
                **kwargs: Arbitrary keyword arguments.

            Returns:
                None
            """

            if method.__name__ == 'train':
                print(f"The beginning of training.")
                start_time = time.time()
                method(*args, **kwargs)
                print(f"The end of training. The training was completed in {'%.3f' % (time.time() - start_time)} seconds.")
            elif method.__name__ in ['__pipeline_prepare_dataframe_email_ratios', '__pipeline_prepare_dataframe_word_frequencies']:
                print(f"\t{'*' * 60}")
                print(f'\tThe beginning of {method.__name__}.')
                start_time = time.time()
                method(*args)
                print(f"\tThe end of {method.__name__}. Time spent: {'%.3f' % (time.time() - start_time)} seconds.")
                print(f"\t{'*' * 60}")
            else:
                print(f"\t\t{'-' * 80}")
                print(f"\t\tThe beginning of {method.__name__}.")
                start_time = time.time()
                method(*args)
                print(f"\t\tThe end of {method.__name__}. Time spent: {'%.3f' % (time.time() - start_time)} seconds.")
                print(f"\t\t{'-' * 80}")

        return wrapper

    @__calculate_execution_time
    def train(self, output_folder_path: str = None) -> None:
        """
        Trains the Multinomial Naive Bayes classifier based on the provided email dataset during object initialization.

        Parameters:
            output_folder_path (str, optional): The folder path where the resulting DataFrames will be saved.
                If not provided or the folder does not exist, the DataFrames will be saved next to the DataFrame file containing emails and labels.

        Returns:
            None

        Note:
            The results of the trained Multinomial Naive Bayes classifier will be stored in two DataFrames:
                ham and spam email ratios to total emails, and word frequencies.
        """

        if output_folder_path is None or not os.path.exists(output_folder_path):
            output_folder_path = os.path.dirname(self.__path_to_email_dataset)

        path_to_save_dataset_word_frequencies = os.path.join(output_folder_path, 'word_frequencies.csv')
        path_to_save_dataset_email_ratios = os.path.join(output_folder_path, 'email_ratios.csv')

        self.__pipeline_prepare_dataframe_email_ratios(path_to_save_dataset_email_ratios)
        self.__pipeline_prepare_dataframe_word_frequencies(path_to_save_dataset_word_frequencies)

        print(f'\tThe dataset_email_ratios has been saved to "{path_to_save_dataset_email_ratios}".')
        print(f'\tThe dataset_word_frequencies has been saved to "{path_to_save_dataset_word_frequencies}".')

    @__calculate_execution_time
    def __pipeline_prepare_dataframe_email_ratios(self, path_to_save_dataset_email_ratios: str) -> None:
        """
        Implements a pipeline to prepare and save the DataFrame with ham and spam email ratios to total emails.
            This includes initializing and preparing the DataFrame with ham and spam email ratios to total emails
            and saving the resulting DataFrame to the specified path.

        Parameters:
            path_to_save_dataset_email_ratios (str): The file path to save the DataFrame file of email ratios.

        Returns:
            None
        """

        self.__initialize_dataframe_email_ratios()
        self.__save_dataframe(self.__dataframe_email_ratios, path_to_save_dataset_email_ratios)

    @__calculate_execution_time
    def __pipeline_prepare_dataframe_word_frequencies(self, path_to_save_dataset_word_frequencies: str) -> None:
        """
        Implements a pipeline to prepare and save the DataFrame with word frequencies.
            This includes merging text categories, processing text, initializing word frequencies, counting word frequencies,
            removing low-frequency words, sorting words by frequency, and saving the resulting DataFrame to the specified path.

        Parameters:
            path_to_save_dataset_word_frequencies (str): The file path to save the DataFrame file of word frequencies.

        Returns:
            None
        """

        self.__merge_text_category()

        print(f"\t\t{'-' * 80}")
        print(f"\t\tThe beginning of __prepare_text.")
        start_time = time.time()
        self.__email_dataset['text'] = self.__email_dataset['text'].apply(self.__prepare_text)
        print(f"\t\tThe end of prepare_text. Time spent: {'%.3f' % (time.time() - start_time)} seconds.")
        print(f"\t\t{'-' * 80}")

        self.__initialize_dict_word_frequencies()
        self.__count_word_frequencies()
        self.__remove_words_with_low_frequency_and_sort_word_frequencies()
        self.__initialize_dataframe_word_frequencies()
        self.__save_dataframe(self.__dataframe_word_frequencies, path_to_save_dataset_word_frequencies)

    @__calculate_execution_time
    def __merge_text_category(self) -> None:
        """
        Merges text categories in the email dataset based on labels.

        Parameters:
            None

        Returns:
            None
        """

        self.__email_dataset = self.__email_dataset.groupby('label')['text'].apply(lambda x: ' '.join(x)).reset_index()
        self.__email_dataset = self.__email_dataset[['text', 'label']]

    def __prepare_text(self, text: str) -> list[str]:
        """
        Processes and tokenizes the input text.

        Parameters:
            text (str): The input text to be processed and tokenized.

        Returns:
            list[str]: The list of processed and tokenized words.

        Note:
            This method utilizes multiprocessing to expedite the text processing.
            Parallelization is performed across all available CPU cores.
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
        Initializes the dictionary for word frequencies.

        Parameters:
            None

        Returns:
            None

        Note:
            The dictionary self.__word_frequencies is initialized such that each unique word in the texts of email messages
                is assigned a list [0, 0, 0], where the first element is the overall frequency of the word,
                the second is the frequency in ham emails, and the third is the frequency in spam emails.
        """

        unique_words = set.union(set(self.__email_dataset['text'][0]), set(self.__email_dataset['text'][1]))
        self.__word_frequencies = {key: [0, 0, 0] for key in unique_words}

    @__calculate_execution_time
    def __count_word_frequencies(self) -> None:
        """
        Counts word frequencies in the email dataset.

        Parameters:
            None

        Returns:
            None
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
        Removes low-frequency words and sorts the word frequencies dictionary in descending order.

        Parameters:
            None

        Returns:
            None

        Note:
            The threshold for low-frequency words is determined by the value set in the class attribute self.__min_word_frequency.
        """

        self.__word_frequencies = dict(sorted(filter(lambda item: item[1][0] >= self.__min_word_frequency, self.__word_frequencies.items()), key=lambda item: item[1][0], reverse=True))

    @__calculate_execution_time
    def __initialize_dataframe_word_frequencies(self) -> None:
        """
        Initializes the DataFrame for word frequencies.

        Parameters:
            None

        Returns:
            None

        Note:
            The DataFrame self.__dataframe_word_frequencies is initialized using the dictionary of word frequencies (i.e., self.__word_frequencies).
        """

        self.__dataframe_word_frequencies = pd.DataFrame.from_dict(self.__word_frequencies, orient='index', columns=['frequency', 'frequency_ham', 'frequency_spam'])

    @__calculate_execution_time
    def __save_dataframe(self, dataframe: pd.DataFrame, path_to_save: str) -> None:
        """
        Saves the given DataFrame to the specified file path.

        Parameters:
            dataframe (pd.DataFrame): The DataFrame to be saved.
            path_to_save (str): The file path to save the DataFrame.

        Returns:
            None
        """

        dataframe.to_csv(path_to_save)

    @__calculate_execution_time
    def __initialize_dataframe_email_ratios(self) -> None:
        """
        Initializes and prepares the DataFrame with ham and spam email ratios to total emails.

        Parameters:
            None

        Returns:
            None

        Note:
            Calculates ham and spam email ratios to total emails in the training DataFrame with email data.
            The resulting DataFrame (self.__dataframe_email_ratios) has 'ham' and 'spam' as columns and 'ratios-to-total-emails' as the index.
        """

        count_ham = len(self.__email_dataset[self.__email_dataset['label'] == 0])
        count_spam = len(self.__email_dataset[self.__email_dataset['label'] == 1])

        self.__dataframe_email_ratios = pd.DataFrame({'ham': [count_ham/(count_ham + count_spam)], 'spam': [count_spam/(count_ham + count_spam)]}, index=['ratios-to-total-emails'])

    @classmethod
    def set_processor(cls, min_length: int = None, user_stop_words: list[str] | tuple[str] | set[str] = None) -> None:
        """
        Set parameters for an instance of TextProcessor to be used for text processing and tokenization.

        Parameters:
            min_length (int, optional): Minimum word length.
                Default is 2. Should be greater than or equal to 0.
            user_stop_words (list[str] | tuple[str] | set[str], optional): The user-defined list of stop words. This list is added to the stop words defined by default.
                Default includes stop words from the nltk and wordcloud libraries.
                Example:
                    user_stop_words = ['ect', 'enron', 'hou', 'hpl', 'subject']

        Returns:
            None
        """

        cls.__processor = TextProcessor.TextProcessor(min_length, user_stop_words)

    @classmethod
    def set_min_word_frequency(cls, min_word_frequency: int) -> None:
        """
        Set the minimum word frequency threshold for filtering words in the dataset.

        Parameters:
            min_word_frequency (int): The minimum word frequency threshold. Should be greater than or equal to 0.

        Raises:
            ValueError: If the provided min_word_frequency is less than 0.

        Returns:
            None
        """

        if min_word_frequency >= 0:
            cls.__min_word_frequency = min_word_frequency
        else:
            raise ValueError(f"Error: min_word_frequency should be greater than or equal to 0. Got: {min_word_frequency}")


if __name__ == "__main__":

    path_to_email_dataset = '../dataset/email_dataset_1/model_data/spam_ham_dataset_ready_to_train.csv'

    output_folder_path = '../dataset/email_dataset_1/model_data/'

    training = TrainMultinomialNB(path_to_email_dataset)
    training.set_processor(user_stop_words=['ect', 'enron', 'hou', 'hpl', 'subject'])

    training.train(output_folder_path)