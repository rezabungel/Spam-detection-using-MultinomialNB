import time

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
            
            # TO DO.
        except:
            # TO DO.
            pass

    def train(self, path_to_save_dataset_word_frequencies: str = None, path_to_save_dataset_email_ratios: str = None) -> None:
        """
        TO DO
        """
        
        print(f"The beginning of training.")
        start_time = time.time()

        self.__pipeline_prepare_dataframe_frequency_words(path_to_save_dataset_word_frequencies)
        self.__pipeline_prepare_dataframe_email_ratios(path_to_save_dataset_email_ratios)

        print(f"The end of training. The training was completed in {'%.3f' % (time.time() - start_time)} seconds.")

    def __pipeline_prepare_dataframe_frequency_words(self, path_to_save_dataset_word_frequencies: str) -> None:
        """
        TO DO
        """

        email_dataset = self.__merge_text_category()
        email_dataset['text'] = email_dataset['text'].apply(self.__prepare_text)

        frequency_words = self.__create_dict_frequency_words(email_dataset['text'][0], email_dataset['text'][1])
        frequency_words = self.__count_words_frequency(email_dataset, frequency_words, -1)
        frequency_words = self.__sort_words_frequency(frequency_words)
        frequency_words = self.__count_words_frequency(email_dataset, frequency_words, 0)
        frequency_words = self.__count_words_frequency(email_dataset, frequency_words, 1)

        dataframe_frequency_words = self.__dict_frequency_words_to_dataframe(frequency_words)
        dataframe_frequency_words = self.__remove_low_frequency_words(dataframe_frequency_words)
        
        self.__save_dataframe(dataframe_frequency_words, path_to_save_dataset_word_frequencies)

    def __pipeline_prepare_dataframe_email_ratios(self, path_to_save_dataset_email_ratios: str) -> None:
        
        dataframe_email_ratios = self.__create_dataframe_email_ratios()
        self.__save_dataframe(dataframe_email_ratios, path_to_save_dataset_email_ratios)

    def __merge_text_category(self) -> pd.DataFrame:
        """
        TO DO
        """

        email_dataset = self.__email_dataset.groupby('label')['text'].apply(lambda x: ' '.join(x)).reset_index()
        return email_dataset[['text', 'label']]

    def __prepare_text(self, text: str) -> list[str]:
        """
        TO DO
        """

        return self.__processor.pipeline(text)
 
    def __create_dict_frequency_words(self, list_ham_words: list[str], list_spam_words: list[str]) -> dict[str, list[int, int, int]]:
        """
        TO DO
        """

        unique_words = set.union(set(list_ham_words), set(list_spam_words))
        frequency_words = {key: [0, 0, 0] for key in unique_words}

        return frequency_words

    def __count_words_frequency(self, email_dataset: pd.DataFrame, frequency_words: dict[str, list[int, int, int]], label: int) -> dict[str, list[int, int, int]]:
        """
        TO DO
        """

        if label == -1:
            for list_words in email_dataset['text']:
                for word in list_words:
                    frequency_words[word][0] += 1

            return frequency_words

        if label == 0:
            for word in email_dataset['text'][0]:
                frequency_words[word][1] += 1

            return frequency_words

        if label == 1:
            for word in email_dataset['text'][1]:
                frequency_words[word][2] += 1

            return frequency_words
        
    def __sort_words_frequency(self, frequency_words: dict[str, list[int, int, int]]) -> dict[str, list[int, int, int]]:
        """
        TO DO
        """

        return dict(sorted(frequency_words.items(), key=lambda item: item[1][0], reverse=True))

    def __dict_frequency_words_to_dataframe(self, frequency_words: dict[str, list[int, int, int]]) -> pd.DataFrame:
        """
        TO DO
        """

        return pd.DataFrame.from_dict(frequency_words, orient='index', columns=['frequency', 'frequency_ham', 'frequency_spam'])

    def __remove_low_frequency_words(self, dataframe_frequency_words: pd.DataFrame) -> pd.DataFrame:
        """
        TO DO
        """

        return dataframe_frequency_words[dataframe_frequency_words['frequency'] >= self.__min_word_frequency]

    def __save_dataframe(self, dataframe: pd.DataFrame, path_to_save: str) -> None:
        """
        TO DO
        """

        dataframe.to_csv(path_to_save)

    def __create_dataframe_email_ratios(self) -> pd.DataFrame:
        """
        TO DO
        """

        count_ham = len(self.__email_dataset[self.__email_dataset['label'] == 0])
        count_spam = len(self.__email_dataset[self.__email_dataset['label'] == 1])

        return pd.DataFrame({'ham': [count_ham/(count_ham + count_spam)], 'spam': [count_spam/(count_ham + count_spam)]}, index=['ratios-to-total-emails'])

    @classmethod
    def set_min_word_frequencyr(cls, min_word_frequency: int) -> None:
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