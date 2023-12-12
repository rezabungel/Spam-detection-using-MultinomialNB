import math

import numpy as np
import pandas as pd

class SpamDetector:
    __smoothing_factor = 1.0 # When the smoothing factor is equal to 1, it corresponds to Laplace smoothing.

    def __init__(self, path_to_dataset_word_frequencies: str, path_to_dataset_email_ratios: str) -> None:
        
        try:
            self.__word_frequencies = pd.read_csv(path_to_dataset_word_frequencies, index_col=0)

            self.__count_unique_words = len(self.__word_frequencies) # N(Unique Words)
            self.__count_words_ham = sum(self.__word_frequencies.frequency_ham) # N(Words ∈ ham)
            self.__count_words_spam = sum(self.__word_frequencies.frequency_spam) # N(Words ∈ spam)

        except:
            pass # TO DO

        try:
            email_ratios = pd.read_csv(path_to_dataset_email_ratios, index_col=0)

            self.__ham_probability = email_ratios.ham['ratios-to-total-emails']
            self.__spam_probability = email_ratios.spam['ratios-to-total-emails']
        except:
            pass # TO DO

    def detecting_spam(self, message: str) -> int:
        '''
        Method for detecting spam in a message.
        
        Parameters:
            message (str): String containing the message text.
        
        Returns:
            int: Returns 1 if the message is considered spam, and 0 if it is ham.
        '''
        
        # TO DO: Preparing a message

        ham = math.log(self.__ham_probability) + self.__calculate_ham(message)
        spam = math.log(self.__spam_probability) + self.__calculate_spam(message)

        return int(spam > ham)

    def __calculate_ham(self, prepared_message: list[str]) -> float:
        '''
        TO DO: description

        Parameters:
            prepared_message (list[str]): TO DO: description.

        Returns:
            float: TO DO: description.

        Formula:
             n              N(Word_i ∈ ham) + smoothing_factor
             ∑ (log(--------------------------------------------------))
            i=1      N(Words ∈ ham) + smoothing_factor*N(Unique Words)

        Where:
            N(Word_i ∈ ham): Count of occurrences of the Word_i in ham emails.
            N(Words ∈ ham): Total count of all Words that appeared in ham emails.
            N(Unique Words): Count of all unique words.
            smoothing_factor: Smoothing parameter to avoid zero.
        '''

        sum_ham = 0
        numerator = self.__count_words_ham + self.__smoothing_factor*self.__count_unique_words

        sum_ham = sum([math.log((self.__word_frequencies.frequency_ham.get(word_i, 0)+self.__smoothing_factor) / numerator) for word_i in prepared_message])

        return sum_ham

    def __calculate_spam(self, prepared_message: list[str]) -> float:
        '''
        TO DO: description

        Parameters:
            prepared_message (list[str]): TO DO: description.

        Returns:
            float: TO DO: description.

        Formula:
             n              N(Word_i ∈ spam) + smoothing_factor
             ∑ (log(---------------------------------------------------))
            i=1      N(Words ∈ spam) + smoothing_factor*N(Unique Words)

        Where:
            N(Word_i ∈ spam): Count of occurrences of the Word_i in spam emails.
            N(Words ∈ spam): Total count of all Words that appeared in spam emails.
            N(Unique Words): Count of all unique words.
            smoothing_factor: Smoothing parameter to avoid zero.
        '''
        
        sum_spam = 0
        numerator = self.__count_words_spam + self.__smoothing_factor*self.__count_unique_words

        sum_spam = sum([math.log((self.__word_frequencies.frequency_spam.get(word_i, 0)+self.__smoothing_factor) / numerator) for word_i in prepared_message])

        return sum_spam

    @classmethod
    def set_smoothing_factor(cls, k: float) -> None:
        cls.__smoothing_factor = k



if __name__ == "__main__":
    
    path1 = '../dataset/model_data/word_frequencies.csv'
    path2 = '../dataset/model_data/email_ratios.csv'

    test = SpamDetector(path1, path2)