import math

import numpy as np
import pandas as pd

import TextProcessor

class SpamDetector:
    __smoothing_factor = 1.0 # When the smoothing factor is equal to 1, it corresponds to Laplace smoothing.

    def __init__(self, path_to_dataset_word_frequencies: str, path_to_dataset_email_ratios: str) -> None:
        
        try:
            self.__word_frequencies = pd.read_csv(path_to_dataset_word_frequencies, index_col=0)

            self.__count_unique_words = len(self.__word_frequencies.frequency) # N(Unique Words)
            self.__count_words_ham = sum(self.__word_frequencies.frequency_ham) # N(Words ∈ ham)
            self.__count_words_spam = sum(self.__word_frequencies.frequency_spam) # N(Words ∈ spam)

        except FileNotFoundError:
            raise FileNotFoundError(
                f'\n\tError: The DataFrame file of word frequencies is not found at "{path_to_dataset_word_frequencies}".\n'
                f"\tThe DataFrame must have the following structure:\n"
                f"\t\tColumns should be labeled as follows: frequency, frequency_ham, frequency_spam.\n"
                f"\t\tIndexes should be made up of words.\n\n"
                f"\tExample:\n"
                f"\t\t,frequency,frequency_ham,frequency_spam\n"
                f"\t\tdeal,3655,3549,106\n"
                f"\t\tpleas,3243,2737,506\n"
                f"\t\tga,3034,2861,173\n"
                f"\t\tmeter,2721,2718,3\n"
                f"\t\tthank,2304,2125,179\n"
            )
        except AttributeError:
            raise AttributeError(
                f"\n\tError: The DataFrame file of word frequencies has an incorrect structure.\n"
                f"\tThe DataFrame must have the following structure:\n"
                f"\t\tColumns should be labeled as follows: frequency, frequency_ham, frequency_spam.\n"
                f"\t\tIndexes should be made up of words.\n\n"
                f"\tExample:\n"
                f"\t\t,frequency,frequency_ham,frequency_spam\n"
                f"\t\tdeal,3655,3549,106\n"
                f"\t\tpleas,3243,2737,506\n"
                f"\t\tga,3034,2861,173\n"
                f"\t\tmeter,2721,2718,3\n"
                f"\t\tthank,2304,2125,179\n"
            )

        try:
            email_ratios = pd.read_csv(path_to_dataset_email_ratios, index_col=0)

            self.__ham_probability = email_ratios.ham['ratios-to-total-emails'] # P(ham)
            self.__spam_probability = email_ratios.spam['ratios-to-total-emails'] # P(spam)

        except FileNotFoundError:
            raise FileNotFoundError(
                f'\n\tError: The DataFrame file containing the ratios of ham and spam emails to total emails is not found at "{path_to_dataset_email_ratios}".\n'
                f"\tThe DataFrame must have the following structure:\n"
                f"\t\tColumns should be labeled as follows: ham, spam.\n"
                f"\t\tIndex should be only one and named: ratios-to-total-emails.\n\n"
                f"\tExample:\n"
                f"\t\t,ham,spam\n"
                f"\t\tratios-to-total-emails,0.7127329192546584,0.28726708074534163\n"
            )
        except (AttributeError, KeyError):
            raise AttributeError(
                f"\n\tError: The DataFrame file containing the ratios of ham and spam emails to total emails has an incorrect structure.\n"
                f"\tThe DataFrame must have the following structure:\n"
                f"\t\tColumns should be labeled as follows: ham, spam.\n"
                f"\t\tIndex should be only one and named: ratios-to-total-emails.\n\n"
                f"\tExample:\n"
                f"\t\t,ham,spam\n"
                f"\t\tratios-to-total-emails,0.7127329192546584,0.28726708074534163\n"
            )

    def detecting_spam(self, message: str) -> int:
        """
        Method for detecting spam in a message.
        
        Parameters:
            message (str): String containing the message text.
        
        Returns:
            int: Returns 1 if the message is considered spam, and 0 if it is ham.
        """
        
        processor = TextProcessor.TextProcessor()
        prepared_message = processor.pipeline(message)

        ham = self.__log_probability_ham_email(self.__ham_probability) + self.__sum_log_prob_words_ham(prepared_message)
        spam = self.__log_probability_spam_email(self.__spam_probability) + self.__sum_log_prob_words_spam(prepared_message)

        return int(spam > ham)
    
    def __log_probability_ham_email(self, ham_email_probability: float) -> float:
        """
        Calculate the log probability of receiving ham email.

        Parameters:
            ham_email_probability (float): The probability of receiving ham email.

        Returns:
            float: The logarithm of ham email probability.

        Formula:
                               N ∈ ham
            log(P(ham)) = log(---------) = log(ham_email_probability)
                                  N

        Where:
            N ∈ ham: Count of ham emails.
            N: Count of all emails.

        Note:
            The ham_email_probability is used; the formula simply underscores information on how this variable was obtained.
        """

        return math.log(ham_email_probability)

    def __log_probability_spam_email(self, spam_email_probability: float) -> float:
        """
        Calculate the log probability of receiving spam email.

        Parameters:
            spam_email_probability (float): The probability of receiving spam email.

        Returns:
            float: The logarithm of spam email probability.

        Formula:
                                N ∈ spam
            log(P(spam)) = log(----------) = log(spam_email_probability)
                                    N

        Where:
            N ∈ spam: Count of spam emails.
            N: Count of all emails.

        Note:
            The spam_email_probability is used; the formula simply underscores information on how this variable was obtained.
        """

        return math.log(spam_email_probability)

    def __sum_log_prob_words_ham(self, prepared_message: list[str]) -> float:
        """
        Calculate the sum of log probabilities of words from a given message, given that they are in the ham category.
        
        Parameters:
            prepared_message (list[str]): The list of processed words in the message.

        Returns:
            float: The sum of log probabilities of words in the ham category from the given message.

        Formula:
            n                              n             N(Word_i ∈ ham) + smoothing_factor
            ∑ log(P(Word_i ∈ ham | ham)) = ∑ log(--------------------------------------------------)
           i=1                            i=1     N(Words ∈ ham) + smoothing_factor*N(Unique Words)

        Where:
            N(Word_i ∈ ham): Count of occurrences of the Word_i in ham emails.
            N(Words ∈ ham): Total count of all Words that appeared in ham emails.
            N(Unique Words): Count of all unique words.
            smoothing_factor: Smoothing parameter to avoid zero.
        """

        sum_log_prob_word_i_ham = 0
        numerator = self.__count_words_ham + self.__smoothing_factor*self.__count_unique_words

        sum_log_prob_word_i_ham = sum([math.log((self.__word_frequencies.frequency_ham.get(word_i, 0)+self.__smoothing_factor) / numerator) for word_i in prepared_message])

        return sum_log_prob_word_i_ham

    def __sum_log_prob_words_spam(self, prepared_message: list[str]) -> float:
        """
        Calculate the sum of log probabilities of words from a given message, given that they are in the spam category.
        
        Parameters:
            prepared_message (list[str]): The list of processed words in the message.

        Returns:
            float: The sum of log probabilities of words in the spam category from the given message.

        Formula:
            n                                n             N(Word_i ∈ spam) + smoothing_factor
            ∑ log(P(Word_i ∈ spam | spam)) = ∑ log(---------------------------------------------------)
           i=1                              i=1     N(Words ∈ spam) + smoothing_factor*N(Unique Words)

        Where:
            N(Word_i ∈ spam): Count of occurrences of the Word_i in spam emails.
            N(Words ∈ spam): Total count of all Words that appeared in spam emails.
            N(Unique Words): Count of all unique words.
            smoothing_factor: Smoothing parameter to avoid zero.
        """
        
        sum_log_prob_word_i_spam = 0
        numerator = self.__count_words_spam + self.__smoothing_factor*self.__count_unique_words

        sum_log_prob_word_i_spam = sum([math.log((self.__word_frequencies.frequency_spam.get(word_i, 0)+self.__smoothing_factor) / numerator) for word_i in prepared_message])

        return sum_log_prob_word_i_spam

    @classmethod
    def set_smoothing_factor(cls, smoothing_factor: float) -> None:
        """
        Set the smoothing factor for the class.

        Parameters:
            smoothing_factor (float): The smoothing factor to be set. Should be in the range 0 < k <= 1.
        
        Returns:
            None
        """

        cls.__smoothing_factor = smoothing_factor



if __name__ == "__main__":
    
    path_to_dataset_word_frequencies = '../dataset/model_data/word_frequencies.csv'
    path_to_dataset_email_ratios = '../dataset/model_data/email_ratios.csv'

    detector = SpamDetector(path_to_dataset_word_frequencies, path_to_dataset_email_ratios)

    # Some examples
    emails = pd.read_csv('../dataset/original_data/spam_ham_dataset.csv')
    
    for i in range(1, 4):
        print(f"Email №{i}\n")
        print(f"{emails.text[i*10]}")
        print(f"\nThe algorithm determined that the email... {detector.detecting_spam(emails.text[i*10])}")
        print(f"It was expected that the email... {emails.label_num[i*10]}")
        print(f"*******************************************************************************\n")