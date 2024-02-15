import math

import pandas as pd

import TextProcessor


class SpamDetector:
    """
    A class for detecting spam in email messages using a Multinomial Naive Bayes classifier.

    Class Attributes:
        __processor (TextProcessor): An instance of TextProcessor is used for text processing and tokenization.
            Default minimum word length is 2. Should be greater than or equal to 0.
            Default includes stop words from the nltk and wordcloud libraries.
        __smoothing_factor (float): The smoothing factor.
            When equal to 1, corresponds to Laplace smoothing.
            Default is 1.0. Should be in the range 0 < __smoothing_factor <= 1.

    Attributes:
        __word_frequencies (pd.DataFrame): The DataFrame of word frequencies.
        __count_unique_words (int): The count of unique words; sourced from DataFrame word frequencies.
        __count_words_ham (int): The count of words in ham emails; sourced from DataFrame word frequencies.
        __count_words_spam (int): The count of words in spam emails; sourced from DataFrame word frequencies.
        __ham_probability (float): The probability of ham emails; sourced from DataFrame email_ratios.
        __spam_probability (float): The probability of spam emails; sourced from DataFrame email_ratios.

    Note:
        Smoothing is used to handle the issue of zero probabilities in the Multinomial Naive Bayes classifier.
    """

    __processor = TextProcessor.TextProcessor()
    __smoothing_factor = 1.0

    def __init__(self, path_to_dataset_word_frequencies: str, path_to_dataset_email_ratios: str) -> None:
        """
        Initializes a SpamDetector object.

        Parameters:
            path_to_dataset_word_frequencies (str): The file path to the DataFrame file of word frequencies.
                The DataFrame must have the following structure:
                    Columns should be labeled as follows: frequency, frequency_ham, frequency_spam.
                    Indexes should be made up of words.
                Example:
                    ,frequency,frequency_ham,frequency_spam
                    deal,3655,3549,106
                    pleas,3243,2737,506
                    ga,3034,2861,173
                    meter,2721,2718,3
                    thank,2304,2125,179

            path_to_dataset_email_ratios (str): The file path to the DataFrame file containing the ratios of ham and spam emails to total emails.
                The DataFrame must have the following structure:
                    Columns should be labeled as follows: ham, spam.
                    Index should be only one and named: ratios-to-total-emails.
                Example:
                    ,ham,spam
                    ratios-to-total-emails,0.7127329192546584,0.28726708074534163

        Raises:
            FileNotFoundError: If the specified DataFrame file of word frequencies or email relationships is not found.
            AttributeError: If the DataFrame file of word frequencies or email ratios has an incorrect structure.

        Returns:
            None
        """

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
        Detects spam in a message.
        
        Parameters:
            message (str): The string containing the message text.
        
        Returns:
            int: Returns 1 if the message is considered spam, and 0 if it is ham.

        Formula:
                                                 n
            (№1) P(ham | Word_i) = log(P(ham)) + ∑ log(P(Word_i | ham))
                                                i=1
                
                                                   n
            (№2) P(spam | Word_i) = log(P(spam)) + ∑ log(P(Word_i | spam))
                                                  i=1

        Where:
            log(P(ham)): The logarithm of the probability of receiving ham email.
                          Calculated using the method "__log_probability_ham_email".

            log(P(spam)): The logarithm of the probability of receiving spam email.
                           Calculated using the method "__log_probability_spam_email".

             n
             ∑ log(P(Word_i | ham)): Sum of log probabilities of words in the ham category from the given message.
            i=1                       Calculated using the method "__sum_log_prob_words_ham".
            
             n
             ∑ log(P(Word_i | spam)): Sum of log probabilities of words in the spam category from the given message.
            i=1                        Calculated using the method "__sum_log_prob_words_spam".
        """

        prepared_message = self.__processor.pipeline(message)

        ham_prob_given_word_i = self.__log_probability_ham_email(self.__ham_probability) + self.__sum_log_prob_words_ham(prepared_message)
        spam_prob_given_word_i = self.__log_probability_spam_email(self.__spam_probability) + self.__sum_log_prob_words_spam(prepared_message)

        return int(spam_prob_given_word_i > ham_prob_given_word_i)
    
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
            n                        n             N(Word_i ∈ ham) + smoothing_factor
            ∑ log(P(Word_i | ham)) = ∑ log(--------------------------------------------------)
           i=1                      i=1     N(Words ∈ ham) + smoothing_factor*N(Unique Words)

        Where:
            N(Word_i ∈ ham): Count of occurrences of the Word_i in ham emails.
            N(Words ∈ ham): Total count of all Words that appeared in ham emails.
            N(Unique Words): Count of all unique words.
            smoothing_factor: Smoothing factor to avoid zero.
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
            n                         n             N(Word_i ∈ spam) + smoothing_factor
            ∑ log(P(Word_i | spam)) = ∑ log(---------------------------------------------------)
           i=1                       i=1     N(Words ∈ spam) + smoothing_factor*N(Unique Words)

        Where:
            N(Word_i ∈ spam): Count of occurrences of the Word_i in spam emails.
            N(Words ∈ spam): Total count of all Words that appeared in spam emails.
            N(Unique Words): Count of all unique words.
            smoothing_factor: Smoothing factor to avoid zero.
        """

        sum_log_prob_word_i_spam = 0
        numerator = self.__count_words_spam + self.__smoothing_factor*self.__count_unique_words

        sum_log_prob_word_i_spam = sum([math.log((self.__word_frequencies.frequency_spam.get(word_i, 0)+self.__smoothing_factor) / numerator) for word_i in prepared_message])

        return sum_log_prob_word_i_spam

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
    def set_smoothing_factor(cls, smoothing_factor: float) -> None:
        """
        Set the smoothing factor for the class.

        Parameters:
            smoothing_factor (float): The smoothing factor to be set. Should be in the range 0 < smoothing_factor <= 1.
        
        Raises:
            ValueError: If the smoothing_factor is not in the valid range.

        Returns:
            None
        """

        if 0 < smoothing_factor and smoothing_factor <= 1:
            cls.__smoothing_factor = smoothing_factor
        else:
            raise ValueError(f"Error: Smoothing factor should be in the range 0 < smoothing_factor <= 1. Got: {smoothing_factor}")


if __name__ == "__main__":

    path_to_dataset_word_frequencies = '../dataset/email_dataset_1/model_data/word_frequencies.csv'
    path_to_dataset_email_ratios = '../dataset/email_dataset_1/model_data/email_ratios.csv'

    detector = SpamDetector(path_to_dataset_word_frequencies, path_to_dataset_email_ratios)
    detector.set_processor(user_stop_words=['ect', 'enron', 'hou', 'hpl', 'subject'])

    # Some examples
    emails = pd.read_csv('../dataset/email_dataset_1/spam_ham_dataset.csv')
    for i in range(1, 4):
        print(f"Email №{i}\n")
        print(f"{emails.text[i*10]}")
        print(f"\nThe algorithm determined that the email... {detector.detecting_spam(emails.text[i*10])}")
        print(f"It was expected that the email... {emails.label_num[i*10]}")
        print(f"*******************************************************************************\n")