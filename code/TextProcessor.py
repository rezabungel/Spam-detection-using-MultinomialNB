import string

import nltk
import wordcloud
from nltk.stem.porter import PorterStemmer # We have various stemmers; let's utilize them.


class TextProcessor:
    """
    Text processing pipeline class.

    Attributes:
        __len_min_ch (int): Minimum word length.
            Default is 2. Should be greater than or equal to 0.
        __stop_words (set[str]): Set of stop words.
            Default includes stop words from the nltk and wordcloud libraries.
    """

    def __init__(self, min_length: int = None, user_stop_words: list[str] | tuple[str] | set[str] = None) -> None:
        """
        Initializes the TextProcessor object.

        Parameters:
            min_length (int, optional): Minimum word length.
                Default is 2. Should be greater than or equal to 0.
            user_stop_words (list[str] | tuple[str] | set[str], optional): The user-defined list of stop words. This list is added to the stop words defined by default.
                Default includes stop words from the nltk and wordcloud libraries.
                Example:
                    user_stop_words = ['ect', 'enron', 'hou', 'hpl', 'subject']

        Raises:
            ValueError: If the provided min_length is less than 0.
            ValueError: If the provided user_stop_words is not a list, tuple, or set.
            ValueError: If the provided user_stop_words contains anything other than strings.

        Returns:
            None
        """

        if min_length is None:
            self.__len_min_ch = 2
        else:
            if min_length >= 0:
                self.__len_min_ch = min_length
            else:
                raise ValueError(f"Error: min_length should be greater than or equal to 0. Got: {min_length}")

        stop_words_n1 = set(nltk.corpus.stopwords.words('english'))
        stop_words_n2 = set(wordcloud.STOPWORDS)

        if user_stop_words is None:
            self.__stop_words = set.union(stop_words_n1, stop_words_n2)
        else:
            if isinstance(user_stop_words, (list, tuple, set)):
                if all(isinstance(word, str) for word in user_stop_words):
                    user_stop_words = set(user_stop_words)
                    self.__stop_words = set.union(stop_words_n1, stop_words_n2, user_stop_words)
                else:
                    raise ValueError("Error: user_stop_words should contain only strings.")
            else:
                raise ValueError(f"Error: user_stop_words should be a list, tuple, or set. Got {type(user_stop_words)}")

    def pipeline(self, raw_text: str) -> list[str]:
        """
        Processes the text through predefined processing stages.

        Parameters:
            raw_text (str): The raw input text.

        Returns:
            list[str]: The list of processed words.
        """

        prepared_text = self.__remove_NonASCII(raw_text)
        prepared_text = self.__lowercase_and_remove_short_words(prepared_text)
        prepared_text = self.__remove_stopwords(prepared_text)
        prepared_text = self.__stemming(prepared_text)
        prepared_text = self.__tokenize(prepared_text)

        return prepared_text

    def __remove_NonASCII(self, text: str) -> str:
        """
        Removes all non-ASCII characters from the text.

        Parameters:
            text (str): The input text.

        Returns:
            str: The processed text.
        """

        for ch in text:
            if not ch in string.ascii_letters:
                text = text.replace(ch, ' ')

        return text

    def __lowercase_and_remove_short_words(self, text: str) -> str:
        """
        Converts the text to lowercase and removes short words.

        Parameters:
            text (str): The input text.

        Returns:
            str: The processed text.
        """

        return ' '.join([word for word in text.split() if len(word) > self.__len_min_ch]).lower()

    def __remove_stopwords(self, text: str) -> str:
        """
        Removes stop words from the text.

        Parameters:
            text (str): The input text.

        Returns:
            str: The processed text.
        """

        return ' '.join([word for word in text.split() if word not in self.__stop_words])

    def __stemming(self, text: str) -> str:
        """
        Performs stemming on words in the text.

        Parameters:
            text (str): The input text.

        Returns:
            str: The processed text.
        """

        stemmer = PorterStemmer()
        return ' '.join([stemmer.stem(word) for word in text.split()])

    def __tokenize(self, text: str) -> list[str]:
        """
        Tokenizes the text.

        Parameters:
            text (str): The input text.

        Returns:
            list[str]: The list of tokens.
        """

        return text.split()


if __name__ == "__main__":

    processor = TextProcessor(min_length=1, user_stop_words=('ect', 'enron', 'hou', 'hpl', 'subject'))

    raw_text = "This is a sample text for processing."
    processed_text = processor.pipeline(raw_text)

    print(processed_text)