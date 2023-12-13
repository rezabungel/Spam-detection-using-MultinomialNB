import string

import nltk
import wordcloud
from nltk.stem.porter import PorterStemmer # We have various stemmers; let's utilize them.


class TextProcessor:
    """
    Text processing pipeline class

    Attributes:
        __len_min_ch (int): Minimum length of words. Default is 2. Should be greater than 0.
        __stop_words (set[str]): Set of stop words. Set of stop words. Default includes stop words from nltk and wordcloud libraries, as well as additional words.
    """

    def __init__(self, min_length: int = None, user_stop_words: list[str] | tuple[str] | set[str] = None) -> None:
        """
        Initializes the TextProcessor object.

        Parameters:
            min_length (int, optional): Minimum word length. If not specified, the default is 2. Should be greater than 0.
            user_stop_words (list[str] | tuple[str] | set[str], optional): User-defined list of stop words. 
                If not specified, only stop words from the nltk and wordcloud libraries, as well as additional words, will be used.
                If specified, it will be added to the previously declared set of stop words.

        Returns:
            None
        """

        if min_length is None:
            self.__len_min_ch = 2
        else:
            self.__len_min_ch = min_length

        stop_words_n1 = set(nltk.corpus.stopwords.words('english'))
        stop_words_n2 = set(wordcloud.STOPWORDS)
        stop_words_n3 = set(['ect', 'enron', 'hou', 'hpl', 'subject'])

        if user_stop_words is None:
            self.__stop_words = set.union(stop_words_n1, stop_words_n2, stop_words_n3)
        else:
            user_stop_words = set(user_stop_words)
            self.__stop_words = set.union(stop_words_n1, stop_words_n2, stop_words_n3, user_stop_words)

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

    processor = TextProcessor()

    raw_text = "This is a sample text for processing."
    processed_text = processor.pipeline(raw_text)

    print(processed_text)