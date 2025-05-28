from abc import ABC, abstractmethod

class BaseRetriever(ABC):
    @abstractmethod
    def retrieve(self, query):
        """
        Abstract method to retrieve documents based on the query.
        """
        pass

    @abstractmethod
    def build_corpus(self, corpus_path):
        """
        Abstract method to build the corpus from the provided data.
        """
        pass