import pickle

import numpy as np

class NaiveBayesModel:
    def __init__(self):
        self.__dict = []




    @property
    def sample_count(self):
        return self.__sample_count


    @property
    def spam_probability(self):
        return self.__spam_probability


    @property
    def ham_probability(self):
        return self.__ham_probability


    @property
    def spam_word_probabilities(self):
        return self.__spam_word_probabilities


    @property
    def ham_word_probabilities(self):
        return self.__ham_word_probabilities


    @property
    def dict(self):
        return self.__dict




    def train(self, samples):
        # Update the dict first
        self.__train_dict(samples)
        # Seperate
        spam_mails, ham_mails = samples
        # Basic statistics
        self.__sample_count = len(spam_mails) + len(ham_mails)
        self.__spam_probability = len(spam_mails) / self.sample_count
        self.__ham_probability = 1 - self.spam_probability
        # Counting words
        self.__spam_word_count = 0
        self.__ham_word_count = 0
        for mail in spam_mails:
            self.__spam_word_count += len(mail.words)
        for mail in ham_mails:
            self.__ham_word_count += len(mail.words)
        # Train
        self.__train(spam_mails, ham_mails)



    def save(self):
        with open("model.dmp", "wb") as fw:
            pickle.dump(self, fw)


    @staticmethod
    def load():
        with open("model.dmp", "rb") as fr:
            return pickle.load(fr)


    def is_spam(self, mail):
        vector = self.__convert_into_vector(mail)
        p_spam = 1
        p_ham = 1
        for i, value in enumerate(vector):
            if value > 0:
                p_spam *= self.spam_word_probabilities[i] * (value / len(vector))
                p_ham *= self.ham_word_probabilities[i] * (value / len(vector))
        return p_spam * self.spam_probability > p_ham * self.ham_probability


    def __train_dict(self, samples):
        for sample in samples:
            self.__extend_dict(sample)


    def __extend_dict(self, mails):
        for mail in mails:
            for word in mail.words:
                if (word not in self.dict):
                    self.dict.append(word)


    def __train(self, spam_mails, ham_mails):
        spam_vectors = self.__convert_into_vectors(spam_mails)
        ham_vectors = self.__convert_into_vectors(ham_mails)
        self.__spam_word_probabilities = np.sum(spam_vectors, axis=0) / self.__spam_word_count
        self.__ham_word_probabilities = np.sum(ham_vectors, axis=0) / self.__ham_word_count



    def __convert_into_vectors(self, mails):
        vectors = []
        for mail in mails:
            vector = self.__convert_into_vector(mail)
            vectors.append(vector)
        return np.array(vectors)


    def __convert_into_vector(self, mail):
        vector = np.zeros((len(self.dict,)))
        for word in mail.words:
            try:
                index = self.dict.index(word)
                vector[index] += 1
            except ValueError:
                continue
        return vector
