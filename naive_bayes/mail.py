import os
import ta

class Mail:
    def __init__(self, subject, is_spam):
        self.__subject = subject
        self.__content = []
        self.__words = []
        self.__is_spam = is_spam
        pass


    @classmethod
    def load_from_file(cls, path):
        mail = cls(path, "/spam/" in path)
        lines = None
        abs_path = "%s/%s" % (os.path.abspath(os.path.dirname(__file__)), path)
        with open(abs_path, "r") as fr:
            lines = fr.readlines()
        for line in lines:
            line = line.strip()
            mail.content.append(line)
            mail.words.extend(ta.extract_words(line))
        return mail


    @property
    def subject(self):
        return self.__subject


    @property
    def content(self):
        return self.__content


    @property
    def words(self):
        return self.__words


    @property
    def is_spam(self):
        return self.__is_spam


    def __str__(self):
        return self.subject
