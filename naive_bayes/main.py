import numpy as np

from mail import *
from model import *

def main():
    # Loading all samples
    samples = __load_samples()
    # Split them into training and validation samples
    training_samples, validation_samples = __split_samples(samples, 5, 21)
    # Training
    model = NaiveBayesModel()
    model.train(training_samples)
    pass


def __load_samples():
    ham_mails = []
    spam_mails = []
    for i in range(25):
        mail = Mail.load_from_file("data/ham/%d.txt" % (i + 1))
        ham_mails.append(mail)
        mail = Mail.load_from_file("data/spam/%d.txt" % (i + 1))
        spam_mails.append(mail)
    return (spam_mails, ham_mails)


def __split_samples(samples, training_spam_count, tranining_ham_count):
    spam_mails, ham_mails = samples
    training_samples = ([], [])
    for i in range(training_spam_count):
        index = int(np.random.uniform(0, len(spam_mails)))
        mail = spam_mails[index]
        training_samples[0].append(mail)
        spam_mails.remove(mail)
    for i in range(tranining_ham_count):
        index = int(np.random.uniform(0, len(ham_mails)))
        mail = ham_mails[index]
        training_samples[1].append(mail)
        ham_mails.remove(mail)
    validation_samples = (spam_mails, ham_mails)
    return (training_samples, validation_samples)

if __name__ == "__main__":
    main()
