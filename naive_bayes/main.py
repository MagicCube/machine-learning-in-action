import numpy as np

from mail import *
from model import *

def main():
    # Load all samples
    samples = __load_samples()
    # Split them into training and validation samples
    training_samples, validation_samples = __split_samples(samples, 20, 20)
    # Training
    model = NaiveBayesModel()
    model.train(training_samples)
    # Validating
    error_count = 0
    for spam_mail in validation_samples[0]:
        spam = model.is_spam(spam_mail)
        if not spam:
            error_count += 1
    for ham_mail in validation_samples[1]:
        spam = model.is_spam(ham_mail)
        if spam:
            error_count += 1
    validation_count = (len(validation_samples[0]) + len(validation_samples[1]))
    print("Error rate: %d%%, %d out of %d" % (error_count / validation_count * 100, error_count, validation_count))


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
