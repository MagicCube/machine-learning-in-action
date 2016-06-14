import re

def extract_words(text):
    reg_exp = re.compile(r'([a-zA-Z]+)')
    words = list(map(lambda word: word.lower(), reg_exp.findall(text)))
    words = [ word for word in words if len(word) > 3 ]
    return words
