import json
from itertools import chain

from misc import indices_to_one_hot_encodings, wd


treebank_file1 = open(wd + '/treebank/OPTA-treebank-0.1.json')
treebank_file2 = open(wd + '/treebank/skladnica_output.json')
treebank = chain(list(json.load(treebank_file1)), list(json.load(treebank_file2)))

data = []
target = []
for entry in treebank:
    tree = entry['parsedSent']
    words = []
    sentiment = None
    for index, node in enumerate(tree):
        word = node.split('\t')[1].lower()
        words.append(word)
        if node.split('\t')[10] == 'S':
            sentiment = index
    if sentiment:
        data.append(words)
        target.append(indices_to_one_hot_encodings(sentiment, len(words)))
