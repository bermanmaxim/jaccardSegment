from __future__ import print_function, division
from random import Random
from datasets.loadvoc import load_extended_voc

random = Random(1234)

train, val, test = load_extended_voc()

# create balanced binary datasets for experimenting

balanced = {}

for c in sorted(train.classes, key=train.classes.get):
    if c != 'void':
        pos = []
        neg = []
        for i, ex in enumerate(val):
            if c in ex.classes:
                pos.append(i)
            else:
                neg.append(i)
        random.shuffle(neg)
        balanced[c] = pos + neg[:len(pos)]



