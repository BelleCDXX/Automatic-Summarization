import sys
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from string import punctuation
from math import log2

PUNCT = set(punctuation)
STOPWORDS = set(stopwords.words('english'))
RATIO = 0.25
WORD_MAX = 500
DIFF = 1.0e-6
DELTA = 0.85

# get title of the doc

title = sys.stdin.readline().strip()

# get content of the doc
content = sys.stdin.read().strip()

# split into sentences
sentences = sent_tokenize(content)

# number of all the sentences
length = len(sentences)

# tokenlize each sentence and delete punctuation and stopwords
# use this to count weight of each sentence
# count the total number of the words
words = []
num = 0
for sent in sentences:
    word_token = word_tokenize(sent)
    num += len(word_token)
    words.append((set(word for word in word_token if word not in PUNCT and word.lower() not in STOPWORDS), len(word_token)))


# choose the less one between RATIO of the num and WORD_MAX
Max_words = min(RATIO * num, WORD_MAX)

# measure the similarity between each pair of the sentence
# store the measurement in a matrix
sent_similarity = [[0] * length for _ in range(length)]

for i in range(length):
    for j in range(i + 1, length):
        intersect = len(words[i][0] & words[j][0]) / (1 + log2(len(words[i][0])) + log2(len(words[j][0])))
        sent_similarity[i][j] = sent_similarity[j][i] = intersect
        
# normalize the matrix as in pagerank
base = [sum(line) for line in sent_similarity]
for i in range(length):
    for j in range(length):
        if base[j]:
            sent_similarity[i][j] /= base[j]


# use pagerank algorathm generate rank of sentences
def pagerank(matrix, delta = DELTA, diff = DIFF):
    n = len(matrix)
    cur = [1/n] * n
    pre = [0] * n
    while sum(abs(pre[i] - cur[i]) for i in range(n)) > diff:
        pre, cur = cur, pre
        for i in range(n):
            cur[i] = delta * sum(matrix[i][j] * pre[j] for j in range(n)) + 1 - delta
    return cur

rank = pagerank(sent_similarity)

# sort the index of sentences by weight
sorted_weight_sent = sorted(list(range(len(sentences))), key = lambda x: rank[x], reverse = True)

# decide the end of the grammar by Max_words
Sum, end = 0, -1
while Sum < Max_words and end < length - 1:
    end += 1
    Sum += words[sorted_weight_sent[end]][1]

# sort the most weight sentences as original order
# generate summary
summary = [sentences[i] for i in sorted(sorted_weight_sent[:end])]

print('Summary:\n')
print(' '.join(summary))









