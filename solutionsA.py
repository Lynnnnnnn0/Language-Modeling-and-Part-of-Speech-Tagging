import math
import nltk
import time
from collections import Counter

# Constants to be used by you when you fill the functions
START_SYMBOL = '*'
STOP_SYMBOL = 'STOP'
MINUS_INFINITY_SENTENCE_LOG_PROB = -1000

# TODO: IMPLEMENT THIS FUNCTION
# Calculates unigram, bigram, and trigram probabilities given a training corpus
# training_corpus: is a list of the sentences. Each sentence is a string with tokens separated by spaces, ending in a newline character.
# This function outputs three python dictionaries, where the keys are tuples expressing the ngram and the value is the log probability of that ngram
def calc_probabilities(training_corpus):
    tokens_uni = []
    tokens_bi = []
    tokens_tri = []
    for sentence in training_corpus:
	tokens_bi.append(START_SYMBOL)
	tokens_tri.append(START_SYMBOL)
	tokens_tri.append(START_SYMBOL)
	tokens_uni.extend(sentence.split())
        tokens_bi.extend(sentence.split())
	tokens_tri.extend(sentence.split())
   	tokens_uni.append(STOP_SYMBOL)
	tokens_bi.append(STOP_SYMBOL)
	tokens_tri.append(STOP_SYMBOL)
    unigram_tuples = tokens_uni
    bigram_tuples = list(nltk.bigrams(tokens_bi))
    trigram_tuples = list(nltk.trigrams(tokens_tri))
    bigram_tuples1 = []
    for bigram in bigram_tuples:
        if bigram[1] != START_SYMBOL:
            bigram_tuples1.append(bigram)

    trigram_tuples1 = []
    for trigram in trigram_tuples:
        if trigram[2] != START_SYMBOL:
            trigram_tuples1.append(trigram)

    uni_count = Counter(unigram_tuples)
    bi_count = Counter(bigram_tuples1)
    tri_count = Counter(trigram_tuples1)
    #uni_count = {item : unigram_tuples.count(item) for item in set(unigram_tuples)}
    #bi_count = {item : bigram_tuples1.count(item) for item in set(bigram_tuples1)}
    #tri_count = {item : trigram_tuples1.count(item) for item in set(trigram_tuples1)}
	
    unigram_p = {}
    for unigram in uni_count.keys():
	unigram_p[unigram] =  math.log(float(uni_count[unigram])/len(unigram_tuples),2)
    bigram_p = {}
    for bigram in bi_count.keys():
	if bigram[0] == START_SYMBOL:
	    bigram_p[bigram] = math.log(float(bi_count[bigram])/uni_count[STOP_SYMBOL],2)
	else:
            bigram_p[bigram] = math.log(float(bi_count[bigram])/uni_count[bigram[0]],2)
    trigram_p = {}
    for trigram in tri_count.keys():
	if trigram[0] == START_SYMBOL and trigram[1] == START_SYMBOL:
	    trigram_p[trigram] = math.log(float(tri_count[trigram])/uni_count[STOP_SYMBOL],2)
	else:
            trigram_p[trigram] = math.log(float(tri_count[trigram])/bi_count[(trigram[0],trigram[1])],2)
    return unigram_p, bigram_p, trigram_p
 
# Prints the output for q1
# Each input is a python dictionary where keys are a tuple expressing the ngram, and the value is the log probability of that ngram
def q1_output(unigrams, bigrams, trigrams, filename):
    # output probabilities
    outfile = open(filename, 'w')

    unigrams_keys = unigrams.keys()
    unigrams_keys.sort()
    for unigram in unigrams_keys:
        outfile.write('UNIGRAM ' + unigram + ' ' + str(unigrams[unigram]) + '\n')

    bigrams_keys = bigrams.keys()
    bigrams_keys.sort()
    for bigram in bigrams_keys:
        outfile.write('BIGRAM ' + bigram[0] + ' ' + bigram[1]  + ' ' + str(bigrams[bigram]) + '\n')

    trigrams_keys = trigrams.keys()
    trigrams_keys.sort()    
    for trigram in trigrams_keys:
        outfile.write('TRIGRAM ' + trigram[0] + ' ' + trigram[1] + ' ' + trigram[2] + ' ' + str(trigrams[trigram]) + '\n')

    outfile.close()


# TODO: IMPLEMENT THIS FUNCTION
# Calculates scores (log probabilities) for every sentence
# ngram_p: python dictionary of probabilities of uni-, bi- and trigrams.
# n: size of the ngram you want to use to compute probabilities
# corpus: list of sentences to score. Each sentence is a string with tokens separated by spaces, ending in a newline character.
# This function must return a python list of scores, where the first element is the score of the first sentence, etc. 
def score(ngram_p, n, corpus):
    scores = []
    tokens = []
    tuples = []
    sentences = []
    for sentence in corpus:
        s = []
        for i in range(n-1):
            s.append(START_SYMBOL)
        s.extend(sentence.split())
        s.append(STOP_SYMBOL)
        sentences.append(s)
    
    for sentence in sentences:
        score = 0
        tokens = sentence
        if n == 1:
            tuples = tokens
        elif n == 2:
            tuples = list(nltk.bigrams(tokens))
        else:
            tuples = list(nltk.trigrams(tokens))
        for item in tuples:
            if item in ngram_p:
                score += ngram_p[item]
            else:
                score = MINUS_INFINITY_SENTENCE_LOG_PROB
		break
	scores.append(score)
    return scores

# Outputs a score to a file
# scores: list of scores
# filename: is the output file name
def score_output(scores, filename):
    outfile = open(filename, 'w')
    for score in scores:
        outfile.write(str(score) + '\n')
    outfile.close()

#TODO: IMPLEMENT THIS FUNCTION    
# Calculcates the perplexity of a language model
# scores_file: one of the A2 output files of scores 
# sentences_file: the file of sentences that were scores (in this case: data/Brown_train.txt) 
# This function returns a float, perplexity, the total perplexity of the corpus
def calc_perplexity(scores_file, sentences_file):

    perplexity = 0
    with open(scores_file) as f1:
    	scores = f1.readlines()
    sm = 0
    for line in scores:
	sm += float(line)
    with open(sentences_file) as f2:
    	sentences = f2.readlines()
    num = 0
    for line in sentences: 
        num += len(line.split())+1
    perplexity = 2**(-sm/num)
    return perplexity 

# TODO: IMPLEMENT THIS FUNCTION
# Calculates scores (log probabilities) for every sentence with a linearly interpolated model
# Each ngram argument is a python dictionary where the keys are tuples that express an ngram and the value is the log probability of that ngram
# Like score(), this function returns a python list of scores
def linearscore(unigrams, bigrams, trigrams, corpus):
    scores = []
    lamda = 1.0/3
    for sentence in corpus:
        s_tri = []
        s_tri.append(START_SYMBOL)
        s_tri.append(START_SYMBOL)
        s_tri.extend(sentence.split())
        s_tri.append(STOP_SYMBOL)
        score = 0
        trigram_tuples = list(nltk.trigrams(s_tri))
        for item in trigram_tuples:
	    if item[2] in unigrams:
		score_uni = math.pow(2,unigrams[item[2]])
	    else:
		score_uni = 0
	    if (item[1],item[2]) in bigrams:
		score_bi = math.pow(2,bigrams[(item[1],item[2])])
	    else:
		score_bi = 0
            if item in trigrams:
		score_tri = math.pow(2,trigrams[item])
	    else:
		score_tri = 0
	    if score_uni == 0 and score_bi == 0 and score_tri == 0:
		score = MINUS_INFINITY_SENTENCE_LOG_PROB
		break
	    else:
                score += math.log(lamda*score_uni+lamda*score_bi+lamda*score_tri,2)
	scores.append(score)
    return scores

DATA_PATH = 'data/'
OUTPUT_PATH = 'output/'

# DO NOT MODIFY THE MAIN FUNCTION
def main():
    # start timer
    time.clock()

    # get data
    infile = open(DATA_PATH + 'Brown_train.txt', 'r')
    corpus = infile.readlines()
    infile.close()

    # calculate ngram probabilities (question 1)
    unigrams, bigrams, trigrams = calc_probabilities(corpus)

    # question 1 output
    q1_output(unigrams, bigrams, trigrams, OUTPUT_PATH + 'A1.txt')

    # score sentences (question 2)
    uniscores = score(unigrams, 1, corpus)
    biscores = score(bigrams, 2, corpus)
    triscores = score(trigrams, 3, corpus)

    # question 2 output
    score_output(uniscores, OUTPUT_PATH + 'A2.uni.txt')
    score_output(biscores, OUTPUT_PATH + 'A2.bi.txt')
    score_output(triscores, OUTPUT_PATH + 'A2.tri.txt')

    # linear interpolation (question 3)
    linearscores = linearscore(unigrams, bigrams, trigrams, corpus)

    # question 3 output
    score_output(linearscores, OUTPUT_PATH + 'A3.txt')

    # open Sample1 and Sample2 (question 5)
    infile = open(DATA_PATH + 'Sample1.txt', 'r')
    sample1 = infile.readlines()
    infile.close()
    infile = open(DATA_PATH + 'Sample2.txt', 'r')
    sample2 = infile.readlines()
    infile.close() 

    # score the samples
    sample1scores = linearscore(unigrams, bigrams, trigrams, sample1)
    sample2scores = linearscore(unigrams, bigrams, trigrams, sample2)

    # question 5 output
    score_output(sample1scores, OUTPUT_PATH + 'Sample1_scored.txt')
    score_output(sample2scores, OUTPUT_PATH + 'Sample2_scored.txt')

    # print total time to run Part A
    print "Part A time: " + str(time.clock()) + ' sec'

if __name__ == "__main__": main()
