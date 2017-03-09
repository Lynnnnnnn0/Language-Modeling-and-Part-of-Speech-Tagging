import sys
import nltk
import math
import time
from collections import Counter
#import numpy as np


START_SYMBOL = '*'
STOP_SYMBOL = 'STOP'
RARE_SYMBOL = '_RARE_'
RARE_WORD_MAX_FREQ = 5
LOG_PROB_OF_ZERO = -1000


# TODO: IMPLEMENT THIS FUNCTION
# Receives a list of tagged sentences and processes each sentence to generate a list of words and a list of tags.
# Each sentence is a string of space separated "WORD/TAG" tokens, with a newline character in the end.
# Remember to include start and stop symbols in yout returned lists, as defined by the constants START_SYMBOL and STOP_SYMBOL.
# brown_words (the list of words) should be a list where every element is a list of the tags of a particular sentence.
# brown_tags (the list of tags) should be a list where every element is a list of the tags of a particular sentence.
def split_wordtags(brown_train):
    brown_words = []
    brown_tags = []
    for sentence in brown_train:
        words = []
	tags = []
	tokens = sentence.split()
        words.append(START_SYMBOL)
        tags.append(START_SYMBOL)
	words.append(START_SYMBOL)
        tags.append(START_SYMBOL)
        for item in tokens:
            ind = item.rfind('/')
            words.append(item[:ind])
            tags.append(item[ind+1:])
	words.append(STOP_SYMBOL)
	tags.append(STOP_SYMBOL)
	brown_words.append(words)
	brown_tags.append(tags)
    return brown_words, brown_tags


# TODO: IMPLEMENT THIS FUNCTION
# This function takes tags from the training data and calculates tag trigram probabilities.
# It returns a python dictionary where the keys are tuples that represent the tag trigram, and the values are the log probability of that trigram
def calc_trigrams(brown_tags):
    q_values = {}
    tags = []
    for sentence in brown_tags:
	tags.extend(sentence)
    bigram_tuples = list(nltk.bigrams(tags))
    trigram_tuples = list(nltk.trigrams(tags))
    bi_count = Counter(bigram_tuples)
    tri_count = Counter(trigram_tuples)
    for item in trigram_tuples:
        q_values[item] = math.log(float(tri_count[item])/bi_count[(item[0],item[1])],2)
    return q_values

# This function takes output from calc_trigrams() and outputs it in the proper format
def q2_output(q_values, filename):
    outfile = open(filename, "w")
    trigrams = q_values.keys()
    trigrams.sort()  
    for trigram in trigrams:
        output = " ".join(['TRIGRAM', trigram[0], trigram[1], trigram[2], str(q_values[trigram])])
        outfile.write(output + '\n')
    outfile.close()


# TODO: IMPLEMENT THIS FUNCTION
# Takes the words from the training data and returns a set of all of the words that occur more than 5 times (use RARE_WORD_MAX_FREQ)
# brown_words is a python list where every element is a python list of the words of a particular sentence.
# Note: words that appear exactly 5 times should be considered rare!
def calc_known(brown_words):
    known_words = set([])
    words = []
    for sentence in brown_words:
	words.extend(sentence)
    count = Counter(words)
    for word in count.keys():
        if count[word] > RARE_WORD_MAX_FREQ:
            known_words.add(word)
    return known_words

# TODO: IMPLEMENT THIS FUNCTION
# Takes the words from the training data and a set of words that should not be replaced for '_RARE_'
# Returns the equivalent to brown_words but replacing the unknown words by '_RARE_' (use RARE_SYMBOL constant)
def replace_rare(brown_words, known_words):
    brown_words_rare = []
    for sentence in brown_words:
	sen = []
        for word in sentence:
       	    if word not in known_words:
    	   	sen.append(RARE_SYMBOL)
            else:
            	sen.append(word)
	brown_words_rare.append(sen)
    return brown_words_rare

# This function takes the ouput from replace_rare and outputs it to a file
def q3_output(rare, filename):
    outfile = open(filename, 'w')
    for sentence in rare:
        outfile.write(' '.join(sentence[2:-1]) + '\n')
    outfile.close()


# TODO: IMPLEMENT THIS FUNCTION
# Calculates emission probabilities and creates a set of all possible tags
# The first return value is a python dictionary where each key is a tuple in which the first element is a word
# and the second is a tag, and the value is the log probability of the emission of the word given the tag
# The second return value is a set of all possible tags for this data set
def calc_emission(brown_words_rare, brown_tags):
    e_values = {}
    taglist = set([])
    tags = []
    for sentence in brown_tags:
        tags.extend(sentence)
    count = Counter(tags)
    taglist = count.keys()
    count_tuple = {}
    for i in range(len(brown_words_rare)):
        sentence = brown_words_rare[i]
        for j in range(len(sentence)):
            count_tuple[(sentence[j], brown_tags[i][j])] = count_tuple.get((sentence[j], brown_tags[i][j]), 0) + 1
    for tuples in count_tuple.keys():
        e_values[tuples] = math.log(float(count_tuple[tuples])/count[tuples[1]],2)
    return e_values, taglist

# This function takes the output from calc_emissions() and outputs it
def q4_output(e_values, filename):
    outfile = open(filename, "w")
    emissions = e_values.keys()
    emissions.sort()  
    for item in emissions:
        output = " ".join([item[0], item[1], str(e_values[item])])
        outfile.write(output + '\n')
    outfile.close()

def forward(brown_dev_words,taglist, known_words, q_values, e_values):
    probs = []
    N = len(taglist)
        
    for words in brown_dev_words:
        words = replace_rare([words],known_words)[0]
        words.insert(0, START_SYMBOL)
        words.insert(0, START_SYMBOL)
        words.append(STOP_SYMBOL)
        T = len(words)
	
	# Initialization
	pi = {}
    	pi[(1,START_SYMBOL,START_SYMBOL)] = 1
        # Recursion
        for t in range(2,T-1):
	    # the third word            
            for i in taglist:
		if (words[t],i) in e_values:
                        emission = e_values[(words[t],i)]
                else:
                    continue
		# the second word		
		for j in taglist:
                    sm = 0.0
		    # the first word
                    for k in taglist:
			if (k,j,i) in q_values:
			    trans = q_values[(k,j,i)]
			else:
			    continue
			if (t-1,k,j) in pi:
                            pre_pi = pi[(t-1,k,j)]
			else:
			    continue
                        sm += math.pow(2,emission+trans)*pre_pi
            	    pi[(t,j,i)] = sm
	t = T-1
	if (words[t],STOP_SYMBOL) in e_values:
	    emission = e_values[(words[t],STOP_SYMBOL)]
	for j in taglist:
            sm = 0.0
            # the first word
            for k in taglist:
                if (k,j,STOP_SYMBOL) in q_values:
                    trans = q_values[(k,j,STOP_SYMBOL)]
                else:
                    continue
                if (t-1,k,j) in pi:
                    pre_pi = pi[(t-1,k,j)]
                else:
                    continue
                sm += math.pow(2,emission+trans)*pre_pi 
            pi[(t,j,STOP_SYMBOL)] = sm
       
        # Termination
        prob = 0
        for j in taglist:
            prob += pi[(T-1,j,STOP_SYMBOL)] #+q_values.get((key[1],key[2],STOP_SYMBOL),LOG_PROB_OF_ZERO))
        if prob > 0:
	    prob = math.log(prob,2)
	else:
	    prob = LOG_PROB_OF_ZERO
	#print(prob)
	probs.append(str(prob)+'\n')
    return probs


# This function takes the output of forward() and outputs it to file
def q5_output(tagged, filename):
    outfile = open(filename, 'w')
    for sentence in tagged:
        outfile.write(sentence)
    outfile.close()


# TODO: IMPLEMENT THIS FUNCTION
# This function takes data to tag (brown_dev_words), a set of all possible tags (taglist), a set of all known words (known_words),
# trigram probabilities (q_values) and emission probabilities (e_values) and outputs a list where every element is a tagged sentence 
# (in the WORD/TAG format, separated by spaces and with a newline in the end, just like our input tagged data)
# brown_dev_words is a python list where every element is a python list of the words of a particular sentence.
# taglist is a set of all possible tags
# known_words is a set of all known words
# q_values is from the return of calc_trigrams()
# e_values is from the return of calc_emissions()
# The return value is a list of tagged sentences in the format "WORD/TAG", separated by spaces. Each sentence is a string with a 
# terminal newline, not a list of tokens. Remember also that the output should not contain the "_RARE_" symbol, but rather the
# original words of the sentence!
def viterbi(brown_dev_words, taglist, known_words, q_values, e_values):
    tagged = []
    N = len(taglist)

    for words in brown_dev_words:
	words_original = words
        words = replace_rare([words],known_words)[0]
        words.insert(0, START_SYMBOL)
        words.insert(0, START_SYMBOL)
        T = len(words)
        pi = {}
        bp = {}
        tags = {}        
        pi[(1, START_SYMBOL,START_SYMBOL)] = 1

        for t in range(2,T):
            for i in taglist:
                if (words[t],i) in e_values:
                    emission = e_values[(words[t],i)]
                else:
                    continue
                for j in taglist:
                    for k in taglist:
                        if (t-1,k,j) in pi:
                            pi_pre = pi[(t-1,k,j)]
                        else:
                            continue
                        if (k,j,i) in q_values:
                            trans = q_values[(k,j,i)]
                        else:
                            trans = LOG_PROB_OF_ZERO
                        if (t,j,i) not in pi:
                            pi[(t,j,i)] = pi_pre + emission + trans
                            bp[(t,j,i)] = k
                        else:
                            if pi_pre+emission+trans > pi[(t,j,i)]:
                                pi[(t,j,i)] = pi_pre + emission + trans
                                bp[(t,j,i)] = k
		
        max_prob = -10000
        boo = True
        t = T-1
        for i in taglist:
            for j in taglist:
                if (t,j,i) not in pi:
                    continue
                if (j,i,STOP_SYMBOL) in q_values:
                    emission = q_values[(j,i,STOP_SYMBOL)]
                else:
                    emission = LOG_PROB_OF_ZERO
                if boo:
                    tags[t] = i
                    tags[t-1] = j
                    max_prob = pi[(t,j,i)] + emission
                    boo = False
                else:
                    if pi[(t,j,i)] + emission > max_prob:
                        max_prob = pi[(t,j,i)] + emission
                        tags[t] = i
                        tags[t-1] = j
                                                
        t = T-3
        while t >= 0:
            tags[t] = bp[(t+2, tags[t+1], tags[t+2])]
            t -= 1    
	
        sentence = ""
        for l in range(T-2):
            sentence += words_original[l] + "/" + tags[l+2] + " "
        sentence = sentence.strip() + "\n"
        tagged.append(sentence)
    return tagged

# This function takes the output of viterbi() and outputs it to file
def q6_output(tagged, filename):
    outfile = open(filename, 'w')
    for sentence in tagged:
        outfile.write(sentence)
    outfile.close()

# TODO: IMPLEMENT THIS FUNCTION
# This function uses nltk to create the taggers described in question 6
# brown_words and brown_tags is the data to be used in training
# brown_dev_words is the data that should be tagged
# The return value is a list of tagged sentences in the format "WORD/TAG", separated by spaces. Each sentence is a string with a 
# terminal newline, not a list of tokens. 
def nltk_tagger(brown_words, brown_tags, brown_dev_words):
    # Hint: use the following line to format data to what NLTK expects for training
    training = [ zip(brown_words[i],brown_tags[i]) for i in xrange(len(brown_words)) ]

    # IMPLEMENT THE REST OF THE FUNCTION HERE
    tagged = []
    default_tagger = nltk.DefaultTagger('NOUN')
    bigram_tagger = nltk.BigramTagger(training, backoff=default_tagger)
    trigram_tagger = nltk.TrigramTagger(training, backoff=bigram_tagger)

    for words in brown_dev_words:
        tagged_tuples = trigram_tagger.tag(words)
        sentence = ""
        for tuples in tagged_tuples:
            sentence += tuples[0] + "/" + tuples[1] + " "
        sentence = sentence.strip() + '\n'
        tagged.append(sentence)
    return tagged

# This function takes the output of nltk_tagger() and outputs it to file
def q7_output(tagged, filename):
    outfile = open(filename, 'w')
    for sentence in tagged:
        outfile.write(sentence)
    outfile.close()

DATA_PATH = 'data/'
OUTPUT_PATH = 'output/'

def main():
    # start timer
    time.clock()

    # open Brown training data
    infile = open(DATA_PATH + "Brown_tagged_train.txt", "r")
    brown_train = infile.readlines()
    infile.close()

    # split words and tags, and add start and stop symbols (question 1)
    brown_words, brown_tags = split_wordtags(brown_train)

    # calculate tag trigram probabilities (question 2)
    q_values = calc_trigrams(brown_tags)

    # question 2 output
    q2_output(q_values, OUTPUT_PATH + 'B2.txt')

    # calculate list of words with count > 5 (question 3)
    known_words = calc_known(brown_words)

    # get a version of brown_words with rare words replace with '_RARE_' (question 3)
    brown_words_rare = replace_rare(brown_words, known_words)

    # question 3 output
    q3_output(brown_words_rare, OUTPUT_PATH + "B3.txt")

    # calculate emission probabilities (question 4)
    e_values, taglist = calc_emission(brown_words_rare, brown_tags)

    # question 4 output
    q4_output(e_values, OUTPUT_PATH + "B4.txt")

    # delete unneceessary data
    del brown_train
    del brown_words_rare



    # open Brown development data (question 6)
    infile = open(DATA_PATH + "Brown_dev.txt", "r")
    brown_dev = infile.readlines()
    infile.close()

    # format Brown development data here
    brown_dev_words = []
    for sentence in brown_dev:
        brown_dev_words.append(sentence.split(" ")[:-1])

    # question 5
    forward_probs = forward(brown_dev_words,taglist, known_words, q_values, e_values)
    q5_output(forward_probs, OUTPUT_PATH + 'B5.txt')

    # do viterbi on brown_dev_words (question 6)
    viterbi_tagged = viterbi(brown_dev_words, taglist, known_words, q_values, e_values)

    # question 6 output
    q6_output(viterbi_tagged, OUTPUT_PATH + 'B6.txt')

    # do nltk tagging here
    nltk_tagged = nltk_tagger(brown_words, brown_tags, brown_dev_words)

    # question 7 output
    q7_output(nltk_tagged, OUTPUT_PATH + 'B7.txt')

    # print total time to run Part B
    print "Part B time: " + str(time.clock()) + ' sec'

if __name__ == "__main__": main()
