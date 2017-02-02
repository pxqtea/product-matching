########################################################################
########################## BRAND NAME EXTRACTOR ########################
########################################################################

## import modules
import os, sys
import re
import json # module for parsing json string
from random import *
from pytrie import SortedStringTrie as trie # pytrie package (https://pypi.python.org/pypi/PyTrie)

########################################################################
##################### DEFINE FUNCTIONS #################################
def brandExtractor(brand_dict, max_brand_length, product_name):
    ''' INPUT: 
               a brand name dictionary brand_dict (a trie, the longest brand name is of length max_brand_length)
               a product name product_name (a string)
        OUTPUT: 
               a extracted brand name (a string, can be empty)        
        RULES: 
               1. prefers longer brand name (e.g, Apple iPhone over Apple)
               2. prefers brand name that appears closer to the beginning the product name
               3. brand name must appear in the first half of the product name
    '''
    brand = '' # extracted brand name
    
    words = product_name.split(" ") # all the words in the product name
    maxLen = min(max_brand_length, len(words))
    
    position = len(product_name) - 1 # the starting position of the extracted brand name in the product name
    for i in range(1, maxLen + 1): # substring of lenght i
        parts = [] # to store all substrings of length i
        if i == 1: 
            parts = words
        else:
            for j in range(0, len(words) - i + 1):
                part = words[j]
                for k in range(1, i):
                    part = part + ' ' + words[j + k]
                parts.append(part)
        
        # up to this point, all substrings of length i were stored in parts
        
        # see if every one of the substring of length i is in the dictionary
        # RULES: 1. prefers longer brand name (e.g, Apple iPhone over Apple)
        #        2. prefers brand name that appears closer to the beginning the product name
        #        3. brand name must appear in the first half of the product name
        for w in parts: 
            brand_candidates = brand_dict.keys(prefix=w) # prefix lookup in the trie
            if (len(brand_candidates) > 0) and (w in brand_candidates): # check for exact match
                if (product_name.index(w) <= position) and (product_name.index(w) <= len(product_name)/2):
                    position = product_name.index(w)
                    brand = w
                break
                    
    return brand

def batchExtraction(brand_dict, maxBrandLength, product_names):
    ''' INPUT:
             brand_dict - brand name dictionary (a trie)
             maxBrandLength - the length of the longest brand name in the dictionary
             product_names - a list of product names
        OUTPUT:
             a tuple [tp, tn, fp, fn], [tp_r, tn_r, fp_r, fn_r]
    '''
    ## statistics under extact match (e.g., labeled "Google" and extracted "Google Nexus" IS NOT a match)
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    ## statistics under relaxed match (e.g., "Google" and "Google Nexus" IS a match 
    ## because the labeled is contained in the extracted, or vice versa)
    tp_r = 0
    tn_r = 0
    fp_r = 0
    fn_r = 0
    counter = 0
    for product in product_names:
        try:
            brand_labelled = product["Brand"].lower().strip()        
            proName = product["Product Name"][0].lower() 

            ## where actually extract brand name
            brand_extracted = brandExtractor(t, maxBrandLength, proName)
            if brand_extracted == brand_labelled:
                if brand_extracted == '':
                    tn += 1
                    tn_r += 1
                else:
                    tp += 1
                    tp_r += 1
            else:
                print ("---MISMATCH---")
                print ("LABEL:", brand_labelled)
                print ("EXTRACT:", brand_extracted)
                print ("PRODUCT NAME:", proName, '\n')
                #print "PRODUCT ID:", product["Product ID"], '\n'

                if brand_extracted == '':
                    fn += 1
                    fn_r += 1
                else:
                    fp += 1
                    if brand_labelled != "" and brand_labelled in brand_extracted:
                        tp_r += 1
                    else:
                        fp_r += 1
            
            counter += 1
        
        except:
            pass  

    print ("\nTOTAL:", counter)
    
    print ("\n-----UNDER THE EXTACT MATCH CRITERION------")
    print ("TRUE POSITIVE:", tp, "\t TRUE NEGATIVE:", tn, "\t FALSE POSITIVE:", fp, "\t FALSE NEGATIVE:", fn)
    print ("PRECISION:", tp * 1.0 /(tp + fp), "\t RECALL:", tp * 1.0 /(tp + fn))
    
    print ("\n-----UNDER THE RELAXED MATCH CRITERION------")
    print ("TRUE POSITIVE:", tp_r, "\t TRUE NEGATIVE:", tn_r, "\t FALSE POSITIVE:", fp_r, "\t FALSE NEGATIVE:", fn_r)
    print ("PRECISION:", tp_r * 1.0 /(tp_r + fp_r), "\t RECALL:", tp_r * 1.0 /(tp_r + fn_r))
    
    return [tp, tn, fp, fn], [tp_r, tn_r, fp_r, fn_r]
    
##################### END OF DEFINE FUNCTIONS ############################
##########################################################################

## data files
dictFile = "elec_brand_dic_augmented_variation.txt"
devFile = "sample_train_updated.txt"
testFile = "sample_test_updated.txt"

print ("################# BUILDING BRAND NAME TRIE #####################\n")
## get current dir as working dir
curdir = os.getcwd()
dataFile = curdir + os.sep + dictFile

# read data
brandDict_List = [] # a list to hold all brand names in the dictionary
maxBrandLength = 0 # maximum length of the brand names (# of words)

# open the file

with open(dataFile, 'r', encoding = "ISO-8859-1") as infile:
    # read in a line
    line = infile.readline()
    if len(line) == 0:
        break
    # split by '\t'
    values = (line.split('\n')[0]).split('\t')
    # extract values       
    brandDict_List.append(values[0].lower())
    
    length = len(values[0].split(' '))    
    if length > maxBrandLength:
        maxBrandLength = length

print ("# BRAND NAMES IN DICTIONARY:", len(brandDict_List))
print ('MAXIMUM BRAND NAME LENGTH:', maxBrandLength,'\n\n')

## build a trie for brand names in dictionary
t = trie()
for i in range(len(brandDict_List)):
    t[brandDict_List[i]] = i
print ('DONE BUILDING BRAND NAMES TRIE...\n\n')

print ("################# LOADING DEVELOPMENT DATA SET #####################\n")
## read in development set
dataFile = curdir + os.sep + devFile
product_dev = []    # development set

# open the file
with open(dataFile, 'r', encoding = "ISO-8859-1") as infile:
    line = infile.readline()
    if len(line) == 0:
        break
    try:
        product_dev.append(json.loads(line.split('\n')[0])) 
    except:
        pass

print ("DONE LOADING DEVELOPMENT DATA SET...\n\n")
        
## develop brand name extractor (debug should focus on this part)
print ("########################### DEBUGING #############################\n")
batchExtraction(t, maxBrandLength, product_dev)
print ("DONE DEBUGING...\n\n")

## held-aside test
print ("################# LOADING TEST DATA SET #####################\n")
## read in test set
dataFile = curdir + os.sep + testFile
product_test = []    # development set

# open the file
with open(dataFile, 'r', encoding = "ISO-8859-1") as infile:
    line = infile.readline()
    if len(line) == 0:
        break
    try:
        product_test.append(json.loads(line.split('\n')[0]))
    except:
        pass
# done reading, close file
print ("DONE LOADING TEST DATA SET...\n\n")
        
## develop brand name extractor (debug should focus on this part)
print ("########################### TESTING #############################\n")
batchExtraction(t, maxBrandLength, product_test)
print ("DONE HELD-ASIDE TESTING...\n\n")