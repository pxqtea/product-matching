############## import modules ############
import os, sys
import json # module for parsing json string
from random import *
from py_stringmatching import simfunctions, tokenizers
#from sklearn.tree import  *
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn import tree
import re
from sklearn.cross_validation import KFold, cross_val_predict , train_test_split, StratifiedKFold, ShuffleSplit
#from collections import defaultdict
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.calibration import CalibratedClassifierCV
from itertools import product
from json.decoder import NaN
from sklearn.preprocessing import Imputer
import numpy as np
import csv
#from string_match_dt2 import trainSet, prediction, testSet
#import matplotlib.pyplot as plt



#from sklearn.decomposition.tests.test_nmf import random_state

####### define utility functions #######
# read in json data
def readJsonData(dataFile):
    '''
    dataFile: file name of the product pairs file
    output: a list of product pair json
    '''
    
    pairJson_List = [] # a list to hold all product pairs. Each item in this list is a dictionary
    
    # open the file
    with open(dataFile, 'r', encoding = "ISO-8859-1") as infile:
        # read in a line
        line = infile.readline()
        if len(line) == 0:
            break
        pairJson = json.loads(line.split('\n')[0], 'ISO-8859-1')
        pairJson_List.append(pairJson)

    return pairJson_List

def readTestData(dataFile):
    '''
    dataFile: file name of the product pairs file
    output: a list of product pair json
    '''
    pairJson_List = [] # a list to hold all product pairs. Each item in this list is a dictionary
    attr_List = [] # a list to hold all attributes appear in the product
    attr_List_prod1 = [] # a list to hold attributes from product 1
    attr_List_prod2 = [] # a list to hold attributes from product 2

    # open the file
    with open(dataFile, 'r', encoding = "ISO-8859-1") as infile:
        # read in a line
        line = infile.readline()
        if len(line) == 0:
            break
        values = (line.split('\n')[0]).split('?') # split by '?'
        # construct a dictionary for this product pair
        pairJson = {'pairId': values[0],
                'prod1_id': values[1],
                'prod1_json': json.loads(values[2], 'ISO-8859-1'), # parse json string. This encoding will handle any umlauts that
                                                                    # might get in the JSON message.
                'prod2_id': values[3],
                'prod2_json': json.loads(values[4], 'ISO-8859-1'),
                'label': 0
                }
        pairJson_List.append(pairJson)
    return pairJson_List

## write json Data ## write the generated features out
def writeJsonData(dataFile, pairJsonList):
    '''
    dataFile: output data file name
    pairJsonList: a list of product pairs json
    '''
    with open(dataFile, 'w') as outfile:
##    for pair in pairJsonList:
        outfile.write(json.dumps(pair) + '\n')

# generate an instance given a list of attributes and a proudct pair json
def generateInstance(attrList, prodPairJson): 
    '''
    attrList: list of attributes on which sim measure will be calculated
    prodPairJson: dictionary holding a product pair
    output: features -- edit distance and Jaccard score on each attribute
            label -- 1 for MATCH, 0 for MISMATCH
    '''
    features = [] # [f1, f2, ..., fn]
    label = 0

    n = len(attrList)
    for i in range(n):
        attrName = attrList[i]
        #print "attrName:", attrName
        attrValue1 = ''
        attrValue2 = ''
        simE = NaN #-1.0  ##last test was 0
        simJ = NaN # -1.0
        simO = NaN # -1.0
        simC = NaN # -1.0
        simME = NaN # -1.0
        simNW = NaN # -1.0
        simJR = NaN # -1.0
        simJW = NaN # -1.0
        simSW = NaN # -1.0
        simA =  NaN #-1.0
        simTI = NaN # -1.0
        simSTI = NaN # -1.0
        simComp = NaN # -1.0
        maxlength = NaN # - 1.0
        diff = NaN # - 1.0
        
        
        #print prodPairJson["prod1_json"].keys()
        
        if attrName in prodPairJson["prod1_json"].keys():
            attrValue1 = prodPairJson["prod1_json"][attrName][0]
        if attrName in prodPairJson["prod2_json"].keys():
            attrValue2 = prodPairJson["prod2_json"][attrName][0]
            
        
        if attrValue1 != '' and attrValue2 != '':
            #print attrValue1, attrValue2
            ## for short strings, no missing value
            if attrName in [ 'Product Segment', 'Product Type', 'Brand' ]: 
                simE = 1 - simfunctions.levenshtein(attrValue1, attrValue2)/max(len(attrValue1), len(attrValue2));
                simJ = simfunctions.jaccard(tokenizers.whitespace(attrValue1), tokenizers.whitespace(attrValue2))
               # simO = simfunctions.overlap_coefficient(tokenizers.whitespace(attrValue1), tokenizers.whitespace(attrValue2))
                simC = simfunctions.cosine(tokenizers.whitespace(attrValue1), tokenizers.whitespace(attrValue2))
               # simME = simfunctions.monge_elkan(tokenizers.whitespace(attrValue1), tokenizers.whitespace(attrValue2))
                simNW = simfunctions.needleman_wunsch(attrValue1, attrValue2)/max(len(attrValue1), len(attrValue2))
                simJR = simfunctions.jaro(attrValue1, attrValue2)
                simJW = simfunctions.jaro_winkler(attrValue1, attrValue2)
               # simSW = simfunctions.smith_waterman(attrValue1, attrValue1) #/max(len(attrValue1), len(attrValue2))
         ##       simA = simfunctions.affine(attrValue1, attrValue2) #/max(len(attrValue1), len(attrValue2))
               # simTI = simfunctions.tfidf(tokenizers.whitespace(attrValue1), tokenizers.whitespace(attrValue1)) 
               # simSTI = simfunctions.soft_tfidf(tokenizers.whitespace(attrValue1), tokenizers.whitespace(attrValue1)) 
                maxlength = max(len(attrValue1), len(attrValue2))
                diff = maxlength - min(len(attrValue1), len(attrValue2))
               

                features.append(simE)
                features.append(simJ)
                features.append(simC)
                features.append(simNW)
                features.append(simJR)
                features.append(simJW)
            ##    features.append(simA)
            #    features.append(maxlength)
                features.append(diff)
                
                '''
                print(attrValue1, " ? ") 
                print(attrValue2, " ? ")
                print("simE: ", simE, "simJ: ", simJ, "simO:", simO, "simC:", simC, "simME:", simME, "simNW:", simNW) 
                print("simJR:", simJR, "simJW:", simJW, "simSW:", simSW, "simA:", simA, "simTI:", simTI, "simSTI:", simSTI)
                print("LABEL: ", prodPairJson["label"])
                print()
                '''
            ## for medium strings, no missing value
            if attrName in [ 'Product Name' ]: 
                simE = 1 - simfunctions.levenshtein(attrValue1, attrValue2)/max(len(attrValue1), len(attrValue2));
                simJ = simfunctions.jaccard(tokenizers.whitespace(attrValue1), tokenizers.whitespace(attrValue2))
               # simO = simfunctions.overlap_coefficient(tokenizers.whitespace(attrValue1), tokenizers.whitespace(attrValue2))
                simC = simfunctions.cosine(tokenizers.whitespace(attrValue1), tokenizers.whitespace(attrValue2))
                simME = simfunctions.monge_elkan(tokenizers.whitespace(attrValue1), tokenizers.whitespace(attrValue2))
                #simNW = simfunctions.needleman_wunsch(attrValue1, attrValue2)/#max(len(attrValue1), len(attrValue2))
                #simJR = simfunctions.jaro(attrValue1, attrValue2)
                #simJW = simfunctions.jaro_winkler(attrValue1, attrValue2)
            ##    simSW = simfunctions.smith_waterman(attrValue1, attrValue1)# /max(len(attrValue1), len(attrValue2))
           ##     simA = simfunctions.affine(attrValue1, attrValue2) #/max(len(attrValue1), len(attrValue2))
               # simTI = simfunctions.tfidf(tokenizers.whitespace(attrValue1), tokenizers.whitespace(attrValue1)) 
               # simSTI = simfunctions.soft_tfidf(tokenizers.whitespace(attrValue1), tokenizers.whitespace(attrValue1)) 
                maxlength = max(len(attrValue1), len(attrValue2))
                diff = maxlength - min(len(attrValue1), len(attrValue2))
                
                features.append(simE)
                features.append(simJ)
                features.append(simC)
                features.append(simME)
             ##   features.append(simSW)
            ##    features.append(simA)
             #   features.append(maxlength)
                features.append(diff)
                '''
                print(attrValue1, " ? ") 
                print(attrValue2, " ? ")
                print("simE: ", simE, "simJ: ", simJ, "simO:", simO, "simC:", simC, "simME:", simME, "simNW:", simNW) 
                print("simJR:", simJR, "simJW:", simJW, "simSW:", simSW, "simA:", simA, "simTI:", simTI, "simSTI:", simSTI)
                print("LABEL: ", prodPairJson["label"])
                print()
                '''
            ## for long strings, no missing value
            if attrName in [ 'Product Long Description', 'Product Short Description' ]: 
                #simE = 1 - simfunctions.levenshtein(attrValue1, attrValue2) /max(len(attrValue1), len(attrValue2));
                simJ = simfunctions.jaccard(tokenizers.whitespace(attrValue1), tokenizers.whitespace(attrValue2))
                simO = simfunctions.overlap_coefficient(tokenizers.whitespace(attrValue1), tokenizers.whitespace(attrValue2))
                #simC = simfunctions.cosine(tokenizers.whitespace(attrValue1), tokenizers.whitespace(attrValue2))
                simME = simfunctions.monge_elkan(tokenizers.whitespace(attrValue1), tokenizers.whitespace(attrValue2))
                #simNW = simfunctions.needleman_wunsch(attrValue1, attrValue2) #/max(len(attrValue1), len(attrValue2))
                #simJR = simfunctions.jaro(attrValue1, attrValue2)
                #simJW = simfunctions.jaro_winkler(attrValue1, attrValue2)
               # simSW = simfunctions.smith_waterman(attrValue1, attrValue1) #/max(len(attrValue1), len(attrValue2))
                #simA = simfunctions.affine(attrValue1, attrValue2) #/max(len(attrValue1), len(attrValue2))
               # simTI = simfunctions.tfidf(tokenizers.whitespace(attrValue1), tokenizers.whitespace(attrValue1)) 
               # simSTI = simfunctions.soft_tfidf(tokenizers.whitespace(attrValue1), tokenizers.whitespace(attrValue1)) 
                maxlength = max(len(attrValue1), len(attrValue2))
                diff = maxlength / min(len(attrValue1), len(attrValue2))
                
                features.append(simJ)
                features.append(simO)
                features.append(simME)
            #    features.append(maxlength)
                features.append(diff)

                '''
                print(attrValue1, " ? ") 
                print(attrValue2, " ? ")
                print("simE: ", simE, "simJ: ", simJ, "simO:", simO, "simC:", simC, "simME:", simME, "simNW:", simNW) 
                print("simJR:", simJR, "simJW:", simJW, "simSW:", simSW, "simA:", simA, "simTI:", simTI, "simSTI:", simSTI)
                print("LABEL: ", prodPairJson["label"])
                print()
                '''
            ## for numeric feature
            if attrName in [ 'Assembled Product Length', 'Assembled Product Width', 'Assembled Product Height' ]: 
                intValue1 = int(re.search(r'\d+', attrValue1).group()) 
                intValue2 = int(re.search(r'\d+', attrValue2).group()) 
                if intValue2 != 0: simComp = intValue1 / intValue2  
                else: simComp = 0 
                features.append(simComp)
                '''
                print(attrValue1, " ? ") 
                print(attrValue2, " ? ")
                print("simComp: ", simComp, "LABEL: ", prodPairJson["label"])
                print()
                '''
                        
                '''
                if simE == 1.0 : print (simE, "   " , prodPairJson["label"])
                if simE == 1.0 and prodPairJson["label"] != "MATCH": 
                    print("product name and label inconsistent")
                '''
                '''    
            if attrName in ['UPC']:
                print(attrValue1, "UPC 1 ?")
                print(attrValue2, "UPC 2 ?")
                print()
                
            if attrName in ['GTIN']:
                print(attrValue1, "GTIN 1 ?")
                print(attrValue2, "GTIN 2 ?")
                print()
            if attrName in ['Product Long Description', 'Product Short Description']:
                simC = simfunctions.cosine(tokenizers.whitespace(attrValue1), tokenizers.whitespace(attrValue2))
                simME = simfunctions.monge_elkan(tokenizers.whitespace(attrValue1), tokenizers.whitespace(attrValue2))
                simJ = simfunctions.jaccard(tokenizers.whitespace(attrValue1), tokenizers.whitespace(attrValue2))

                
            if attrName not in ['GTIN', 'UPC']:
                simJ = simfunctions.jaccard(tokenizers.whitespace(attrValue1), tokenizers.whitespace(attrValue2))
                simO = simfunctions.overlap_coefficient(tokenizers.whitespace(attrValue1), tokenizers.whitespace(attrValue2))
                simC = simfunctions.cosine(tokenizers.whitespace(attrValue1), tokenizers.whitespace(attrValue2))
                '''
                
        if attrValue1 == '' or attrValue2 == '':
            if attrName in [ 'Product Segment', 'Product Type', 'Brand' ]: 
                features.append(simE)
                features.append(simJ)
                features.append(simC)
                features.append(simNW)
                features.append(simJR)
                features.append(simJW)
            ##    features.append(simA)
             #   features.append(maxlength)
                features.append(diff)
            if attrName in [ 'Product Name' ]:
                features.append(simE)
                features.append(simJ)
                features.append(simC)
                features.append(simME)
             ##   features.append(simSW)
             ##   features.append(simA)
            #    features.append(maxlength)
                features.append(diff)
            if attrName in [ 'Product Long Description', 'Product Short Description' ]:
                features.append(simJ)
                features.append(simO)
                features.append(simME)
            #    features.append(maxlength)
                features.append(diff)
            
            if attrName in [ 'Assembled Product Length', 'Assembled Product Width', 'Assembled Product Height' ]: 
                features.append(simComp)
             

        ''' 
            if attrName in ['UPC']:
                print(attrValue1, "UPC 1 ?")
                print(attrValue2, "UPC 2 ?")
                
            if attrName in ['GTIN']:
                print(attrValue1, "GTIN 1 ?")
                print(attrValue2, "GTIN 2 ?")
        print()
        '''
        '''    
        features.append(simE)
        features.append(simJ)
        features.append(simO)
        features.append(simC)
        features.append(simME)
        features.append(simNW)
        features.append(simJR)
        features.append(simJW)
        features.append(simSW)
        features.append(simA)
        features.append(simTI)
        features.append(simSTI)
        '''
        
      
        '''  
        if attrName in ['UPC']:
            if attrValue2 != '':
                print("product 2 contains UPC")
            else: 
                if attrValue1 =='':
                    print( " product 1 doesnot contain UPC")

        
            
        if attrName in ['GTIN'] :
            if attrValue2 != '':
                print("product 2 contains GTIN")
            else: 
                if attrValue1 == '' :
                    print(" product 1 doesnot contain GTIN")            
        '''
    #### for test here     
    if prodPairJson["label"] == "MATCH":
        label = 1    

        
    return features, label

## compute accuracy
def calcAccuracy(trueLables, predictedLables, ids):
    '''
    trueLables: a list of true lables (0/1)
    predictedLables: a list of predicted lables (0/1)
    ids: pair id of the instance
    output: precision, recall, F1
    '''
    
    n = len(trueLables)
    
    if len(predictedLables) != n:
        print ("the length of predictions does not match that of the truth. exit")
        sys.exit(1)
    
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    
    for i in range(n):
        trueLabel = trueLables[i]
        predictedLabel = predictedLables[i]
        #print trueLabel, predictedLabel
        if trueLabel == predictedLabel:
            if predictedLabel == 1: tp += 1
            else: tn += 1
        else:
            if predictedLabel == 1 and trueLabel == 0: 
                fp += 1
                #print ('FP --', ids[i])
            else: 
                fn += 1
                #print ('FN --', ids[i])
    
    precision = float(tp) / (tp + fp)
    recall = float(tp) / (tp + fn)
    F1 = 2 * precision * recall / (precision + recall )
    
    print ("TOTAL:", n )
    print ("TP:", tp, "FP:", fp, "TN:", tn, "FN:", fn)
    
    return precision, recall, F1
    
    
def calcAccuracy_prob(trueLables, predictedLables, ids, a, b):
    '''
    trueLables: a list of true lables (0/1)
    predictedLables: a list of predicted lables (0/1)
    ids: pair id of the instance
    output: precision, recall, F1
    '''
    
    n = len(trueLables)
    
    if len(predictedLables) != n:
        print ("the length of predictions does not match that of the truth. exit")
        sys.exit(1)
    
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    ttotal = 0
    unknown = 0
    
    fpf = open("./results/test_fp.txt", "a")
    fnf = open("./results/test_fn.txt", "a")
    tpf = open("./results/test_tp.txt", "a")
    tnf = open("./results/test_tn.txt", "a")
    
    fpf.write(str(a))
    fpf.write("\t")  
    fpf.write(str(b))
    fpf.write("\n")  

  
    fnf.write(str(a))  
    fnf.write("\t")      
    fnf.write(str(b))  
    fnf.write("\n")  

    
    tpf.write(str(a))  
    tpf.write("\t")          
    tpf.write(str(b)) 
    tpf.write("\n")  

  
    tnf.write(str(a)) 
    tnf.write("\t")           
    tnf.write(str(b)) 
    fnf.write("\n")  
        
    for i in range(n):
        trueLabel = trueLables[i]
        predictedLabel = predictedLables[i]
        if (predictedLabel > a): 
            Label = 1
        else: 
            if (predictedLabel <= b): 
                Label = 0
            else: 
                Label = -1
                unknown += 1
            
        #print(predictedLabel)
        #print trueLabel, predictedLabel
        if trueLabel == 1: ttotal += 1
        #print(ttotal)
        if trueLabel == Label:
            if Label == 1: 
                tp += 1
                tpf.write(str(predictedLabel))
                tpf.write('\n')
                #tpary.add(predictedLabel)
            else: 
                tn += 1
                tnf.write(str(predictedLabel))
                tnf.write('\n')
                #tnary.add(predictedLabel)

        else:
            if Label == 1 and trueLabel == 0: 
                fp += 1
                #print ('FP --', ids[i], 'true label ', trueLabel, 'predicted label ', \
                 #      Label , 'prob', predictedLabel )
                fpf.write(str(predictedLabel))
                fpf.write('\n')
                #fpary.add(predictedLabel)
            else: 
                if Label != -1: 
                    fn += 1
                    #print ('FN --', ids[i],  'true label ', trueLabel , 'predicted label ', \
                     #  Label, 'prob', predictedLabel)                   
                    fnf.write(str(predictedLabel))
                    fnf.write('\n')
                    #fnary.add(predictedLabel)
    
    
    fpf.write('   \n')
    fnf.write('   \n')
    tpf.write('   \n')
    tnf.write('   \n')
    
    fpf.close()
    fnf.close()
    tpf.close()
    tnf.close()
    
    

    precision = tp * 1.0 / (tp + fp)
    recall = tp * 1.0 / ttotal
    F1 = 2 * precision * recall / (precision + recall )
    
    #print ("TOTAL:", n , "unknow: ", unknown)
    #print ("unknown: ", unknown)
    print ("TP:", tp, "FP:", fp, "TN:", tn, "FN:", fn)
    
    return precision, recall, F1

def writeCSV(prediction, a, b, n , z ): 
    csvfile = open("./results/test_prediction.csv", 'a')
    writer = csv.writer(csvfile)
    list = ['Parameters:', n, z,   a, b]
    writer.writerow(list)
    
    n = len(prediction)
    for i in range(n):
        predictedLabel = prediction[i]
        if (predictedLabel > a): 
            Label = "MATCH"
        else: 
            if (predictedLabel <= b): 
                Label = "MISMATCH"
            else: 
                Label = "UNKNOWN"
            
        list = [i+1,  Label]
        writer.writerow(list)
        
    #csvfile.Close()

################## End of define functions ##########################

######################## Main Program ###############################

## STEP 4: REAND IN DATA
# get current dir as working dir
curdir = os.getcwd()
trainfn = curdir + os.sep + "stage3_train.txt" ##"train_half.txt" #"trainSet_s.txt" # "train_half_1.txt" ##"
## the test set is not used in cross validation
testfn = curdir + os.sep + "elec_pairs_stage3_test1_20K_anon.txt" #"test_half.txt" #
trainSample = readJsonData(trainfn)
testSample = readTestData(testfn)
 
trainSet = {"data":[],"label":[]}
testSet = {"data":[],"label":[]}

trainIDs = []
testIDs = []

traincsv = open('trainSet_features_reduced_nn.csv', 'r', encoding = "ISO-8859-1")
reader = csv.reader(traincsv)

for rd in reader: 
    trainSet["data"].append(rd)
    #print(rd)    
    
for pair in trainSample:
    if pair["label"] == "MATCH":
        label = 1 
    else: label = 0
    trainSet["label"].append(label) 
    trainIDs.append(pair["pairId"])

testcsv = open('testSet_features_reduced_nn.csv', 'r', encoding = "ISO-8859-1")
treader = csv.reader(testcsv)

for rdt in treader: 
    testSet["data"].append(rdt)
    testSet["label"].append(0)
    #print(rd)    
    
    
##impute the missing value
#i = Imputer (strategy = 'median')
#i.fit(trainSet["data"])
#trainSet["data"] = i.transform(trainSet["data"])
#testSet["data"] = i.transform(testSet["data"])
#imputed_test = i.transform(test_data)


###################### Train a decision tree ####################
## cross validation
'''
print(len(testSet["data"]))
kf = KFold(len(testSet["data"]), n_folds=3)

x = trainSet
y = trainSet

print("testing K fold")
for trainSet, testSet in kf:
    print(trainSet, testSet)
'''
'''        
#for trainSet, testSet in kf : 
clf_dt = DecisionTreeClassifier(random_state=0)
##X_new = SelectKBest(chi2, k=10).fit_transform(trainSet["data"], trainSet["label"])        

prediction = cross_val_predict(clf_dt, trainSet["data"], \
    trainSet["label"], cv = 10, )

print("prediction from decison tree")
print (calcAccuracy(trainSet["label"], prediction, trainIDs))
print ()


clf_rf = RandomForestClassifier(n_estimators=200, oob_score= True, criterion='gini' )
prediction = cross_val_predict(clf_rf, trainSet["data"], \
    trainSet["label"], cv = 10, )

#print(prediction)
print("prediction from random forest")
print (calcAccuracy(trainSet["label"], prediction, trainIDs))
print ()


clf_rf2 = RandomForestClassifier(n_estimators=200, oob_score= True, criterion='entropy', max_features='auto' )
prediction = cross_val_predict(clf_rf2, trainSet["data"], \
    trainSet["label"], cv = 10, )

#print(prediction)
print("prediction from random forest _ 2")
print (calcAccuracy(trainSet["label"], prediction, trainIDs))
print ()
'''
'''
clf_rf3 = RandomForestClassifier(n_estimators=1000, oob_score= True, criterion='entropy', max_features='log2' )
prediction = cross_val_predict(clf_rf3, trainSet["data"], \
    trainSet["label"], cv = 10, )

#print(prediction)
print("prediction from random forest _ 3")
print (calcAccuracy(trainSet["label"], prediction, trainIDs))
print ()

X_train, X_test, y_train, y_test = train_test_split(trainSet["data"], trainSet["label"])

skf = StratifiedKFold(trainSet["label"], n_folds = 10)
clf_rf4 = RandomForestRegressor(n_estimators=200, oob_score= True,  criterion='mse', max_features='auto', verbose=0)
prediction = cross_val_predict(clf_rf4, trainSet["data"], \
    trainSet["label"], cv = 10, )
##prediction = cross_val_predict_proba(clf_rf4, trainSet["data"], \
##    trainSet["label"], cv = 10, )

#print(prediction)
print("prediction from random forest _ 4")
for a, b in (0.6, 0.4), (0.62, 0.4), (0.65, 0.4), (0.67, 0.4), (0.7, 0.7) :## (0.7, 0.7), (0.6, 0.4), (0.65, 0.4), (0.5, 0.4), (0.45, 0.4): 
    print (a, b)
    print (calcAccuracy_prob(trainSet["label"], prediction, trainIDs, a, b))
print ()
'''
parameter_grid = [
#                  ( 1000, 2000),    ## n_estimators
                 # (0.3, 0.5),      ## max_features
                 # (3, 5 ),            ## min_samples_leaf
                  ( 6, 8, 10, 12, 15 )             ## min_samples_split
]


#for n, z in (1000, 6): #product (*parameter_grid):
n, z = 1000, 6
clf_rf5 = RandomForestRegressor(n_estimators=1000, oob_score= True,  criterion='mse', max_features= 0.3, verbose=0, min_samples_leaf = 3, min_samples_split= z, n_jobs=16)
prediction = cross_val_predict(clf_rf5, trainSet["data"], \
trainSet["label"], cv = 10, )

'''
list = ["parameter: ", 1000, z]
result = prediction
result.append(trainSet["label"])
featurecsv = open("./results/trainSet_prediction_rfr.csv", 'a')
writer = csv.writer(featurecsv)
writer.writerow(list)
writer.writerow(result)
lbcsv = open("./results/trainSet_label_rfr.csv", 'a')
writer = csv.writer(lbcsv)
writer.writerow(list)
writer.writerow(trainSet["label"])
'''
#prediction = cross_val_predict_proba(clf_rf4, trainSet["data"], \
#  trainSet["label"], cv = 10, )
#print (n, z)
#print(prediction)
print("prediction from random forest _ 5")
a, b = 0.626, 0.626 # range(600, 630, 1):
    #a = a / 1000 
    #b = 0.4
print (1000, z , a, b)
print (calcAccuracy_prob(trainSet["label"], prediction, trainIDs, a, b ))


#for  n, z in (1000, 6):
n, z = 1000, 6
clf_rf6 = RandomForestRegressor(n_estimators=1000, oob_score= True,  max_features=0.3, verbose=0, min_samples_leaf = 3, min_samples_split= z, n_jobs=16)

clf_rf6.fit(trainSet["data"], trainSet["label"])
prediction = clf_rf6.predict(testSet["data"])
'''
list = ["parameter: ", 1000, z]
with open("./results/testSet_prediction_rfr.csv", 'a') as featurecsv:
    writer = csv.writer(featurecsv)
writer.writerow(list)
writer.writerow(prediction)
'''
print("prediction from random forest _ 5")
a, b =  0.626, 0.626  #(0.6, 0.4), (0.62, 0.4), (0.65, 0.4), (0.67, 0.4), (0.68, 0.5), (0.7, 0.7): 
print(a, b, 1000 , z)
    #print (calcAccuracy_prob(y_test, prediction, trainIDs))
writeCSV(prediction, a, b, 1000 , z )
print ()

# n, z in (1000, 6): #product (*parameter_grid):
n, z = 1000, 6
clf_rf7 = RandomForestClassifier(n_estimators=1000, oob_score= True, criterion='entropy', max_features=0.3, min_samples_leaf = 3, min_samples_split= z, n_jobs=16)

prediction = cross_val_predict(clf_rf7, trainSet["data"], \
trainSet["label"], cv = 10, )
'''
list = ["parameter: ", n, z]
result = prediction
#result.append(trainSet["label"])
with open("./results/trainSet_prediction_rf.csv", 'a') as featurecsv:
    writer = csv.writer(featurecsv)
writer.writerow(list)
writer.writerow(result)
with open("./results/trainSet_label_rf.csv", 'a') as lbcsv:
    writer = csv.writer(lbcsv)
writer.writerow(list)
writer.writerow(trainSet["label"])
#print(prediction)
'''
print("prediction from random forest _ 3")
print (calcAccuracy(trainSet["label"], prediction, trainIDs))
print ()
clf_rf7.fit(trainSet["data"], trainSet["label"])
prediction = clf_rf7.predict(testSet["data"])

print("prediction from random forest _ 3, 0.7, 0.3", n, z)
writeCSV(prediction, 0.7, 0.3, 1000, z )
print ()

    
print ('Done ...')

