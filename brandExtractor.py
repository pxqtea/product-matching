from operator import itemgetter
import json

class BrandExtractor:
    def __init__(self):
        #load lookuptable
        #{brandTuples: [predict most frequent brand, most frequent brand occurrence]}
        with open("../data/LookupTable.json", 'r') as file:
            self.LookupTable = json.load(file)
            
        #brands set
        #{brand name: [synonym list]}
        with open("../data/brandset_lens.json", 'r') as file:
            self.brands_by_lens = json.load(file)

    #predict algo
    #Step1: find all candidate brandnames of a product by searching brandnames in our brandname set.
    def _find_brand_candidates(self, test_productnames):
        test_brandtuples = {}
        for prodname in test_productnames:
            prod = prodname.split()
            test_brandtuples[prodname]=list()
            for length in range(5):
                for i in range(len(prod)-length):
                    cand_brand = " ".join(prod[i:i+length+1])
                    if cand_brand in self.brands_by_lens[length]:
                        if cand_brand not in test_brandtuples[prodname]:
                            test_brandtuples[prodname].append(cand_brand)
        return test_brandtuples

    #predict algo
    #Step2: look up the lookupTable with the brandname set from Step1, and assign prediction of brandname   
    def extract_brandnames(self, test_productnames):
        test_brandtuples = self._find_brand_candidates(test_productnames)
        
        predict_brandnames = {}
        notintable = 0
        for prodname in test_productnames:
            tupleStr = "_".join(sorted(test_brandtuples[prodname]))
            startpositions = [(prodname.find(brand), brand) for brand in test_brandtuples[prodname]]
            startpositions.sort(key=lambda b: len(b[1]), reverse=True)
            startpositions.sort(key=itemgetter(0))

            if not tupleStr:
                predict_brandnames[prodname] = 'n/a'
            elif len(test_brandtuples[prodname]) == 1:
                predict_brandnames[prodname] = startpositions[0][1]
            elif tupleStr not in self.LookupTable:
                notintable += 1
                predict_brandnames[prodname] = startpositions[0][1]
            else:
                most_freq_brandintuple = self.LookupTable[tupleStr]
                if most_freq_brandintuple[1] > 5:
                    predict_brandnames[prodname] = most_freq_brandintuple[0]
                else:
                    #predict_brandnames[prodname] = test_brandtuples[prodname][0]
                    predict_brandnames[prodname] = startpositions[0][1]
        return predict_brandnames
    
class TrueBrandManager:
    def __init__(self):
        self.true_brand_map = {}
        with open('../data/synonyms_All.json') as infile:
            synonyms_dict = json.load(infile)
        for brand, synonyms in synonyms_dict.items():
            same_brand_list = [brand] + synonyms
            self.add_synonyms(same_brand_list)
    def _get_true_brand_id(self, brand_list):
        for brand in brand_list:
            if brand in self.true_brand_map:
                return self.true_brand_map[brand]
        return None
    def _set_true_brand_id(self, true_brand_id, brand_list):
        for brand in brand_list:
            self.true_brand_map[brand] = true_brand_id
    def add_synonyms(self, brands_list):
        true_brand_id = self._get_true_brand_id(brands_list)
        if true_brand_id is None:
            true_brand_id = len(self.true_brand_map)
        self._set_true_brand_id(true_brand_id, brands_list)
    def match(self,brand1, brand2):
        if brand1 in self.true_brand_map and brand2 in self.true_brand_map:
            return self.true_brand_map[brand1] == self.true_brand_map[brand2]
        else:
            return brand1 == brand2

#load testing data
test_data = json.load(open("../data/golden_data_test.json"))

test_productnames = []
test_brandnames = {}
for key in test_data:
    test_productnames.append(test_data[key][0])
    test_brandnames[test_data[key][0]] = test_data[key][1]
    
# Make predictions
brand_extractor = BrandExtractor()
predict_brandnames = brand_extractor.extract_brandnames(test_productnames)

# Evaluate predictions
brand_manager = TrueBrandManager()

tp = sum([1 if brand_manager.match(predict_brandnames[prodname], test_brandnames[prodname]) and 
          predict_brandnames[prodname] != 'n/a' else 0 for prodname in test_productnames])
fp = sum([1 if not brand_manager.match(predict_brandnames[prodname], test_brandnames[prodname]) and 
          predict_brandnames[prodname] != 'n/a' else 0 for prodname in test_productnames])
tn = sum([1 if brand_manager.match(predict_brandnames[prodname], test_brandnames[prodname]) and 
          predict_brandnames[prodname] == 'n/a' else 0 for prodname in test_productnames])
fn = sum([1 if not brand_manager.match(predict_brandnames[prodname], test_brandnames[prodname]) and 
          predict_brandnames[prodname] == 'n/a' else 0 for prodname in test_productnames])

# Print results
print('tp', tp)
print('fp', fp)
print('tn', tn)
print('fn', fn)

print('accuracy', float(tp + tn)/len(test_productnames))
print('precision', float(tp)/(tp + fp))
print('recall', float(tp)/(tp + fn))