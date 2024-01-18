Name :- Hari singhr 
batch id :-DSWDMCOD 25082022 B

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data=pd.read_csv("D:/assignments of data science/10 association rules 2 & recommendation engine/transactions_retail.csv")

retail = []
with open('D:/assignments of data science/10 association rules 2 & recommendation engine/transactions_retail.csv') as f:
    retail = f.read()
  
retail = retail.split('\n')

retail_list = []
for i in retail:
    retail_list.append(i.split(","))
    
all_retail_list = [i for item in retail_list for i in item]

from collections import Counter

retail_frequencies = Counter(all_retail_list)

retail_frequencies = sorted(retail_frequencies.items(), key = lambda x:x[1])

frequencies = list(reversed([i[1] for i in retail_frequencies]))
items = list(reversed([i[0] for i in retail_frequencies]))

plt.bar(height = frequencies[0:11], x = list(range(0, 11)), color = 'red')
plt.xticks(list(range(0, 11), ), items[0:11], rotation = 90)
plt.xlabel("items")
plt.ylabel("Count")
plt.show()

retail_series = pd.DataFrame(pd.Series(retail_list))

retail_series.columns = ["transactions"]

X = retail_series['transactions'].str.join(sep = '*').str.get_dummies(sep = '*')

frequent_itemsets = apriori(X, min_support = 0.0075, max_len = 4, use_colnames = True)

frequent_itemsets.sort_values('support', ascending = False, inplace = True)

plt.bar(x = list(range(0, 11)), height = frequent_itemsets.support[0:11], color ='red')
plt.xticks(list(range(0, 11)), frequent_itemsets.itemsets[0:11], rotation=90)
plt.xlabel('item-sets')
plt.ylabel('support')
plt.show()

rules = association_rules(frequent_itemsets, metric = "lift", min_threshold = 1)
rules.head(20)
rules.sort_values('lift', ascending = False).head(10)

def to_list(i):
    return (sorted(list(i)))

ma_X = rules.antecedents.apply(to_list) + rules.consequents.apply(to_list)

ma_X = ma_X.apply(sorted)

rules_sets = list(ma_X)

unique_rules_sets = [list(m) for m in set(tuple(i) for i in rules_sets)]

index_rules = []

for i in unique_rules_sets:
    index_rules.append(rules_sets.index(i))
    
rules_no_redudancy = rules.iloc[index_rules, :]
    
rules_no_redudancy.sort_values('lift', ascending = False).head(10)




















































    