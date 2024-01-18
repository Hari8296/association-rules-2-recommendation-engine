Name :- Hari singhr 
batch id :-DSWDMCOD 25082022 B

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

movie=pd.read_csv("D:/assignments of data science/10 association rules 2 & recommendation engine/my_movies.csv")

movie.head()
movie.info()
movie.describe()
movie.duplicated().sum()
movie.isnull().sum()

movie.mean()
movie.median()
list(movie.mode())
movie.skew()
movie.kurt()
movie.var()

columns_to_drop=['V1','V2','V3','V4','V5']
df=movie.drop(columns=columns_to_drop)

from mlxtend.frequent_patterns import apriori, association_rules

liked_movies=apriori(df,min_support=0.05,use_colnames=False)

liked_movies.sort_values('support',ascending=False,inplace=True)

plt.bar(x = list(range(0,30)), height = liked_movies.support[0:30], color = 'red')
plt.xticks(list(range(0,30)), liked_movies.itemsets[0:30], rotation = 90)
plt.ylabel('item-sets')
plt.xlabel('support')
plt.show()

plt.bar(x = list(range(0,7)), height = liked_movies.support[0:7], color = 'red')
plt.xticks(list(range(0,7)), liked_movies.itemsets[0:7], rotation = 90)
plt.ylabel('item-sets')
plt.xlabel('support')
plt.show()

rules = association_rules(liked_movies, metric = 'lift', min_threshold=2)
rules1 = association_rules(liked_movies, metric = 'confidence', min_threshold=1)


def to_list(i):
    return(sorted(list(i)))

ma_x = rules.antecedents.apply(to_list) + rules.consequents.apply(to_list)
ma_x = ma_x.apply(sorted)
rules_set = list(ma_x)
unique_rules_set = [list(m) for m in set(tuple(i)for i in rules_set)]

index_rules = []

for i in unique_rules_set:
    index_rules.append(rules_set.index(i))

rules_no_redudancy = rules.iloc[index_rules, :]

rules_no_redudancy.sort_values('lift', ascending = False).head(10)

ma_x1 = rules1.antecedents.apply(to_list) + rules1.consequents.apply(to_list)
ma_x1 = ma_x1.apply(sorted)
rules_set1 = list(ma_x1)
unique_rules_set1 = [list(m) for m in set(tuple(i)for i in rules_set1)]

index_rules1 = []

for i in unique_rules_set1:
    index_rules1.append(rules_set1.index(i))

rules_no_redudancy1 = rules1.iloc[index_rules1, :]

rules_no_redudancy1.sort_values('lift', ascending = False).head(10)























