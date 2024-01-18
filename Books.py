Name :- Hari singhr 
batch id :-DSWDMCOD 25082022 B

import pandas as pd
from mlxtend.frequent_patterns import apriori , association_rules
import matplotlib.pylab as plt

books=pd.read_csv("D:/assignments of data science/Data sceince/10 association rules 2 & recommendation engine/book.csv")
books

books.head()
books.info()
books.describe()
books.duplicated().sum()
books.isnull().sum()

books.mean()
books.median()
list(books.mode())
books.skew()
books.kurt()
books.var()

for i in books.columns:
    plt.hist(books[i])
    plt.xlabel(i)
    plt.show()


frequent_itemsets=apriori(books,min_support=0.05,max_len=5,use_colnames=True)

frequent_itemsets.sort_values('support',ascending = False ,inplace=True)

plt.bar(x=list(range(0,11)),height=frequent_itemsets.support[0:11],color='red')
plt.xticks(list(range(0,11)),frequent_itemsets.itemsets[0:11],rotation=90)
plt.xlabel('item-set')
plt.ylabel('support')
plt.show()

rules=association_rules(frequent_itemsets,metric="lift",min_threshold=1)
rules.head(20)
rules.sort_values('lift',ascending=False).head(10)

def to_list(i):
    return (sorted(list(i)))
    
ma_X=rules.antecedents.apply(to_list)+rules.consequents.apply(to_list)      
ma_X=ma_X.apply(sorted)
rules_sets=list(ma_X)
unique_rules_sets=[list(m)for m in set(tuple(i)for i in rules_sets)]
index_rules=[]

for i in unique_rules_sets:
    index_rules.append(rules_sets.index(i)) 

rules_no_redudancy=rules.iloc[index_rules,:]

rules_no_redudancy.sort_values('lift',ascending=False).head(10)



















    