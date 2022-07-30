import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#读取交易数据
df = pd.read_csv('Market_Basket.csv', header = None)
df.head()
df.count()#对各列的非空数值进行计数
#增添交易纪录
transactions = []
for i in range(0, 7501):
    transactions.append([str(df.values[i,j]) for j in range(0, 20)]) #第i次购买事物(即第i个客户)购买的商品清单
#transactions[1]
#使用Apriori 产生关联规则
from apyori import apriori
rules = apriori(transactions, min_support = 0.003, min_confidence = 0.2, min_lift = 3, min_length = 2)#套用Apriori算法

# Visualising the results
results = list(rules)
results[0]
results[0].ordered_statistics[0]
#产生关联规则
for rec in results:
    left_hands = rec.ordered_statistics[0].items_base
    right_hands = rec.ordered_statistics[0].items_add
    l = ';'.join([item for item in left_hands])
    r = ';'.join([item for item in right_hands])
    print('{} => {}'.format(l,r))
#产生频繁交易集
# 'milk', 'spaghetti', 'avocado'
#'milk', 'spaghetti'
#'milk',  'avocado'
#'spaghetti', 'avocado'
import itertools
for ele in itertools.combinations(['milk', 'spaghetti', 'avocado'], 2):
    print(ele)
itemsets = []
for rec in results:    
    #print(rec.items)
    for ele in itertools.combinations(rec.items, 2):
        itemsets.append(ele)
#itemsets[0:10]
#通过Gephi来可视化网络,通过网络来关联。
import pandas
df2 = pandas.DataFrame(itemsets)
df2.columns = ['Source','Target']
df2['Type'] = 'undirected'
df2.to_csv('transactions.csv')    

