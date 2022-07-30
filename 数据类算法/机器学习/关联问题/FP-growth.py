import pandas
movie = pandas.read_csv('movies.csv')
movie.head()
movie_dic = {}
for rec in movie.iterrows():
    movie_dic[rec[1].movieId]  = rec[1].title #将DataFrame的一行选取两个字段,构造成字典
movie_dic.get(1)
import pandas
import datetime
df = pandas.read_csv('ratings.csv')
df.info()
df = df[df['timestamp'] >= 1325376000] #筛选13年的数据
df.info()
from apyori import apriori
#如下代码是整个Apriori算法的输入,理解了数据矩阵的物理意义，泛化算法的使用就会轻而易举。
#dfdf1=[ele for ele in df.groupby('userId')['movieId'].apply(list)] #某个用户喜欢的电影集合
transactions = [ele for ele in df.groupby('userId')['movieId'].apply(list)]
rules = apriori(transactions, min_support = 0.2, min_confidence = 0.5, min_lift = 3, min_length = 2)
results = list(rules)
for rec in results:    
    print('  ;\n'.join([movie_dic.get(item) for item in rec.items]))
#[print(item) for item in results[1].items]
from pymining import itemmining
fp_input = itemmining.get_fptree(transactions)
#FPgrowth
report = itemmining.fpgrowth(fp_input, min_support=30, pruning=True)
for ele in report:
    if len(ele) >=6: #选取频繁6项集
        print(' ;'.join([movie_dic.get(item) for item in ele]))
        print('\n')

