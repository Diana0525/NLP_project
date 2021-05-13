import re  #正则表达式
from bs4 import BeautifulSoup  #html标签处理
import pandas as pd  #读取数据
from nltk.corpus import stopwords  #删除停用词
import jieba #中文分词
from sklearn.naive_bayes import MultinomialNB as MNB
from sklearn.feature_extraction.text import TfidfVectorizer as TFIV
from sklearn.model_selection import cross_val_score
import numpy as np
from sklearn.metrics import classification_report
 
def review_to_wordlist(review):
 
    # 去掉<br /><br />
    review_text = BeautifulSoup(review, "html.parser").get_text()
    # 去除标点
    review_text = re.sub("[^a-zA-Z]"," ", review_text)
    # 分词
    words = review_text.lower().split()
    #去除停用词
    #words = [w for w in words if not w in stopwords.words("chinese")]
    # 返回words
    return words
 
# 使用pandas读入训练和测试csv文件这里,“header= 0”表示该文件的第一行包含列名称,
#“delimiter=\t”表示字段由\t分割, quoting=3告诉Python忽略双引号


train_origin = pd.read_csv(r'D:\\cnn\\NLP\\NLP_project\\train_data.csv', header=0, delimiter=";", quoting=3)
unlabeled_origin = pd.read_csv(r'D:\\cnn\\NLP\\NLP_project\\unlabel_data.csv', header=0, delimiter=",")
test_origin = pd.read_csv(r'D:\\cnn\\NLP\\NLP_project\\test1_data.csv', header=0, delimiter=",")
test2_origin = pd.read_csv(r'D:\\cnn\\NLP\\NLP_project\\test2_data.csv', header=0, delimiter=",")
#预览数据
# 取出情感标签，positive/褒 或者 negative/贬
#使用中文分词构造文本，便于tf-idf
train_origin_text = train_origin['text'][0:1000]
unlabeled_origin_text = unlabeled_origin['text'][0:99999]
test_origin_text = test_origin['text'][0:1000]
test2_origin_text = test2_origin['text'][0:20000]
train_text = []
unlabeled_text = []
test_text = []
test2_text = []
# 将训练和测试数据都转成词list
for stri in train_origin_text:
    seg_list = jieba.cut(stri,use_paddle=True) # 使用paddle模式
    train_text.append(' '.join(list(seg_list)))
for stri in unlabeled_origin_text:
    seg_list = jieba.cut(stri,use_paddle=True)
    unlabeled_text.append(' '.join(list(seg_list)))

for stri in test_origin_text:
    seg_list = jieba.cut(stri,use_paddle=True)
    test_text.append(' '.join(list(seg_list)))

for stri in test2_origin_text:
    seg_list = jieba.cut(stri,use_paddle=True)
    test2_text.append(' '.join(list(seg_list)))

train_label = []
for i in range(0,1000):
    train_label.append(" ".join(review_to_wordlist(train_origin['label'][i])))



# min_df=3去除低词频的词,分析的视角是词，启用ngram_range，启用ITF，启用idf平滑smooth_idf=1
#用1 + log（tf）替换tfsublinear_tf=1
tfv = TFIV(min_df=3,  strip_accents='unicode', analyzer='word',ngram_range=(1, 2)
, use_idf=1,smooth_idf=1,sublinear_tf=1)
 
# 注意我只用训练集训练
tfv.fit(train_text)


X_all = train_text + unlabeled_text +test_text + test2_text
len_train = len(train_text)
len_unlabeled = len(unlabeled_text)
len_test = len(test_text)
X_all = tfv.transform(X_all)
 
 
# 恢复成训练集和测试集部分
# 左闭右开
train_X = X_all[:len_train]
unlabeled_X = X_all[len_train:len_train+len_unlabeled]
test_X = X_all[len_train+len_unlabeled:len_train+len_unlabeled+len_test]
test2_X = X_all[len_train+len_unlabeled+len_test:]
MNB(alpha=1.0, class_prior=None, fit_prior=True)
'''
alpha ： float，optional（默认值= 1.0）
拉普拉斯平滑参数（0表示无平滑）。
fit_prior ： boolean，optional（default = True）
如果为假，则使用统一的先验。
class_prior ： 可选（默认=无）
类的先验概率。如果指定，则不根据数据调整先验。
'''
model_NB = MNB()
model_NB.fit(train_X, train_label) #特征数据直接灌进来

#使用测试集测试效果。输出信息供参考
print("predict")
unlabeled_label = model_NB.predict(unlabeled_X)
test_label = model_NB.predict(test_X)
test2_label = model_NB.predict(test2_X)
# print(unlabeled_label[0:20])
# print("text")
# print(unlabeled_origin_text[0:20])
# print("predict")
# print(model_NB.predict(test_X)[0:20])
# print("text")
# print(test_origin_text[0:20])

#EM Test
'''
for i in range(0,40):
    model_NB.fit(unlabeled_X, unlabeled_label)
    unlabeled_label = model_NB.predict(unlabeled_X)
    #使用测试集测试效果。输出信息供参考
    print(i)
    print("\n\n")
    print("predict")
    unlabeled_label = model_NB.predict(unlabeled_X)
    print(unlabeled_label[0:20])
    print("text")
    print(unlabeled_origin_text[0:20])
    print("predict")
    print(model_NB.predict(test_X)[0:20])
    print("text")
    print(test_origin_text[0:20])
    print("\n\n")
'''

#实验结果表明，EM循环两次就行了
#第3次开始会有大量negative
#但为什么?
model_NB.fit(unlabeled_X, unlabeled_label)
unlabeled_label = model_NB.predict(unlabeled_X)
model_NB.fit(unlabeled_X, unlabeled_label)
unlabeled_label = model_NB.predict(unlabeled_X)


unlabeled_label = model_NB.predict(unlabeled_X)
test_label = model_NB.predict(test_X)
test2_label = model_NB.predict(test2_X)
print("text:")
print(test2_origin_text[0:20])
print("predict:")
print(test2_label[0:20])
# print(len(test_label))
# print(unlabeled_label[0:20])
# print("text")
# print(unlabeled_origin_text[0:20])
# print("predict")
# print(model_NB.predict(test_X)[0:20])
# print("text")
# print(test_origin_text[0:20])

# 对test的1000条数据写出预测结果
result = []
for i in test2_label:
    if i == 'positive':
        result.append(1)
    elif i == 'negative':
        result.append(0)
print(len(result))

file = open('answer.txt', 'w')
for i in range(len(result)):
    s = str(result[i]).replace('{', '').replace('}', '').replace("'", '').replace(':', ',') + '\n'
    file.write(s)
file.close()