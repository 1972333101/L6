# -*- coding: utf-8 -*-
from __future__ import print_function
import pandas as pd # pandas、csv、numpy是读取文件或处理数组等工具包
import csv
import numpy as np
import time # 获取时间

from sklearn.preprocessing import LabelBinarizer

from Segment_ import Seg #Segment自己编写的数据预处理模块，包含分词等功能
import gensim # 从gensim工具包中导入Word2Vec工具包
from gensim.models.word2vec import LineSentence
from gensim.models import Word2Vec
from sklearn import svm # sklearn工具包导入支持向量机算法
from sklearn.model_selection import train_test_split #从sklearn工具包导入数据集划分工具
from sklearn.metrics import confusion_matrix #从sklearn工具包导入评价指标：混淆矩阵和f1值
from classification_utilities import display_cm #给混淆矩阵加表头
import joblib #储存或调用模型时使用
import multiprocessing #多进程模块
import PySimpleGUI as sg # gui工具包
import codecs
from sklearn.metrics import plot_roc_curve
import matplotlib.pyplot as plt
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc
import warnings #忽略告警
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')


def read_data():
    wds = Seg()
    # 分词后的数据集存放在data文件夹中data.seg.txt里
    target = codecs.open('./data/data.seg.txt', 'w', encoding='utf8')
    # 待分词文档导入
    with open('./data/data.txt',encoding='utf8') as f:
        line = f.readline()
        # 逐行进行分词处理
        while line:
            seg_list = wds.cut(line, cut_all=False)
            line_seg = ' '.join(seg_list)
            if len(line_seg)<50:
                pass
            else:
                target.writelines(line_seg)
            line = f.readline()
        f.close()
        target.close()

# 返回特征词向量
def getWordVecs(wordList, model):
    vecs = []
    for word in wordList:
        word = word.replace('\n', '')
        try:
            vecs.append(model[word])
        except KeyError:
            continue
    return np.array(vecs, dtype='float')

def buildVecs(data,model):
    label = []
    fileVecs = []
    for line in data:
        wordList = line.split(' ')
        vecs = getWordVecs(wordList, model)
        if len(vecs) > 0:
            vecsArray = sum(np.array(vecs)) / len(vecs)  # mean
            fileVecs.append(vecsArray)
            # print(vecsArray)
            label.append(line[0])
    return fileVecs,label
def get_data_wordvec():
    # inp为输入语料,outp为word2vec的vector格式
    inp = './data/data.seg.txt'
    f = codecs.open(inp, mode='r', encoding='utf-8')
    line = f.readlines()

    data = []
    for i in line:
        data.append(i)
    f.close()
    return data


def word2vec_():
    # inp为输入语料,outp为word2vec的vector格式
    inp = './data/data.seg.txt'
    outp = './data/data.seg.text.vector'
    f = codecs.open(inp, mode='r', encoding='utf-8')
    line = f.readlines()

    data = []
    for i in line:
        data.append(i)
    f.close()

    # 训练skip-gram模型
    model_ = Word2Vec(LineSentence(inp), vector_size=100, window=5, min_count=5, workers=multiprocessing.cpu_count())
    model_.wv.save_word2vec_format(outp, binary=False)
    Input22,label = buildVecs(data, model_)

    X = Input22[:]
    df_x = pd.DataFrame(X)
    df_y = pd.DataFrame(label)
    data = pd.concat([df_y, df_x], axis=1)
    # 将结果保存在data.csv文件里面
    data.to_csv('./data/word2vec.csv')

def classification_():
    df = pd.read_csv('./data/word2vec.csv')
    # 读取标签
    y = df.iloc[:, 1]
    # 标签对应的情感
    labels = ["happy", "sad", "disgust", "anger", "fear", "surprise"]
    # 读取数据
    x = df.iloc[:, 2:]
    # 将训练集划分训练、验证两部分
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
    print('support vector machine')
    clf = svm.SVC(kernel='rbf', C=100, probability=True)
    clf.fit(X_train, y_train)
    joblib.dump(clf, "model.m")
    print('Confusion Matrix')
    cv_conf = confusion_matrix(y_test, clf.predict(X_test))
    display_cm(cv_conf, labels, display_metrics=True, hide_zeros=False)

    print('accuracy: %.2f' % clf.score(x, y))
    lb = LabelBinarizer()
    y_test_bin = lb.fit_transform(y_test)

    # 绘制 ROC 曲线
    plt.figure(figsize=(8, 6))
    for i in range(len(labels)):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], clf.predict_proba(X_test)[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label='ROC curve for class {} (AUC = {:.2f})'.format(labels[i], roc_auc))

    plt.plot([0, 1], [0, 1], 'k--', lw=2)  # 对角线
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Multi-Class ROC Curve')
    plt.legend(loc="lower right")
    plt.show()
    print('..................................')

def predict_(a):
    inp = './data/data.seg.text.vector'

    model = gensim.models.KeyedVectors.load_word2vec_format(inp, binary=False)
    wds = Seg()
    seg_list = wds.cut(a, cut_all=False)
    # print(11,seg_list)
    line_seg = ' '.join(seg_list)
    line_seg = line_seg.split(' ')
    vecs = getWordVecs(line_seg, model)
    # print(vecs)
    if len(vecs) > 0:
        vecsArray = sum(np.array(vecs)) / len(vecs)  # mean
        clf = joblib.load("model.m")
        vecsArray = vecsArray.reshape(1, 100)
        kk = clf.predict(vecsArray)

        if kk == [0]:
            return "express happy"
        if kk == [1]:
            return "express sad"
        if kk == [2]:
            return "express disgust"
        if kk == [3]:
            return "express anger"
        if kk == [4]:
            return "express fear"
        if kk == [5]:
            return "express surprise"

def read_table_data(filename):
    with open(filename, "r", encoding='gbk') as infile:
        reader = csv.reader(infile)
        data = list(reader)  # read everything else into a list of rows
    return data

def make_window(theme):
    sg.theme(theme)

    menu_def = [['Help', ['About...', ['hello']]], ]

    News_detection = [
        [sg.Menu(menu_def, tearoff=True)],
        [sg.Text('')],
        [sg.Multiline(s=(60, 20), key='_INPUT_news_', expand_x=True)],
        [sg.Text('')],
        [sg.Text('', s=(12)), sg.Text('recognition results：', font=("Helvetica", 15)),
         sg.Text('     ', key='_OUTPUT_news_', font=("Helvetica", 15))],
        [sg.Text('')],
        [sg.Text('', s=(12)), sg.Button('recognize', font=("Helvetica", 15)), sg.Text('', s=(10)),
         sg.Button('Delete', font=("Helvetica", 15)),
         sg.Text('', s=(4))],
        [sg.Text('')],
        [sg.Sizegrip()]
    ]


    News_management = [
        [sg.Table(values=read_table_data('table_data.csv')[1:][:], headings=['text content', 'recognition time', 'recognition results'], max_col_width=30,
                  auto_size_columns=True,
                  display_row_numbers=False,
                  justification='center',
                  num_rows=20,
                  key='-TABLE_de-',
                  selected_row_colors='red on yellow',
                  enable_events=True,
                  expand_x=True,
                  expand_y=True,
                  vertical_scroll_only=False,
                  enable_click_events=True,  # Comment out to not enable header and other clicks
                  )
         ],

        [sg.Button('Delete selected result', font=("Helvetica", 15)), sg.Button('View recognition results', font=("Helvetica", 15))],
        [sg.Sizegrip()]

    ]
    empty = []

    layout = [[sg.MenubarCustom(menu_def, key='-MENU-', font='Courier 15', tearoff=True)],
              [sg.Text('emotion detection system', size=(50, 1), justification='center', font=("Helvetica", 16),
                       relief=sg.RELIEF_RIDGE, k='-TEXT HEADING-', enable_events=True, expand_x=True)]]
    layout += [[sg.TabGroup([[
        sg.Tab(' text detection ', News_detection),
        sg.Tab('                                                     ', empty),
        sg.Tab(' result management  ', News_management,element_justification="right",)]], expand_x=True, expand_y=True,font=("Helvetica", 16)),

    ]]
    # layout[-1].append(sg.Sizegrip())
    window = sg.Window('emotion detection system', layout,
                       right_click_menu_tearoff=True, grab_anywhere=True, resizable=True, margins=(0, 0),
                       use_custom_titlebar=True, finalize=True, keep_on_top=True)
    window.set_min_size(window.size)
    return window

def main_WINDOW():

    window = make_window(sg.theme())
    while True:
        event, values = window.read(timeout=100)

        if event in (None, 'Exit'):
            print("[LOG] Clicked Exit!")
            break

        elif event == 'recognize':
            kk = predict_(values['_INPUT_news_'])
            time2 = time.strftime('%Y-%m-%d %H:%M:%S')
            newuser = [values['_INPUT_news_'], time2, kk]
            with open('./data/table_data.csv', 'a', newline='') as studentDetailsCSV:
                writer = csv.writer(studentDetailsCSV, dialect='excel')
                writer.writerow(newuser)
            window['_OUTPUT_news_'].update(kk)
            window["-TABLE_de-"].update(values=read_table_data('./data/table_data.csv')[1:][:])

        elif event == 'delete':
            window['_OUTPUT_news_'].update(' ')
            window['_INPUT_news_'].update('')
        elif event == 'View recognition results':
            window["-TABLE_de-"].update(values=read_table_data('./data/table_data.csv')[1:][:])

        elif event == 'Delete selected result':
            data = pd.read_csv('./data/table_data.csv', encoding='gbk')
            data.drop(data.index[int(values['-TABLE_de-'][0])], inplace=True)

            data.to_csv("./data/table_data.csv", index=None, encoding="gbk")

            window["-TABLE_de-"].update(values=read_table_data('./data/table_data.csv')[1:][:])

    window.close()
    exit(0)

if __name__ == '__main__':
    read_data()
    #word2vec_()
    #classification_()
    #

    sg.theme()
    main_WINDOW()