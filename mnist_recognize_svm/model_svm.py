# -*- coding: utf-8 -*-
"""
Created on Sun May  5 11:16:16 2019

@author: WellenWoo
"""
import numpy as np # numpy：用于数值计算。
import os # os：用于文件和目录操作。
from PIL import Image # PIL 中的 Image：用于处理图像。
from sklearn.svm import SVC
# sklearn.svm 中的 SVC：支持向量机分类器。
from sklearn.externals import joblib
# sklearn.externals 中的 joblib：用于模型的保存和加载。
from sklearn.metrics import confusion_matrix, classification_report
# sklearn.metrics 中的 confusion_matrix 和 classification_report：用于评估分类器性能。
import glob # glob：用于文件匹配。
import time

class DataLoader(object):
    """训练前的预处理"""
    def get_files(self, fpath, fmt = "*.png"):
        #获取指定文件夹中指定格式的文件列表;

        tmp = os.path.join(fpath,fmt)
        # os.path.join 函数用于将文件夹路径 fpath 和文件格式 fmt 连接成一个完整的文件路径。
        fs = glob.glob(tmp)
        #glob.glob 函数用于在文件系统中查找匹配指定模式的文件，并返回符合条件的文件列表。
        #glob.glob(tmp) 将会查找文件夹 fpath 中所有符合文件格式为 *.png 的文件，并将它们的路径存储在 fs 列表中。
        return fs

    def get_data_labels(self, fpath="train"):
        # get_data_labels 方法：获取训练数据的图像和标签。

        # 获取指定文件夹下的所有子文件夹的路径
        paths = glob.glob(fpath + os.sep + "*")
        X = []  # 用于存储图像数据
        y = []  # 用于存储标签数据

        # 遍历每个子文件夹的路径
        for fpath in paths:
            # 获取子文件夹中的文件列表
            fs = self.get_files(fpath)

            # 遍历子文件夹中的每个文件
            for fn in fs:
                # 将图像文件转换为向量，并添加到 X 列表中
                X.append(self.img2vec(fn))
                #self.img2vec(fn)：这部分代码调用了 DataLoader 类的 img2vec 方法，该方法用于将指定文件名 fn 的图像文件转换为一个向量。返回的结果是一个一维数组（向量），代表了图像的像素值。这个向量通常会包含图像的每个像素的数值。

            # 为当前子文件夹中的图像添加标签，标签的值为子文件夹的名称，重复 len(fs) 次
            label = np.repeat(int(os.path.basename(fpath)), len(fs))
            y.append(label)

        labels = y[0]  # 初始化 labels 变量为第一个子文件夹的标签

        # 遍历其他子文件夹的标签，将它们追加到 labels 中
        for i in range(len(y) - 1):
            labels = np.append(labels, y[i + 1])

        # 返回图像数据 X 和对应的标签 labels
        return np.array(X), labels

    def img2vec(self, fn):
        '''将jpg等格式的图片转为向量'''
        im = Image.open(fn).convert('L') # 使用PIL库中的Image.open()方法打开图像文件，然后将其转换为灰度图像（'L'表示灰度模式）
        im = im.resize((28, 28)) # 调整图像的大小为28x28像素，通常这是手写数字图像的标准大小
        tmp = np.array(im) # 使用NumPy库将图像数据转换为一个多维数组
        vec = tmp.ravel() # 使用.ravel()方法将多维数组（矩阵）展平成一维数组，这个一维数组即为图像的向量表示

        return vec # 返回图像的向量表示

    def save_data(self, X_data, y_data, fn = "mnist_train_data"):
        """将数据保存到本地;"""
        np.savez_compressed(fn, X = X_data, y = y_data)
        
    def load_data(self, fn = "mnist_train_data.npz"):
        """从本地加载数据;"""
        data = np.load(fn)
        X_data = data["X"]
        y_data = data["y"]
        return X_data, y_data


class Trainer(object):
    '''训练器;'''
    def svc(self, x_train, y_train):
        #svc 方法用于构建一个支持向量机（SVM）分类器。x_train 是训练数据的特征（图像向量）。y_train 是训练数据的标签。
        model = SVC(kernel = 'poly',degree = 4,probability= True)
        # kernel = 'poly：使用多项式核函数。'
        # 'degree = 4：多项式核函数的阶数为4。'
        # 'probability = True：启用概率估计，以便在后续的分类概率预测中使用。'
        model.fit(x_train, y_train)
        # 使用 model.fit(x_train, y_train) 对模型进行训练，其中 x_train 是特征数据，y_train 是标签数据。
        return model
        
    def save_model(self, model, output_name):
        '''保存模型'''
        joblib.dump(model,output_name, compress = 1)
        # 函数将模型以二进制格式保存到本地文件，以便在以后加载和使用。

    def load_model(self, model_path):
        '''加载模型'''
        clf = joblib.load(model_path) # 函数从指定文件路径加载模型，然后返回加载的模型对象。
        return clf

class Tester(object):
    '''测试器;'''
    def __init__(self, model_path):
        trainer = Trainer()      
        self.clf = trainer.load_model(model_path) #加载已保存的模型，并将其赋值给 self.clf，以便在后续方法中使用。
        
    def clf_metrics(self,X_test,y_test):
        #评估分类器效果,接受两个参数：X_test 表示测试数据的特征（通常是图像向量），y_test 表示测试数据的真实标签。
        pred = self.clf.predict(X_test) #对测试数据进行预测
        cnf_matrix = confusion_matrix(y_test, pred) #使用 confusion_matrix 函数计算混淆矩阵 cnf_matrix，以评估模型的分类效果。
        score = self.clf.score(X_test, y_test) # 计算模型在测试数据上的准确度
        clf_repo = classification_report(y_test, pred) #函数生成分类报告，其中包括了各种分类性能指标如精确度、召回率、F1分数等。
        return cnf_matrix, score, clf_repo #return 以上三个
    
    def predict(self, fn):
        '''样本预测;'''
        loader = DataLoader()
        tmp = loader.img2vec(fn)
        X_test = tmp.reshape(1, -1)
        ans = self.clf.predict(X_test)
        return ans

def run_train():
    t0 = time.time()
    loader = DataLoader()
    trainer = Trainer()
    
    X, y = loader.get_data_labels()
    t1 = time.time()
    print(t1 - t0)
    clf = trainer.svc(X, y)
    print(time.time() - t1)
    
    trainer.save_model(clf, "mnist_svm.m")
    
    X_test, y_test = loader.get_data_labels("test")
    
    tester = Tester("mnist_svm.m")
    mt, score, repo = tester.clf_metrics(X_test, y_test)
    return clf, X, y