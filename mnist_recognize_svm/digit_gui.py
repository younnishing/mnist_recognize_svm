#-*- coding: utf-8 -*-

import wx #wx 是 wxPython 库，用于创建 GUI 程序
from collections import namedtuple #collections.namedtuple 是一个命名元组，用于创建具有命名字段的元组
from PIL import Image #PIL.Image 是 Python Imaging Library (PIL) 的一部分，用于处理图像
import os #os 是用于与操作系统进行交互的库
from model_svm import Tester #model_svm.Tester 是一个自定义的 SVM 模型类

#获取当前工作目录，并定义了一个文件选择对话框中的文件类型筛选条件
origin_path = os.getcwd()
wildcard ="png (*.png)|*.png|" \
           "jpg(*.jpg) |*.jpg|"\
           "jpeg(*.jpeg) |*.jpeg|"\
           "tiff(*.tif) |*.tiff|"\
           "All files (*.*)|*.*"

class MainWindow(wx.Frame): #定义了一个 MainWindow 类，继承自 wx.Frame，用于创建主窗口
    def __init__(self,parent,title):
        # 在 __init__ 函数中，进行了一些初始化操作，包括创建主窗口，并设置了窗口的标题和大小
        # 参数self表示当前创建的MainWindow类的实例对象。
        # parent 表示该窗口的父级窗口，如果为 None，则表示该窗口是应用程序的主窗口。

        wx.Frame.__init__(self,parent,title=title,size=(600,-1))
        # 创建了一个字体对象 static_font，用于设置按钮的字体样式
        # 设置了字体的大小、族类、样式、粗细
        static_font = wx.Font(12, wx.DEFAULT, wx.NORMAL, wx.NORMAL)

        # 定义了一个命名元组 Size，用于设置按钮的尺寸
        Size = namedtuple("Size",['x','y'])
        s = Size(100,100)
        # 指定了 SVM 模型文件的路径
        # os.path.join(): 将多个路径片段拼接成一个完整的路径
        model_path = os.path.join(origin_path,'mnist_svm.m')

        # 初始化了 fileName 和 model 两个变量
        self.fileName = None
        # 创建了一个名为 Tester 的对象，并将 model_path 作为参数传递给 Tester 的构造函数
        self.model = Tester(model_path)
        # 定义了两个按钮的标签和提示文字
        b_labels = [u'选择图片', u'识别数字']
        TipString = [u'选择图片', u'识别数字']

        # 定义了两个回调函数的列表
        funcs = [self.choose_file,self.run]
        
        '''create input area'''
        # 创建输入区域和输出区域，使用了 wx.TextCtrl构造函数，创建一个具有指定尺寸的文本输入控件
        # in1 是用于显示选择的文件路径的文本框，大小为 (5*s.x, s.y)
        # -1：表示控件的标识符，通常使用 -1 表示自动生成一个新的标识符。
        # out1 是用于显示识别结果的文本框，大小为 (s.x, 3*s.y)
        self.in1 = wx.TextCtrl(self,-1,size = (5*s.x,s.y))
        self.out1 = wx.TextCtrl(self,-1,size = (5*s.x,s.y))

        '''create button'''
        # 创建按钮区域，使用了wx.FlexGridSizer布局管理器，用于控制界面元素的位置和大小
        # 设置了1列的网格布局，按钮之间的水平间隔为10垂直间隔为 0
        # 将输入文本框in1添加到布局中
        self.sizer0 = wx.FlexGridSizer(cols=1, hgap=10, vgap=0)
        self.sizer0.Add(self.in1)

        # 创建两个按钮，并添加到布局中
        buttons = []
        for i,label in enumerate(b_labels):
            b = wx.Button(self, id = i,label = label,size = (1.5*s.x,0.3*s.y))
            buttons.append(b)
            self.sizer0.Add(b)

         # 将输出文本框 out1 添加到布局中
        self.sizer0.Add(self.out1)

        '''set the color and size of labels and buttons'''
        # 为按钮设置字体颜色、大小和提示文字，并绑定对应的回调函数
        for i,button in enumerate(buttons):
            button.SetForegroundColour('red')
            button.SetFont(static_font)
#            button.SetToolTipString(TipString[i]) #wx2.8
            button.SetToolTip(TipString[i])   #wx4.0
            button.Bind(wx.EVT_BUTTON,funcs[i])

        '''layout'''
        # 将布局应用到主窗口上，并自动调整布局大小以适应窗口
        self.SetSizer(self.sizer0)
        self.SetAutoLayout(1)
        self.sizer0.Fit(self)

        # 创建状态栏，并显示主窗口
        self.CreateStatusBar()
        self.Show(True)
    
    def run(self,evt):
        # run 函数是点击 "run" 按钮后的回调函数
        # 判断是否选择了图片文件，如果没有，则弹出一个警告对话框
        if self.fileName is None:
            self.raise_msg(u'请选择一幅图片')
            return None
        # 否则，使用 SVM 模型对图片进行数字识别，并将结果显示在输出文本框中
        else:
            ans = self.model.predict(self.fileName)
            self.out1.Clear()
            self.out1.write(str(ans))

    # choose_file 函数是点击 "open" 按钮后的回调函数
    def choose_file(self,evt):
        '''choose img'''
        # 创建一个文件选择对话框
        dlg = wx.FileDialog(
            # 设置对话框的标题、默认目录和文件类型筛选条件
            self, message="请选择一幅图片",
            defaultDir=os.getcwd(), 
            defaultFile="",
            wildcard=wildcard,
#            style=wx.OPEN | wx.MULTIPLE | wx.CHANGE_DIR #wx2.8
            style = wx.FD_OPEN | wx.FD_MULTIPLE |     #wx4.0
                    wx.FD_CHANGE_DIR | wx.FD_FILE_MUST_EXIST |
                    wx.FD_PREVIEW
            )
        if dlg.ShowModal() == wx.ID_OK:
            paths = dlg.GetPaths()
            dlg.Destroy()
            # 将选择的文件路径赋值给 fileName 变量，并打开该图片文件进行预览展示
            self.in1.Clear()
            self.in1.write(paths[0])
            self.fileName = paths[0]
            im = Image.open(self.fileName)
            im.show()
        else:
            return None

    # raise_msg 函数用于弹出一个警告对话框，并显示指定的警告消息。
    def raise_msg(self,msg):
        '''warning message'''
        info = wx.AboutDialogInfo()
        info.Name = "Warning Message"
        info.Copyright = msg
        wx.AboutBox(info)
    # 在程序的入口处，创建了一个 wx.App 应用程序对象,并进入主事件循环，以处理用户交互和界面更新
if __name__ == '__main__':
    app = wx.App(False)
    frame = MainWindow(None,'手写数字识别')
    app.MainLoop()
