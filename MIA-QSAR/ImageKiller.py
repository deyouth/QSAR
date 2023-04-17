from PIL import Image
import datetime
import numpy as np
import os,pickle 

class MIA:
    def __init__(self):
        
        #后缀名列表,程序仅识别表中存在的后缀名文件
        self.extension = ('.jpg','.jpeg','.gif','.png','.bmp')
        self.dontRead = ('HeatMap.png','Exp_Cal.png','Y_randomization.png')

        #工作目录为当前目录
        file = os.listdir()

        #将目录中的文件按文件名排序
        #注意！不要使用数字与字母混合的文件名！
        fileList1 = []
        fileList2 = []

        for i in file:
            front,back = i.split('.')[0],i.split('.')[1]
            if front.isdigit() == True:
                fileList1.append((int(front),back))
            elif front.isalpha() == True:
                fileList2.append((front,back))

        fileList1 = sorted(fileList1,key = lambda x : x[0])
        fileList1 = [str(a) + '.' + b for (a,b) in fileList1]

        fileList2 = sorted(fileList2)
        fileList2 = [a + '.' + b for (a,b) in fileList2]

        self.files = fileList1 + fileList2

    def loadImage(self,ImgName):
        '''读取图片,转化成矩阵'''
        im = Image.open(ImgName)

        width,height = im.size
        im = im.convert("L") 
        data = im.getdata()
        data = np.matrix(data)/255.0
        
        #新矩阵中空白部分的值是1
        new_data = np.reshape(data,(height,width))

        return new_data

    def findBorder(self,data):
        '''寻找图片非空白部分的边界'''
        for u in range(np.shape(data)[0]):
            if np.mean(data[u,:]) != 1:
                up = u
                break
                    
        for d in range(np.shape(data)[0])[::-1]:
            if np.mean(data[d,:]) != 1:
                down = d
                break

        for l in range(np.shape(data)[1]):
            if np.mean(data[:,l]) != 1:
                left = l
                break
            
        for r in range(np.shape(data)[1])[::-1]:
            if np.mean(data[:,r]) != 1:
                right = r
                break

        #返回非空白部分的上下左右界限
        return up,down,left,right

    def calcSize(self):
        '''依次打开文件,寻找最大的图片边界'''

        maxHeight = 0
        maxWidth = 0
        
        for img in self.files:
            
            #ext为文件后缀名
            if os.path.splitext(img)[1] in self.extension\
               and img not in self.dontRead:
                
                data = self.loadImage(img)

                #去掉图片四周的空白部分
                up,down,left,right = self.findBorder(data)
                
                new_data = data[up:down+1,left:right+1]
                
                #保存图片
                new_data *= 255
                new_im = Image.fromarray(new_data.astype(np.uint8))
                new_im.save(img)
                    
                H,W = np.shape(new_data)

                if H > maxHeight:
                    maxHeight = H

                if W > maxWidth:
                    maxWidth = W
                    
        #返回最大宽度和高度
        return maxHeight,maxWidth

    def unifyImg(self):
        '''依次打开文件,将文件改为统一大小后保存'''
        
        #第一次打开图片,寻找最大边界
        H,W = self.calcSize()

        #第二次打开图片,将所有图片统一大小    
        for img in self.files:
            if os.path.splitext(img)[1] in self.extension \
               and img not in self.dontRead:
                
                new_data = self.loadImage(img)

                #计算长和宽的差值
                H_D = H - np.shape(new_data)[0]
                W_D = W - np.shape(new_data)[1]

                #将空白加入图片
                H_add = np.ones((H_D,np.shape(new_data)[1]))
                new_data = np.vstack((new_data,H_add))

                W_add = np.ones((np.shape(new_data)[0],W_D))
                new_data = np.hstack((new_data,W_add))

                #保存图片
                new_data *= 255
                new_im = Image.fromarray(new_data.astype(np.uint8))
                new_im.save(img)

    def makeRef(self):

        #选择第一个图片文件作为参照
        num = 0
        
        while True:
            if os.path.splitext(self.files[num])[1] not in self.extension \
               or self.files[num] in self.dontRead:
                num += 1
            else:
                ref_data = self.loadImage(self.files[num])
                break

        #返回Ref矩阵
        return ref_data

    def align(self,ref):
        '''将图片与ref按照均值差最小原理对齐'''

        #计算参照图的每行及每列均值
        def calcMean(data):
            I_mean = []
            J_mean = []
            
            for i in range(np.shape(data)[0]):
                I_mean.append(np.mean(data[i,:]))

            for j in range(np.shape(data)[1]):
                J_mean.append(np.mean(data[:,j]))
                
            return I_mean,J_mean

        ref_I,ref_J = calcMean(ref)
        #将ref居中
        ref_I = len(ref_I)*[1] + ref_I + len(ref_I)*[1]
        ref_J = len(ref_J)*[1] + ref_J + len(ref_J)*[1]

        for img in self.files:
            if os.path.splitext(img)[1] in self.extension \
               and img not in self.dontRead:
                img_data = self.loadImage(img)

                #将绝对误差添加到ERO中
                I_ERO = []

                #计算矩阵的每行和每列均值
                img_I,img_J = calcMean(img_data)
                #将均值列表分别拓展为3倍,相当于将图片置于左上角
                H = len(img_I)
                W = len(img_J)
                
                img_I += 2*H*[1]
                img_J += 2*W*[1]
                
                while True:
                    I_error = sum([abs(a - b) for(a,b) in zip(img_I,ref_I)])
                    I_ERO.append(I_error)

                    if img_I[-1] != 1:
                        break
                    last = img_I.pop()
                    img_I.insert(0,last)


                J_ERO = []
                
                while True:
                    J_error = sum([abs(a - b) for(a,b) in zip(img_J,ref_J)])
                    J_ERO.append(J_error)

                    if img_J[-1] != 1:
                        break
                    last = img_J.pop()
                    img_J.insert(0,last)

                I_move = I_ERO.index(min(I_ERO))
                J_move = J_ERO.index(min(J_ERO))

                #真实图片置于左上
                down_add = np.ones((2*H,np.shape(img_data)[1]))
                img_data = np.vstack((img_data,down_add))
                right_add = np.ones((np.shape(img_data)[0],2*W))
                img_data = np.hstack((img_data,right_add))

                H_add = np.ones((I_move,np.shape(img_data)[1]))
                new_data = np.vstack((H_add,img_data[:-I_move,:]))

                W_add = np.ones((np.shape(new_data)[0],J_move))
                new_data = np.hstack((W_add,new_data[:,:-J_move]))

                #保存图片
                new_data *= 255
                new_im = Image.fromarray(new_data.astype(np.uint8))
                new_im.save(img)

    def removeEmpty(self):
        '''去除对齐后图片的空白部分'''
        f_up = []
        f_down = []
        f_left = []
        f_right = []

        #寻找最大边界
        for img in self.files:
            if os.path.splitext(img)[1] in self.extension \
               and img not in self.dontRead: 
                data = self.loadImage(img)
                
                up,down,left,right = self.findBorder(data)
                
                f_up.append(up)
                f_down.append(down)
                f_left.append(left)
                f_right.append(right)
                
        cut_up = min(f_up)
        cut_down = max(f_down)
        cut_left = min(f_left)
        cut_right = max(f_right)

        #将对齐的图片统一大小
        for img in self.files:
            if os.path.splitext(img)[1] in self.extension \
               and img not in self.dontRead: 
                data = self.loadImage(img)

                new_data = data[cut_up:cut_down+1,cut_left:cut_right+1]
                
                #保存图片
                new_data *= 255
                new_im = Image.fromarray(new_data.astype(np.uint8))
                new_im.save(img)

        Height = cut_down - cut_up
        Width = cut_right - cut_left

        #将图片最大长和宽保存
        with open('HW.pkl','wb') as f:
            pickle.dump((Height,Width),f)

        #返回最大的长和宽
        return Height,Width

    def generateMatrix(self,Height,Width):
        '''
        读取图片尺寸,保存坐标
        将所有图片转换为一维向量，然后合并成一个矩阵
        '''
        
        #将位置信息保存于post列表中,每个元素为坐标元组
        post = [(i,j) for i in range(Height + 1) for j in range(Width + 1)]

        #将所有图片的矩阵保存在new_data列表中
        new_data = []
        for img in self.files:
            if os.path.splitext(img)[1] in self.extension \
               and img not in self.dontRead:
                data = self.loadImage(img)
                data_1d = data.reshape(1,-1).tolist()
                new_data += data_1d

        #删除新矩阵中的空列
        data_array = np.array(new_data)
        
        fliter = [j for j in range(np.shape(data_array)[1]) \
                  if np.mean(data_array[:,j]) != 1]
            
        matrix = data_array[:,fliter]
        
        f_post = [post[i] for i in fliter]
        
        with open('MIA_post.pkl','wb') as f:
            pickle.dump(f_post,f)
        
        with open('MIA_data.pkl','wb') as f:
            pickle.dump(data_array[:,fliter],f)                
        
    def run(self):
        print('图片对齐中...')
        
        #去掉图片空白,统一大小,拓展尺寸
        self.unifyImg()

        #利用第一张图片制作一个分子结构式居中的ref矩阵
        ref = self.makeRef()

        #将所有图片与ref对齐
        self.align(ref)

        #去掉图片的多余空白
        H,W = self.removeEmpty()

        #将所有图片合成一个矩阵,保存成文件
        self.generateMatrix(H,W)
    
def runImageKiller():
    start = datetime.datetime.now()
    
    mia = MIA()
    mia.run()
    end = datetime.datetime.now()

    #记录运行时间
    delta_time = (end - start).seconds
    
    h = delta_time//3600
    m = (delta_time - 3600*h) // 60
    s = delta_time - 3600*h - 60*m
    
    if h == 0:
        print('程序运行时间为:%s分,%s秒' % (m,s))

    elif h == 0 and m == 0:
        print('程序运行时间为:%s秒' % s)

    else:
        print('程序运行时间为:%s小时,%s分,%s秒' % (h,m,s))
    
if __name__ == '__main__':
    runImageKiller()
