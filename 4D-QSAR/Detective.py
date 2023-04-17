#!/usr/bin/env
# coding: utf-8

from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold,LeaveOneOut
import matplotlib.pyplot as plt
import numpy as np
import random,math,pickle,os,shutil

test_set = []
sps = []

report = ['\n----- 数据分析 -----\n']
report_en = ['\n----- DATA ANALYSIS -----\n']

def reporter(text,language = 'CN'):
    '''添加报告内容'''
    global report,report_en
    
    if language == 'CN':
        report += [text]
    elif language == 'EN':
        report_en += [text]
    else:
        report += [text]
        report_en += [text]

def haReporter():
    '''保存中英文报告文件'''
    
    str_cn = '\n'.join(report)
    str_en = '\n'.join(report_en)
    
    fileName = 'Result'
    
    with open(fileName + '_CN.txt','w') as f:
        f.write(str_cn)

    with open(fileName + '_EN.txt','w') as f:
        f.write(str_en)

def readFile(X_train,X_test,y_train,y_test):
    '''用参数指定文件名,读取对应的列表,返回对应的矩阵'''

    #读取X数据
    with open(X_train,'rb') as f:
        a = pickle.load(f)
    a = np.array(a)
    
    with open(X_test,'rb') as f:
        b = pickle.load(f)
    b = np.array(b)

    #读取y标签    
    with open(y_train,'rb') as f:
        c = pickle.load(f)
    c = np.array(c)

    with open(y_test,'rb') as f:
        d = pickle.load(f)

    d = np.array(d)

    #返回X训练集,X测试集,y训练集,y测试集
    return a,b,c,d


class LoadResult():
    '''从LQTAgrid的结果文件中读取数据'''
    
    def loadPost(self):
        '''读取格点位置和对应的数据类别'''
        with open('matrix.txt') as f:
            line = f.readlines()

        postList = line[0].split()
        postList = np.array(postList).reshape(1,-1)

        return postList
        
    def loadData(self):
        '''读取所有样本的数据'''
        with open('matrix.txt') as f:
            line = f.readlines()

        dataList = []
        count = 1
        LJcutNum = 0
        CcutNum = 0
        inf = 0
        
        for i in line[1:]:
            print('正在处理样本%s的数据...' % count)
            
            temp = i.split()
            new = []
            
            for j in temp[:int(len(temp)/2)]:
                
                if not np.isfinite(float(j)):
                    new.append(35)
                    inf += 1

                #对LJ值进行截断,截断值30
                elif float(j) > 30:
                    v = round(30 + math.log(float(j) - 29),6)
                    new.append(v)
                    LJcutNum += 1     
                    
                else:
                    new.append(float(j))

            for k in temp[int(len(temp)/2):]:
                
                if not np.isfinite(float(j)):
                    new.append(35)
                    inf += 1

                #对Coulomb值进行截断,截断值|30|
                elif float(k) > 30:
                    big =  round(30 + math.log(float(k) - 29),6)
                    CcutNum += 1
                    new.append(big)
                elif float(k) < -30:
                    small = - round(30 + math.log(-float(k) - 29),6)
                    CcutNum += 1
                    new.append(small)

                else:
                    new.append(float(k))
                    
            dataList.append(new)

            count += 1

        v = len(dataList)
        s = len(dataList[0])
        
        print('初始X数据矩阵:(%s,%s)' % (v,s))
        print('共截断了%s个LJ数值' % LJcutNum)
        print('共截断了%s个Coulomb数值' % CcutNum)
        print('inf的个数为 %s' % inf )
        
        reporter('初始X数据矩阵:(%s,%s)' % (v,s))
        reporter(('The original matrix : (%s,%s)' % (v,s)) ,'EN')

        dataList = np.array(dataList)
        
        return dataList   

    def run(self):
        
        print('\n*** Load Result ***\n')
        reporter('\n*** Load Result ***\n','both')

        postList = self.loadPost()
        dataList = self.loadData()
        

        #描述符方差筛选
        keep = [j for j in range(np.shape(dataList)[1]) if np.var(dataList[:,j]) > 0.01]

        postList = postList[:,keep]
        dataList = dataList[:,keep]

        descriptor = np.shape(dataList)[1]
        print('方差过滤后描述符个数为: %s' % descriptor)
        reporter('方差过滤后描述符个数为: %s' % descriptor)
        reporter('%s descriptors were kept after variance cut-off'\
                 % descriptor,'EN')

        #将格点位置和数据类型列表保存在post.pkl文件中
        with open('post.pkl','wb') as s:
            pickle.dump(postList,s)

        #将数据列表保存在data.pkl文件中
        with open('data.pkl','wb') as v:
            pickle.dump(dataList,v)
            
class Train_Test():
    '''随机分割训练集和测试集'''

    def __init__(self,mission = 'King'):
        self.mission = mission
    
    def read(self):
        #读取X数据
        if self.mission != 'Queen':
            with open('data.pkl','rb') as f:
                X = pickle.load(f)
        
        else:
            
            with open('MIA_data.pkl','rb') as f:
                X = pickle.load(f)

        X = np.array(X)

        if self.mission == 'Royalty':
            
            with open('MIA_data.pkl','rb') as f:
                X2 = pickle.load(f)
                
            X2 = np.array(X2)
            
            #合并4D和MIA描述符
            X = np.hstack((X,X2))

            with open('post.pkl','rb') as f:
                post = pickle.load(f)

            post2 = np.ones((1,np.shape(X2)[1]))
            post = np.hstack((post,post2))
            
            with open('post.pkl','wb') as f:
                pickle.dump(post,f)

        #读取y标签
        with open('act.txt','r') as f:
            y = f.read().split()
            
        for i in range(len(y)):
            y[i] = float(y[i])
            
        y = np.array(y)
        y = y.reshape(-1,1)

        return X,y       
        
    def train_test_split(self,X,y,test_size = 0.2):
        '''X,y为矩阵,随机分割测试集和训练集'''
        
        lines = len(X)
        linesList = list(range(lines))
        num = int(lines * test_size)+1

        #测试集索引号
        if os.path.exists('test.txt') == True:
            #可通过test.txt指定测试集
            with open('test.txt') as f:
                test_file = f.read().split()
            testList = sorted([int(i)-1 for i in test_file])
        else:
            testList = sorted(random.sample(linesList,num))

        for i in testList:
            linesList.remove(i)

        X_train = X[linesList,:]
        X_test = X[testList,:]
        y_train = y[linesList,:]
        y_test = y[testList,:]

        testNum = [str(i+1) for i in testList]

        #多次运行,记录测试集序号
        global test_set
        test_set.append(testNum)

        if self.mission == 'Royalty':
            with open('MIA_data.pkl','rb') as f:
                X2 = pickle.load(f)

            X2 = np.array(X2)

            with open('MIA_data_train.pkl','wb') as f:
                pickle.dump(X2[linesList,:],f)

        print('测试集的样本序号为:\n',testNum)
        reporter('测试集的样本序号为:\n['+ ','.join(testNum)+']')
        reporter('The serial number of test samples:\n['+ ','.join(testNum)+']','EN')
        
        return X_train,X_test,y_train,y_test

    def run(self):
        
        print('\n*** Split Train&Test set ***\n')
        reporter('\n*** Split Train&Test set ***\n','both')
        
        X,y = self.read()
        
        #测试集的比例为20%        
        X_train,X_test,y_train,y_test=\
        self.train_test_split(X,y,test_size=0.2)

        with open('X_train.pkl','wb') as f:
            pickle.dump(X_train,f)

        with open('X_test.pkl','wb') as f:
            pickle.dump(X_test,f)

        with open('y_train.pkl','wb') as f:
            pickle.dump(y_train,f)

        with open('y_test.pkl','wb') as f:
            pickle.dump(y_test,f)

        sample_train = np.shape(X_train)[0]
        sample_test = np.shape(X_test)[0]
        descriptor = np.shape(X_train)[1]
        print('训练集X矩阵 ：', np.shape(X_train))
        print('测试集X矩阵 ：', np.shape(X_test))
        reporter('训练集样本数量： %s' % sample_train)
        reporter('size of TRAIN set： %s' % sample_train,'EN')
        reporter('测试集样本数量： %s' % sample_test)
        reporter('size of TEST set： %s' % sample_train,'EN')
        reporter('描述符数量： %s' % descriptor)
        reporter('There were %s descriptors' % descriptor ,'EN')

class PearsonCorrelation():
    '''皮尔森相关分析（协方差除以两向量的标准差）'''
    def __init__(self,mission = 'King'):
        self.mission = mission
        
    def calcPearson(self,x,y):
        '''计算两向量的pearson相关系数'''
        x_mean,y_mean = np.mean(x),np.mean(y)
        
        n = len(x)
        
        sumTop = 0
        sumBottom = 0
        x_pow = 0
        y_pow = 0

        for i in range(n):
            sumTop += (x[i] - x_mean)*(y[i] - y_mean)
            x_pow += math.pow(x[i] - x_mean,2)
            y_pow += math.pow(y[i] - y_mean,2)
            
        sumBottom = math.sqrt(x_pow * y_pow)
        
        if sumBottom == 0:
            p = [0]
        else:
            p = sumTop/sumBottom
        
        return round(p[0],4)

    def run(self):
        
        print('\n*** Calculate Pearson ***\n')
        reporter('\n*** Calculate Pearson ***\n','both')
        
        #读取矩阵
        a = 'X_train.pkl'
        b = 'X_test.pkl'
        c = 'y_train.pkl'
        d = 'y_test.pkl'
        
        X,X_test,y,y_test = readFile(a,b,c,d)

        print('正在计算Pearson相关系数...')
        
        #将所有描述符的的相关系数加入列表r
        r = [self.calcPearson(X[:,j],y) for j in range(np.shape(X)[1])]

        #随机产生5万个描述符,与y进行相关分析,取99置信区间值为截断值
        rand_vectors = np.random.random((np.shape(y)[0],50000))
        r_rand = [self.calcPearson(rand_vectors[:,j],y) \
                  for j in range(np.shape(rand_vectors)[1])]

        r_cutoff = 2.33 * np.std(r_rand)
        r_cutoff = round(r_cutoff,2)
        
        if r_cutoff < 0.3:
            r_cutoff = 0.3

        print('|r|保留值为：|r| >= %s' % r_cutoff)
        reporter('|r|_cutoff : %s '% r_cutoff + \
                 ' (|r| >= |r|_cutoff 将被保留)')
        reporter('|r|_cutoff : %s '% r_cutoff + \
                 ' (|r| >= |r|_cutoff would be kept)' ,'EN')

        #将相关系数符合条件的描述符加入fliter列表
        fliter = [i for i in range(len(r)) if abs(r[i]) >= r_cutoff]
        
        descriptor = np.shape(X[:,fliter])[1]
        print('|r|过滤后的描述符个数为：%s' % descriptor)
        reporter('|r|过滤后的描述符个数为：%s' % descriptor)
        reporter('%s descriptors were kept after |r| cut-off'\
                 % descriptor , 'EN')
            
        with open('pearson_X_train.pkl','wb') as f:
            pickle.dump(X[:,fliter],f)

        with open('pearson_X_test.pkl','wb') as f:
            pickle.dump(X_test[:,fliter],f)

        #作2D热力图
        if self.mission == 'Royalty':
            
            with open('MIA_data_train.pkl','rb') as f:
                X = pickle.load(f)

            X = np.array(X)

            #计算图片部分Pearson相关系数
            r = [self.calcPearson(X[:,j],y) for j in range(np.shape(X)[1])]

        if self.mission != 'King':
            
            #载入图片大小
            with open('HW.pkl','rb') as f:
                H,W = pickle.load(f)

            #载入位置信息
            with open('MIA_post.pkl','rb') as f:
                post = pickle.load(f)
                                   
            #产生背景矩阵
            bg = np.zeros((H+1,W+1))

            #将相关系数归一化后变换至[-1,1]区间
            min_r = min(r)
            max_r = max(r)
            delta_r = max_r - min_r
            r2 = [2*((i - min_r)/delta_r)-1 for i in r]
            
            #将相关系数写入背景
            for num in range(len(post)):
                #设置截断线,去除噪音
                if abs(r2[num]) >= 0.25:
                    bg[post[num][0],post[num][1]] = r2[num] 

            #产生母环矩阵
            bg2 = np.zeros((H+1,W+1))
            
            #确定母环
            mainRing = []
            for i in range(np.shape(X)[1]):
                if 1 not in X[:,i]:
                    mainRing += [1]
                else:
                    mainRing += [0]
                    
            #将母环结构写入背景
            for num in range(len(post)):
                if mainRing[num] == 1:
                    bg2[post[num][0],post[num][1]] = 1
                    
            #作热力图
            plt.figure()
            
            #plt.subplot(2,1,1)
            #plt.imshow(bg2,cmap=plt.get_cmap('binary'))
            #plt.axis('off')
            
            #plt.subplot(2,1,2)
            plt.imshow(bg,cmap=plt.get_cmap('bwr'))
            plt.axis('off')
            plt.colorbar(orientation = 'horizontal')
            
            plt.savefig('HeatMap.png',dpi = 500,bbox_inches='tight')

            plt.close('all')

        #更新位置信息
        if self.mission != 'Queen':
            with open('post.pkl','rb') as f:
                post = pickle.load(f)

            with open('pearson_post.pkl','wb') as f:
                pickle.dump(post[:,fliter],f)


class CDDA:
    '''按X的描述符和y的分布进行筛选,分布与y极为不一致的描述符将被移除'''

    def __init__(self,mission = 'King'):
        self.mission = mission
        
    def calc_Fk(self,vector):
        #n设为4
        n = 4
        Fk_j = []
        
        for j in range(np.shape(vector)[1]):
            #将矩阵的每一列归一化到[0,1]区间
            xj = vector[:,j]
            delta = xj.max() - xj.min()
            
            if delta != 0:
                #当描述符中数值完全相同时,将该列所有元素改为0
                xj = len(xj) * [0]
                xj = np.array(xj)
            else:
                xj = (xj - xj.min())/delta
                
            xj = xj.reshape(-1,1)

            F = []
            
            #将[0,1]分为2**n个区间
            for k in range(1,2**n+1):
                
                fi = []
                
                for i in range(np.shape(xj)[0]):
                    #统计描述符的值在各个区间的频数
                    xi = xj[i,0]
                    if (xi >= 2**(-n)*(k-1) and xi < 2**(-n)*k)\
                       or (xi == 2**(-n)*k and xi == 1):
                        fi.append(1)
                    else:
                        fi.append(0)
                
                F.append(fi)

            #F(I,K)
            F = np.array(F).T
            
            fk_j = np.sum(F,axis = 0).tolist()
            Fk_j.append(fk_j)

        Fk_j = np.array(Fk_j)

        #返回分布频数矩阵
        return Fk_j

    def calcCDDA(self,X,y):
        
        print('正在计算分布...')
        Fk_j = self.calc_Fk(X)
        Fk_y = self.calc_Fk(y)

        #比较描述符和y的分布频数
        Vk_j = Fk_j - Fk_y

        Ej = []
        for i in range(np.shape(Vk_j)[0]):
            epsilon = sum(abs(Vk_j[i,:]))
            ej = 1 - epsilon/(2*np.shape(X)[0] - 2)
            Ej.append(ej)

        fliter = [j for j in range(len(Ej)) if Ej[j] > 0.5]
        
        return fliter

    def run(self):
        '''按X描述符和y的分布进行筛选'''

        print('\n*** Comparative Distribution Detection Algorithm ***\n')
        reporter('\n*** Comparative Distribution Detection Algorithm ***\n' ,'both')
        
        #读取矩阵
        a = 'pearson_X_train.pkl'
        b = 'pearson_X_test.pkl'
        c = 'y_train.pkl'
        d = 'y_test.pkl'
        
        X,X_test,y,y_test = readFile(a,b,c,d)
        
        fliter = self.calcCDDA(X,y)
        Xnew = X[:,fliter]
        Xnew_test = X_test[:,fliter]
        
        descriptor = np.shape(Xnew)[1]
        print('CDDA保留的描述符个数为：',descriptor)
        reporter('CDDA过滤后的描述符个数为：%s' % descriptor)
        reporter('%s descriptors were kept after CDDA'\
                 % descriptor , 'EN')
        
        with open('flitered_X_train.pkl','wb') as f:
            pickle.dump(Xnew,f)
            
        with open('flitered_X_test.pkl','wb') as f:
            pickle.dump(Xnew_test,f)

        #更新位置信息
        if self.mission != 'Queen':
            with open('pearson_post.pkl','rb') as f:
                post = pickle.load(f)

            with open('flitered_post.pkl','wb') as f:
                pickle.dump(post[:,fliter],f)
    
class Method_OPS():
    '''OPS算法'''
    def __init__(self,mission = 'King'):
        self.mission = mission
    
    def quickSort(self,array):
        '''对数组进行从大到小的快速排序'''
        
        #基线条件
        if len(array) < 2:
            return array
        
        #递归条件
        else:
            #选择最左侧数字为基准值
            baseValue = array[0]
            #由所有小于基准值的元素组成子数组
            less = [m for m in array[1:] if m < baseValue]
            #包括基准值在内的和基准值相等的元素
            equal = [w for w in array if w == baseValue]
            #由所有大于基准值的元素组成的子数组
            greater = [z for z in array[1:] if z > baseValue]
            
        #递归
        return self.quickSort(greater) + equal + self.quickSort(less)

    def transform(self,X,X_test,y,Lv,kind = 'RC',pov = 1):
        '''
        :Lv:最佳主成分数
        :kind:排序依据的信息向量类型
        :pov:保留描述符的百分比(0至1)
        '''
        
        if kind == 'Cor':
            vector = self.correlationVector(X,y)

        elif kind == 'Reg':
            vector = self.regressionVector(X,y,Lv)

        elif kind == 'RC' or kind == 'CR':
            vector1 = self.correlationVector(X,y)
            vector2 = self.regressionVector(X,y,Lv)
            vector = [a * b for (a,b) in zip(vector1,vector2)]

        else:
            print('没有这种向量类型')
        
        #读取vector列表,进行快速排序
        qs = self.quickSort(vector)

        print('正在重新排列矩阵...')

        new = []
        new_test = []
        
        if self.mission != 'Queen':
            with open('flitered_post.pkl','rb') as f:
                post = pickle.load(f)
                
            new_post = []
        
        for j in qs[:int(len(qs)* pov)]:
            #只保留前pov%的描述符
            #(Percentage of variables = 1)
            position = vector.index(j)
            new.append(X[:,position])
            new_test.append(X_test[:,position])
            
            if self.mission != 'Queen':
                new_post.append(post[:,position])
                                
            vector[position] = 'read'          
            
        #按顺序排列的新矩阵
        new = np.array(new).T
        new_test = np.array(new_test).T

        if self.mission != 'Queen':
            new_post = np.array(new_post).T
                                
            with open('flitered_post.pkl','wb') as f:
                pickle.dump(new_post,f)

        return new,new_test

    def k_fold(self,X,y,Lv,k = 10):
        '''
        K-Fold交叉验证，用全部数据进行PLS回归，
        选出最佳主成分数，用于OPS筛选信息向量
        '''
        
        #将样本集分为k份
        #未设置洗牌，即每次运行的分组情况都一样
        kf = KFold(n_splits = k,shuffle = False)
        
        temp = []
        
        for train,test in kf.split(X):
            trainX,testX = X[train],X[test]
            trainY,testY = y[train],y[test]

            pls = PLSRegression(n_components = Lv)
            pls.fit(trainX,trainY)
            pred_test = pls.predict(testX)
            
            RMSE = np.sqrt(np.mean(np.square(
                pred_test - testY)))

            temp.append(RMSE)

        return temp

    def regressionVector(self,X,y,Lv):
        '''计算所有描述符的PLS回归相关性'''

        print('正在计算回归相关性...')

        pls = PLSRegression(n_components = Lv)
        pls.fit(X,y)

        W = pls.x_weights_
        P = pls.x_loadings_
        Q = pls.y_loadings_
        
        trans = lambda a : np.matrix(a)

        W = trans(W)
        P = trans(P)
        Q = trans(Q).reshape(-1,1)

        b = W * (P.T * W).I * Q

        b = b.reshape(1,-1).tolist()
        regVector = [round(i,4) for i in sum(b,[])]

        return regVector
    
    def correlationVector(self,X,y):
        '''计算所有描述符的Pearson相关系数'''
        
        print('\n正在计算Pearson相关系数...')
        
        pr = PearsonCorrelation()
        
        corVector = []

        for i in range(np.shape(X)[1]):
            ps = pr.calcPearson(X[:,i],y)
            corVector.append(abs(ps))

        return corVector
    
    def run(self,sortVector = 'Cor',window = 0.05,increment = 0.01):
        '''使用PLS建模,对排序后的信息向量进行筛选'''

        print('\n*** Ordered Predictors Selection ***\n')
        reporter('\n*** Ordered Predictors Selection ***\n' ,'both')

        if sortVector == 'RC' or sortVector == 'CR':
            reporter('informative vector : Reg. & Cor.','both')

        elif sortVector == 'Reg':
            reporter('informative vector : Reg.','both')

        elif sortVector == 'Cor':
            reporter('informative vector : Cor.','both')
        
        #读取矩阵
        a = 'flitered_X_train.pkl'
        b = 'flitered_X_test.pkl'
        c = 'y_train.pkl'
        d = 'y_test.pkl'
        
        X,X_test,y,y_test = readFile(a,b,c,d)
        
        #将X数据标准化
        X_mean = np.std(X,axis = 0)
        X_std = np.std(X,axis = 0)

        X_test = X_test - X_mean
        X_test = X_test / X_std
        
        X -= X_mean
        X /= X_std
        
        #OPS选取建模主成分数用的RMSECV列表
        RMSECV_n = []

        #分别得出1-10个主成分的RMSECV
        for n in range(10):
            temp = self.k_fold(X,y,Lv = n+1)
            RMSECV_n.append(round(np.mean(temp),4))

        #hMod为K-Fold选出的建模最佳主成分数
        hMod = RMSECV_n.index(min(RMSECV_n)) + 1

        print('OPS建模所使用的最佳主成分数为: %s' % hMod)
        reporter('hMod : %s' % hMod ,'both')

        def OPS(h,X,X_test,y,sortVector):
            '''
            OPS排序后的筛选过程
            :h:产生regVector的最佳主成分数
            '''
            #new为重排后的X矩阵
            new,new_test = self.transform(X,X_test,y,Lv = h,kind = sortVector)

            #OPS筛选描述符用的RMSECV列表
            f_RMSECV = []

            open_window = window

            #以前window%列建模,每次循环加入increment%列
            while open_window <= 1:
                X = new[:,:int(np.shape(new)[1] * open_window)]
                
                f_temp = self.k_fold(X,y,Lv = h)
                f_RMSECV.append(round(np.mean(f_temp),4))
                
                open_window += increment

            #保留描述符的百分比
            precent_vector = window + f_RMSECV.index(min(f_RMSECV)) * increment
            vector = int(np.shape(new)[1] * precent_vector)

            return new,new_test,vector,min(f_RMSECV)

        #regVector需要进行OPS过程,计算得出hOPS
        if sortVector == 'RC' or sortVector == 'CR' or sortVector == 'Reg':
            
            RMSE = []
            dataOPS = []

            #主成分数从hMod逐渐上升至10
            for n in range(hMod,11):
                total = OPS(n,X,X_test,y,sortVector)
                RMSE.append(total[3])
                dataOPS.append(total)

            #采用RMSE最小的主成分数
            hOPS = RMSE.index(min(RMSE))
            print('OPS最终筛选使用的最佳主成分数为:',hMod + hOPS)
            reporter('hOPS: %s' % (hMod + hOPS) ,'both')
            
            new,new_test,vector,f_RMSE = dataOPS[hOPS]

        elif sortVector == 'Cor':
            new,new_test,vector,f_RMSE = OPS(hMod,X,X_test,y,sortVector)

        #判断描述符是否全保留
        if RMSECV_n[hMod - 1] < f_RMSE:
            vector = int(np.shape(new)[1])
            print('\nOPS保留了所有描述符')
            reporter('OPS保留了所有描述符')
            reporter('all of descriptors were kept after OPS' ,'EN')
            
        else:
            print('\nOPS选择前%s列用于正式模型构建' % vector)
            reporter('OPS选择前%s列用于正式模型构建' % vector)
            reporter('%s descriptors were kept after OPS' % vector ,'EN')
            

        with open('OPS_X_train.pkl','wb') as f:
            pickle.dump(new[:,:vector],f)

        with open('OPS_X_test.pkl','wb') as f:
            pickle.dump(new_test[:,:vector],f)

        #更新位置信息
        if self.mission != 'Queen':
            with open('flitered_post.pkl','rb') as f:
                post = pickle.load(f)

            with open('OPS_post.pkl','wb') as f:
                pickle.dump(post[:,:vector],f)
        
class Modeling():
    '''PLS回归建模'''
    def __init__(self,mission = 'King'):
        self.mission = mission
    
    def calcSP(self,exp,pred):
        '''计算统计量'''
        #exp,pred都是列表
        a = np.array(exp).reshape(-1,1)
        b = np.array(pred).reshape(-1,1)
        
        R2 = 1 - (np.sum(np.square(a - b))/np.sum(np.square(a - np.mean(a))))

        RMSE = np.sqrt(np.mean(np.square(a - b)))

        R2 = round(R2,3)
        RMSE = round(RMSE,3)
        
        #返回R2和RMSE
        return R2,RMSE

    def Leave1Out(self,X,y,n_c):
        '''留一法交叉验证'''
        loo = LeaveOneOut()
        
        epY = []
        ptY = []
        for train,test in loo.split(X):
            trainX,testX = X[train],X[test]
            trainY,testY = y[train],y[test]
            
            pls = PLSRegression(n_components = n_c)
            pls.fit(trainX, trainY)
            pred_test = pls.predict(testX)
            ptY.append(round(pred_test[0][0],3))
            epY.append(testY.tolist())

        for i in range(len(epY)):
            epY[i] = epY[i][0]

        #返回实验y值和预测y值
        return epY,ptY

    def find_N(self,X,y):
        '''使用训练集进行交叉验证,寻找最佳主成分数'''
        
        nQ2 = []
        nRMSECV = []

        #最大主成份数取10
        for n in range(10):
    
            #将训练集预测结果保存在列表中,用于计算交叉验证系数
            
            print('正在计算%s个主成分' % (n+1))

            epY,ptY = self.Leave1Out(X,y,n_c = n+1)

            Q2,RMSECV = self.calcSP(epY,ptY)
            nQ2.append(Q2)
            nRMSECV.append(RMSECV)

        best_n = nQ2.index(max(nQ2)) + 1
        RMSECV = nRMSECV[best_n - 1]

        return best_n,max(nQ2),RMSECV

    def drawImg(self,exp_train,pre_train,exp_test,pre_test):
        '''使用matplotlib作实验值vs预测值图'''
        v1 = exp_train
        v2 = pre_train
        v3 = exp_test
        v4 = pre_test
        
        plt.figure()
        
        train = plt.scatter(v1,v2,marker = '2',c = 'b')
        test = plt.scatter(v3,v4,marker = '*',c = 'r')

        #添加图例
        plt.legend([train,test],['Train','Test'])

        #添加坐标轴标签
        plt.xlabel('Experimental')
        plt.ylabel('Calculated')
        
        #设置坐标轴范围
        axis_min = min([min(v1),min(v2),min(v3),min(v4)])
        axis_max = max([max(v1),max(v2),max(v3),max(v4)])

        d = [0,0.25,0.5,0.75,1]
        
        for i in range(4):
            if d[i] < axis_min - int(axis_min) < d[i+1]:
                axis_min = int(axis_min) + d[i]
            if d[i] < axis_max - int(axis_max) < d[i+1]:
                axis_max = int(axis_max) + d[i+1]

        plt.xlim(axis_min,axis_max)
        plt.ylim(axis_min,axis_max)

        #保存图片
        plt.savefig('Exp_Cal.png',dpi = 300,bbox_inches='tight')
        
        plt.close('all')

    def draw4Dmap(self,X,y):
        '''将描述符信息生成一个gro文件，用VMD打开'''
        
        with open('OPS_post.pkl','rb') as f:
            post = pickle.load(f)
            post = sum(post.tolist(),[])
            
            
            post_index = [i for i in range(len(post)) if post[i] != '1.0']
            
            post = [post[i] for i in post_index]
            
            new = ''
            pr = PearsonCorrelation()
            
            r = [pr.calcPearson(X[:,j],y) \
                  for j in range(np.shape(X)[1])]
            r = [r[i] for i in post_index]
            
            for i in range(len(post)):
                px,py,pz = post[i].split('_')[0:3]
                pkind = post[i].split('_')[-1].replace(':','')
                re = lambda a:str(float(a)/10)+ '00'
                px,py,pz = re(px),re(py),re(pz)

                #【LJ+ :氧】【LJ- :氮】【C+ :钾】【C- ：硫】
                if pkind == 'LJ' and r[i] > 0:
                    atom_kind = 'O'
                elif pkind == 'LJ' and r[i] < 0:
                    atom_kind = 'N'
                elif pkind == 'C' and r[i] > 0:
                    atom_kind = 'K'
                else:
                    atom_kind = 'S'

                #根据数字位数调整格式
                if i+1 < 10:
                    space1 = ' ' * 5
                    space2 = ' ' * 4
                elif i+1 < 100:
                    space1 = ' ' * 4
                    space2 = ' ' * 3
                elif i+1 < 1000:
                    space1 = ' ' * 3
                    space2 = ' ' * 2
                else:
                    space1 = ' ' * 2
                    space2 = ' ' * 1

                line = '    1MOL'+ space1 + atom_kind + str(i+1) + space2 + str(i+1)\
                      + '   ' + px +'   ' + py +'   '+ pz + '\n'
                
                new += line

            #合并文件内容
            default_str1 = 'MOL in water t=  60.00000 step= 30000\n   '\
                           + str(len(post)) + '\n'
            default_str2 = '1.00000\t1.00000\t1.00000\n'
            new = default_str1 + new + default_str2

            #写入gro文件
            with open('4D_MAP.gro','w') as f:
                f.write(new)
            

    def run(self):
        '''使用最佳主成分数进行PLS回归建模'''

        print('\n*** Modeling ***\n')
        reporter('\n*** Modeling ***\n','both')
        
        #读取矩阵
        a = 'OPS_X_train.pkl'
        b = 'OPS_X_test.pkl'
        c = 'y_train.pkl'
        d = 'y_test.pkl'
        
        X,X_test,y,y_test = readFile(a,b,c,d)

        #作4D_map
        if self.mission != 'Queen':
            self.draw4Dmap(X,y)

        best_n,highest_Q2,RMSECV = self.find_N(X,y)
        
        print('最佳主成分数为: %s' % best_n)
        print('Q2值为: %s ' % highest_Q2)
        print('RMSECV为: %s' % RMSECV)

        reporter('最佳主成分数 : %s' % best_n)
        reporter('the best N components : %s' % best_n ,'EN')
        reporter('Q^2(LOO) : %s' % highest_Q2 ,'both')
        reporter('RMSECV : %s' % RMSECV ,'both')
        
        def regression(X,y,X_test,y_test,draw = True,y_scrambling = False):
            pls = PLSRegression(n_components = best_n)
            pls.fit(X,y)
            pred_train = pls.predict(X)
            pred_test = pls.predict(X_test)

            #计算R2_pred,注意分母中的均值使用训练集的
            R2_pred = 1 - (np.sum(np.square(y_test - pred_test))\
                           /np.sum(np.square(y_test - np.mean(y))))
            R2_pred = round(R2_pred,3)

            #将预测值和实验值统一转换为列表
            array2list = lambda x : list(x.reshape(1,-1)[0])
                    
            pred_train = array2list(pred_train)
            pred_test = array2list(pred_test)
            y = array2list(y)
            y_test = array2list(y_test)

            #obs为实验值,pred为预测值
            obs = y + y_test
            pred = pred_train + pred_test

            #计算统计量
            R2,RMSEC = self.calcSP(y,pred_train)
            RMSEP = self.calcSP(y_test,pred_test)[1]

            if draw == True:
                print('R2为：',R2)
                print('R2_pred为：',R2_pred)
                print('RMSEC为：' ,RMSEC)                
                print('RMSEP为：',RMSEP)

                reporter('R^2 : %s' % R2 ,'both')
                reporter('R^2 _pred : %s' % R2_pred ,'both')
                reporter('RMSEC : %s' % RMSEC ,'both')
                reporter('RMSEP : %s' % RMSEP ,'both')

                #用pyecharts绘制“实验值-预测值”图
                self.drawImg(y,pred_train,y_test,pred_test)

            #计算残差百分比
            residuals = [round((y_test[i] - pred_test[i])*100/y_test[i],1)\
                         for i in range(len(y_test))]

            #报告测试集实验值及预测值
            if y_scrambling == False:
                reporter('\n-------- 测试集实验值与预测值表 --------')
                reporter('\n------- Exp. & Pred. of Test set -------','EN')
                reporter('\n实验值\t\t预测值\t\t% 残差')
                reporter('\n Exp.\t\tPred.\t     % residuals','EN')
                reporter('----------------------------------------','both')
                
                for i in range(len(y_test)):
                    output1 = str(y_test[i])+'\t\t'+str(round(pred_test[i],3))+'\t\t'
                    if residuals[i] > 0:
                        output2 = ' '+ str(residuals[i])+'%'
                    else:
                        output2 = str(residuals[i])+'%'
                        
                    reporter(output1 + output2 ,'both')
                reporter('----------------------------------------','both')
            
            return R2,R2_pred
            
        R2,R2_pred = regression(X,y,X_test,y_test)

        def y_scrambling(y,y_test):
            '''随机重排y标签的顺序,建立假标签模型'''

            print('\n*** Y-scrambling ***\n')
            reporter('\n*** Y-scrambling ***\n','both')

            print('正在计算Y-scrambling...')
            
            listQ2 = []
            listR2 = []
            correlations = []
            y = y.tolist()
            pc = PearsonCorrelation()

            #循环50次
            for i in range(50):
                new_y = np.array(y[:])
                
                #将y随机打乱顺序
                np.random.shuffle(new_y)
                
                #用原始y和打乱后的y计算相关系数
                r = abs(pc.calcPearson(np.array(y),new_y))
                correlations.append(round(r,4))
                
                exp,pred = self.Leave1Out(X,new_y,n_c = best_n)
                listQ2.append(self.calcSP(exp,pred)[0])
                listR2.append(regression(X,new_y,X_test,y_test,\
                                         draw = False,y_scrambling = True)[0])

            #用相关性和Q2、R2分别作图，计算截距
            arrayQ2 = np.array(listQ2 + [highest_Q2]).reshape(-1,1)
            arrayR2 = np.array(listR2 + [R2]).reshape(-1,1)
            arrayCor = np.array(correlations + [1]).reshape(-1,1)
            
            lr_Q = LinearRegression().fit(arrayCor,arrayQ2)
            Q2_intercept = round(lr_Q.intercept_[0],2)
            
            lr_R = LinearRegression().fit(arrayCor,arrayR2)
            R2_intercept = round(lr_R.intercept_[0],2)
            
            print('Q2的截距为：',Q2_intercept)
            print('R2的截距为：',R2_intercept)
            reporter('Q^2的截距为：%s' % Q2_intercept)
            reporter('Q^2 intercept：%s' % Q2_intercept ,'EN')
            reporter('R^2的截距为：%s' % R2_intercept)
            reporter('R^2 intercept：%s' % R2_intercept ,'EN')
            
            #作图
            plt.figure()
            permuted = plt.scatter(listR2,listQ2,marker = 'x',c = 'b')
            original = plt.scatter(R2,highest_Q2,marker = '*',c = 'r')

            #添加图例
            plt.legend([permuted,original],['Permuted data model',
                                            'Original data model'])
            
            #添加坐标轴标签
            plt.xlabel(r'$R^2$')
            plt.ylabel(r'$Q^2$')

            #保存图片
            plt.savefig('Y_scrambling.png',dpi = 300,bbox_inches='tight')

            plt.close('all')

            return Q2_intercept,R2_intercept
        
        R2_intercept = y_scrambling(y,y_test)[1]

        #多次运行，记录统计量
        global sps
        sps.append([highest_Q2,R2,R2_pred,R2_intercept])

def moveResult():
    '''将运行结果相关文件保存到Result文件夹下'''

    #如果文件夹已存在,将其删除
    if os.path.exists('Result') == True:
        shutil.rmtree('Result')

    #创建结果文件夹
    os.mkdir('Result')

    #移动文件
    shutil.move('Result_CN.txt','Result')
    shutil.move('Result_EN.txt','Result')
    shutil.move('HeatMap.png','Result')
    shutil.move('Y_scrambling.png','Result')
    shutil.move('Exp_Cal.png','Result')
        
def runDetective():
    #使用mission来选择任务类型(King,Queen,Royalty)
    mission = 'King'
    
    if mission != 'Queen':
        #数据格式转换
        ld = LoadResult()
        ld.run()
    
    #分割训练集和测试集
    tt = Train_Test(mission)
    tt.run()

    #Pearson相关分析
    pr = PearsonCorrelation(mission)
    pr.run()
    
    #CDDA算法
    cd = CDDA(mission)
    cd.run()
    
    #OPS算法
    ops = Method_OPS(mission)
    ops.run()
    
    #PLS回归建模并作图
    mo = Modeling(mission)
    mo.run()

    #保存报告文件
    haReporter()
    
    if mission == 'Queen':
        moveResult()
    
if __name__ == '__main__':
    runDetective()
    
