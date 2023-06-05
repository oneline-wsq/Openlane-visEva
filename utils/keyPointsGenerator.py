import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import math
# from utils import indexToXY

class keyPointsGenerator(object):
    def __init__(self,seq,width=48,height=24,resolution=0.8):
        self.seq=seq # 读取所有的点的值
        self.width=width #
        self.height=height
        self.resolution = resolution
        x_grids = self.width / self.resolution  # x和y的所有格子数
        y_grids = self.height / self.resolution
        self.all_gridsNums=int(x_grids*y_grids)
        # 拓扑类型无效点
        self.invaildIndex=[0,1,2] # pad,bos,eos
        # 拓扑类型中的关键点
        self.kpIndex=[3,4,6,7]  # start=3; end=4; fork=6; merge=7
        # 获得对应的所有xy坐标值,并标记出真正的分叉点和汇聚点
        self.seqXY=self.turnSeqXY()
        # 删掉所有的无意义点
        # self.seqVaild=self.getValid(self.seqXY) # 不能删掉，因为有link到前一个点，直接删掉，索引会改变
        # 获得关键点的数目 (3,4,12,14)
        self.kpnums=self.getKeypointsNum(self.seqXY)
        # 为所有的点建立一个邻接矩阵,包括控制点
        self.allMatrix = self.buildAllMatrix(self.seqXY)
        # 为所有关键点建立邻接矩阵
        self.kpMatrix=self.buildKpMatrix(self.seqXY)
        # 计算每个点的到下一个点的方向
        self.angles = self.getAngles(self.seqXY, self.allMatrix)


    def getKeypointsNum(self,seq):
        # 这里的keypoints为start,end和真正的fork 和merge点
        # start=3; end=4; fork=12; merge=14

        nums=0 # 记录关键点的数量
        newkpIndex=[3,4,12,14]
        pnums=seq.shape[0]
        for i in range(pnums):
            point=seq[i,:]
            if int(point[2]) in newkpIndex:
                # 记录下该点的所有值和在原seq中的标号
                nums+=1
        return nums

    def getValid(self,seq):
        # 找到所有的有效点（去除padding, bos,eos）
        seq=seq
        keypoints = []  # 用来存储关键点
        pnums = seq.shape[0] # 所有点，包括无意义点的数量
        for i in range(pnums):
            point = seq[i, :]
            if int(point[2]) in self.invaildIndex:
                continue
            else:
                keypoints.append(point)
        return np.array(keypoints)

    def getAngles(self,seq,allMatrix):
        """
        计算每个点到下一个点的方向角
        :param seq:
        :param allMatrix:
        :return:
        """
        pnums=seq.shape[0]
        angles=np.zeros([pnums]) # 记录每个点的角度
        for i in range(pnums):
            p=seq[i,:]
            t=p[2]
            if t in self.invaildIndex: # 无效点跳过
                continue
            # 对于每个点，找到to的所有的点
            # 对于merge点，要单独考虑
            if t==14:
                fromPoints=np.nonzero(allMatrix[:,i])
                fromPoints=fromPoints[0]
                # 计算平均值
                angs=0
                if fromPoints.shape[0]==0:
                    continue
                for j in fromPoints:
                    # 计算角度
                    p2=seq[j,:]
                    aa=calc2Angles([p2[0],p2[1]],[p[0],p[1]])
                    angs+=aa
                # 计算均值
                angs=angs/fromPoints.shape[0]
            elif t==4:
                # 终止点，角度等于上一个角度
                angs=angles[i-1]
            else:
                toPoints=np.nonzero(allMatrix[i,:])
                toPoints=toPoints[0]
                # 计算平均值
                angs=0
                if toPoints.shape[0]==0:
                    continue
                if toPoints.any():
                    for j in toPoints:
                        # 计算角度
                        p2=seq[j,:]
                        aa=calc2Angles([p[0],p[1]],[p2[0],p2[1]])
                        angs+=aa
                    # 计算均值
                    angs=angs/toPoints.shape[0]
            angles[i]=angs

        return angles

    def buildAllMatrix(self,seq):
        # 建立所有点，包括control points的邻接矩阵
        # 用于后续计算每个点的方向
        # 首先计算所有关键点的数量
        pnums=seq.shape[0]
        matrix=np.zeros([pnums,pnums])
        # 默认输入的seq为seqXY，已经标注出真正的分叉和汇聚点

        beforeKey=None # 用来记录前一个点和后一个点的索引
        afterKey=None
        for i in range(pnums):
            p=seq[i,:]
            t = int(p[2])  # 拓扑类型
            if t in self.invaildIndex:
                continue
            if t == 3: # 起点
                # 起点，则该点和下一个点有连接关系
                if i+1<pnums:
                    matrix[i,i+1]=1
            if t == 4: # 终点
                continue
            if t == 12:
                # 真正的分叉点

                if i + 1 < pnums and (not seq[i + 1, 2] in self.invaildIndex):
                    matrix[i, i + 1] = 1
            if t==14:
                # merge点
                # 首先判断下一个点是否是起点或无效点，下一个点不能是起点，或无效点
                if i+1<pnums and (not seq[i+1,2] in self.invaildIndex) and (seq[i+1,2]!=3):
                    matrix[i,i+1]=1
            if t == 6:
                # 分叉点Link到的是它的入点
                # fork点前面一般为end(4)，before和after都为None，后面跟一个end点
                # 首先把入点的关系表示为1
                if int(p[3])>=pnums:
                    continue
                matrix[int(p[3]), i] = 1
                if i + 1 < pnums and (not seq[i + 1, 2] in self.invaildIndex):
                    matrix[i, i + 1] = 1
            if t == 7:
                if int(p[4])>=pnums:
                    continue
                # mergelink到的是它的出点
                matrix[i,int(p[4])]=1
            if t==5:
                if i + 1 < pnums and (not seq[i + 1, 2] in self.invaildIndex):
                    matrix[i, i + 1] = 1
        return matrix


    def turnSeqXY(self):
        seqXY=self.seq.copy() # 拷贝
        # seqXY=seqXY.astype(np.float64)
        for i in range(self.seq.shape[0]):
            point=self.seq[i,:]
            x=point[0]
            y=point[1]
            z=point[2]
            if point[3] in self.invaildIndex:
                # 如果不是有效点，则不转化
                continue
            # 重新保存
            seqXY[i,0]=x
            seqXY[i,1]=y
        # 要把链接到的真正的fork和merge标注出来
        for i in range(seqXY.shape[0]):
            p=seqXY[i,:]
            t=int(p[3])# 拓扑类型
            if int(p[4]) < seqXY.shape[0]:
                if t==6 and seqXY[int(p[4]),3]!=3:
                    # fork
                    seqXY[int(p[4]),3]=12 # 类别为索引3，标记为新的fork点
            if int(p[5]) < seqXY.shape[0]:
                if t==7 and seqXY[int(p[5]),3]!=4:
                    # merge
                    seqXY[int(p[5]),3]=14 # 标记为新的merge点
        return seqXY


    def buildKpMatrix(self,seq):
        """
        生成所有关键点的邻接矩阵
        :param seq: 待建序列，默认为seqXY
        :return: matrix
        """
        # 生成一个邻接矩阵
        pnums=seq.shape[0]
        matrix=np.zeros([pnums,pnums])

        beforeKey=None
        afterKey=None # 用来记录前一个点和后一个点的数据
        beforeIndex=-1 # 用来记录前一个点和后一个点的索引
        afterIndex=-1

        for i in range(pnums):
            p=seq[i,:]
            t=int(p[3]) # 拓扑类型
            if t in self.invaildIndex or t==5:
                continue
            if t==3:
                # 起点
                beforeKey=p
                afterKey=None
                beforeIndex=i
                afterIndex=-1
            if t==4:
                # 终点
                if  type(afterKey) is np.ndarray:
                    beforeKey = afterKey
                    afterKey = p
                    beforeIndex=afterIndex
                    afterIndex=i
                else:
                    afterKey=p
                    afterIndex=i
                matrix[beforeIndex,afterIndex]=1
                beforeKey=None
                afterKey=None
                beforeIndex=-1
                afterIndex=-1

            if t==12 or t==14:
                # 标记的新的分叉点和merge点
                if  type(afterKey) is np.ndarray:
                    beforeKey = afterKey
                    afterKey = p
                    beforeIndex=afterIndex
                    afterIndex=i
                else:
                    afterKey=p
                    afterIndex = i
                matrix[beforeIndex, afterIndex] = 1
            if t==6:
                # fork点前面一般为end(4)，before和after都为None，后面跟一个end点
                if int(p[3])>=pnums:
                    continue
                beforeKey=seq[int(p[3]),:] # 找到之前的索引值
                beforeIndex=int(p[3])
                afterKey=None
                afterIndex=-1
            if t==7:
                if int(p[4])>=pnums:
                    continue
                # merge点前面通常有一个start点，那么没有afterKey
                # merge点前面也有可能是一个fork点，由afterkey
                if type(afterKey) is np.ndarray:
                    # 如果afterKey有值，则
                    beforeKey=afterKey
                    beforeIndex=afterIndex
                afterKey=seq[int(p[4]),:] # 找到之前的索引值
                afterIndex=int(p[4])
                matrix[beforeIndex,afterIndex] = 1
        return matrix


    def plotLines(self,fig,ax,seq):
        # 给定 seqEva，画出连接线

        for i in range(seq.shape[0]):
            point=seq[i,:]

            if point[3] == 3:
                # 起始点
                ax.scatter(point[0],point[1],color='green',marker='o')
                ax.annotate(i,(point[0],point[1]))
                # ax.annotate(point[5], (point[0], point[1]), weight="bold", color="r")
                if i<seq.shape[0]-1 and (not seq[i + 1, 2] in self.invaildIndex):
                    plt.plot([point[0],seq[i+1,:][0]],[point[1],seq[i+1,:][1]])
            if point[3]==4:
                # 终点或者 segment点
                ax.scatter(point[0],point[1],color='#A9A9A9',marker='o')
                ax.annotate(i,(point[0],point[1]))
            
            if point[3]==8:
                # segment点
                ax.scatter(point[0],point[1],color='fuchsia',marker='D')
                ax.annotate(i,(point[0],point[1]))
                if i<seq.shape[0]-1 and (seq[i+1,:][2] not in self.invaildIndex):
                    # 画与下一条的连线
                    plt.plot([point[0],seq[i+1,:][0]],[point[1],seq[i+1,:][1]])
            
            if point[3]==6:
                # 车道语言的分叉点
                ax.scatter(point[0],point[1],color='#F4A460',marker='+')
                # ax.annotate(i,(point[0],point[1]+1))
                # 将链接的点标为真正的分叉点
                if int(point[4])>=seq.shape[0]:
                    continue
                fp=seq[int(point[4]),:]
                if fp[3]==12:
                    ax.scatter(fp[0],fp[1],color='#A52A2A',marker='o')
                    ax.annotate(point[4], (fp[0], fp[1]))

                if i<seq.shape[0]-1 and (seq[i+1,:][2] not in self.invaildIndex):
                    plt.plot([point[0],seq[i+1,:][0]],[point[1],seq[i+1,:][1]])
                plt.plot([point[0], fp[0]], [point[1], fp[1]])
            
            if point[3]==7:
                # 车道语言的merge
                ax.scatter(point[0],point[1],color='#00BFFF',marker='+')
                ax.annotate(i,(point[0],point[1]))
                # 将链接的点标为真正的merge
                if int(point[5]) >= seq.shape[0]:
                    continue
                fp=seq[int(point[5]),:]
                if fp[3]==14:
                    ax.scatter(fp[0],fp[1],color='#1E90FF',marker='o')
                    ax.annotate(point[5], (fp[0], fp[1]))
                plt.plot([point[0], fp[0]], [point[1], fp[1]])
            
            if point[3]==5 or point[3]==12 or point[3]==14:
                # continue点
                ax.scatter(point[0],point[1],color='#000000',marker='*')
                ax.annotate(i,(point[0],point[1]))
                # ax.annotate(point[5],(point[0],point[1]),weight="bold", color="r")
                if i<seq.shape[0]-1:
                    if seq[i+1,:][3]!=3 and (seq[i+1,:][3] not in self.invaildIndex):
                        plt.plot([point[0],seq[i+1,:][0]],[point[1],seq[i+1,:][1]])
        return fig,ax

    def getLines(self,seq):
        # 将点放到对应的线中，按照线的方式存储
        lines=[]
        new_line={'start':None,'end':None,'controls':[]}
        
        for i in range(seq.shape[0]):
            point=seq[i,:]
            if point[3] in [0,1,2]:
                continue
            if point[3] == 3:                
                new_line={'start':None,'end':None,'controls':[]}
                new_line['start']=(point[0],point[1],point[2]) # 将该点作为一条线的起点

            elif point[3]==4:
                # 终点
                if i>0 and seq[i-1][3]==14 :
                    # 如果终点的前一个点是真正的merge点
                    new_line={'start':None,'end':None,'controls':[]}
                    new_line['controls']=[]
                    new_line['start']=(seq[i-1][0],seq[i-1][1],seq[i-1][2])
                
                new_line['end']=(point[0],point[1],point[2])
                lines.append(new_line)
                new_line={'start':None,'end':None,'controls':[]}
            
            elif point[3]==5:
                # controls
                if i>0 and seq[i-1][3]==12 or seq[i-1][3]==14:
                    # 上一个点是真正的分叉点或者汇聚点
                    new_line={'start':None,'end':None,'controls':[]}
                    new_line['controls']=[]
                    new_line['start']=(seq[i-1][0], seq[i-1][1], seq[i-1][2])

                new_line['controls'].append((point[0],point[1],point[2]))
            
            elif point[3]==6:
                # 车道语言的分叉点, 实际是分叉点的下一个点
                # 将链接的点标为真正的分叉点
                if int(point[4])>=seq.shape[0]:
                    print('fork的索引大于序列长度')
                    continue
                fp=seq[int(point[4]),:]
                if fp[3]==12:
                    new_line={'start':None,'end':None,'controls':[]}
                    new_line['start']=(fp[0],fp[1],fp[2])
                    new_line['controls']=[]
                    new_line['controls'].append((point[0],point[1],point[2]))
                else:
                    # print('fp[3]!=12')
                    new_line={'start':None,'end':None,'controls':[]}
                    new_line['start']=(fp[0],fp[1],fp[2])
                    new_line['controls']=[]
                    new_line['controls'].append((point[0],point[1],point[2]))

            elif point[3]==7:
                # 车道语言的汇聚点, 实际是汇聚点的前一个点
                new_line['controls'].append((point[0],point[1],point[2]))
                # 将链接的点标为真正的merge
                if int(point[5]) >= seq.shape[0]:
                    print('merge的索引大于序列长度')
                    continue
                fp=seq[int(point[5]),:]
                
                if fp[3]==14:
                    new_line['end']=(fp[0],fp[1],fp[2])
                else:
                    # print('fp[3]!=14')
                    new_line['end']=(fp[0],fp[1],fp[2])

                lines.append(new_line)
                new_line={'start':None,'end':None,'controls':[]}
            elif point[3]==12:
                # 真正的分叉点
                new_line['end']=(point[0],point[1],point[2])
                lines.append(new_line)
                new_line={'start':None,'end':None,'controls':[]}
            elif point[3]==14:
                # 真正的汇聚点
                new_line['end']=(point[0],point[1],point[2])
                lines.append(new_line)
                new_line={'start':None,'end':None,'controls':[]}
                new_line['controls']=[]
            else:
                continue
        return lines

    def getLinesNotMerge(self,seq):
            # 将点放到对应的线中，按照线的方式存储
            lines=[]
            new_line={'start':None,'end':None,'controls':[],'confidence':[]}
            
            for i in range(seq.shape[0]):
                point=seq[i,:]
                if point[3] in [0,1,2]:
                    continue
                if point[3] == 3:                
                    new_line={'start':None,'end':None,'controls':[],'confidence':[]}
                    new_line['start']=(point[0],point[1],point[2]) # 将该点作为一条线的起点
                    new_line['confidence'].append(point[6:] if len(point)>6 else []) 
                    
                elif point[3]==4 :
                    # 终点
                    if i>0 and seq[i-1][3]==14 :
                        # 如果终点的前一个点是真正的merge点
                        new_line={'start':None,'end':None,'controls':[],'confidence':[]}
                        new_line['controls']=[]
                        new_line['start']=(seq[i-1][0],seq[i-1][1],seq[i-1][2])
                        new_line['confidence'].append(seq[i-1][6:] if len(point)>6 else [])
                    
                    new_line['end']=(point[0],point[1],point[2])
                    new_line['confidence'].append(point[6:] if len(point)>6 else []) 
                    # assert len(new_line['confidence'])==len(new_line['controls'])+2
                    
                    lines.append(new_line)
                    new_line={'start':None,'end':None,'controls':[],'confidence':[]}

                elif point[3]==8:
                    # segment
                    if i>0 and seq[i-1][3]==4:
                        # 前一个是终点
                        new_line={'start':None,'end':None,'controls':[],'confidence':[]}
                        new_line['controls']=[]
                        new_line['start']=(point[0],point[1],point[2])
                        new_line['confidence'].append(point[6:] if len(point)>6 else []) 
                    else:
                        if i>0 and (seq[i-1][3]==14) :
                            # 如果终点的前一个点是真正的merge点
                            new_line={'start':None,'end':None,'controls':[],'confidence':[]}
                            new_line['controls']=[]
                            new_line['start']=(seq[i-1][0],seq[i-1][1],seq[i-1][2])
                            new_line['confidence'].append(seq[i-1][6:] if len(point)>6 else []) 
                        
                        new_line['end']=(point[0],point[1],point[2])
                        new_line['confidence'].append(point[6:] if len(point)>6 else []) 
                        # assert len(new_line['confidence'])==len(new_line['controls'])+2
                        
                        lines.append(new_line)
                        # 然后新建一条线
                        new_line={'start':None,'end':None,'controls':[],'confidence':[]}
                        new_line['controls']=[]
                        new_line['start']=(point[0],point[1],point[2])
                        new_line['confidence'].append(point[6:] if len(point)>6 else []) 

                elif point[3]==5:
                    # controls
                    if i>0 and seq[i-1][3]==12 or seq[i-1][3]==14:
                        # 上一个点是真正的分叉点或者汇聚点
                        # 或者上一个点是segment点
                        new_line={'start':None,'end':None,'controls':[],'confidence':[]}
                        new_line['controls']=[]
                        new_line['start']=(seq[i-1][0], seq[i-1][1], seq[i-1][2])
                        new_line['confidence'].append(seq[i-1][6:] if len(point)>6 else []) 

                    new_line['controls'].append((point[0],point[1],point[2]))
                    new_line['confidence'].append(point[6:] if len(point)>6 else []) 
                
                elif point[3]==6:
                    # 车道语言的分叉点, 实际是分叉点的下一个点
                    # 将链接的点标为真正的分叉点
                    if int(point[4])>=seq.shape[0]:
                        print('fork的索引大于序列长度')
                        continue
                    fp=seq[int(point[4]),:]
                    if fp[3]==12:
                        new_line={'start':None,'end':None,'controls':[],'confidence':[]}
                        new_line['start']=(fp[0],fp[1],fp[2])
                        new_line['confidence'].append(fp[6:] if len(point)>6 else []) 
                        new_line['controls']=[]
                        new_line['controls'].append((point[0],point[1],point[2]))
                        new_line['confidence'].append(point[6:] if len(point)>6 else []) 
                    else:
                        # print('fp[3]!=12')
                        new_line={'start':None,'end':None,'controls':[],'confidence':[]}
                        new_line['start']=(fp[0],fp[1],fp[2])
                        new_line['confidence'].append(fp[6:] if len(point)>6 else []) 
                        new_line['controls']=[]
                        new_line['controls'].append((point[0],point[1],point[2]))
                        new_line['confidence'].append(point[6:] if len(point)>6 else []) 

                elif point[3]==7:
                    # 车道语言的汇聚点, 实际是汇聚点的前一个点
                    new_line['controls'].append((point[0],point[1],point[2]))
                    new_line['confidence'].append(point[6:] if len(point)>6 else []) 
                    # 将链接的点标为真正的merge
                    if int(point[5]) >= seq.shape[0]:
                        print('merge的索引大于序列长度')
                        continue
                    fp=seq[int(point[5]),:]
                    
                    if fp[3]==14:
                        new_line['end']=(fp[0],fp[1],fp[2])
                        new_line['confidence'].append(fp[6:] if len(point)>6 else []) 
                    else:
                        # print('fp[3]!=14')
                        new_line['end']=(fp[0],fp[1],fp[2])
                        new_line['confidence'].append(fp[6:] if len(point)>6 else []) 

                    lines.append(new_line)
                    new_line={'start':None,'end':None,'controls':[],'confidence':[]}
                
                elif point[3]==12:
                    # 真正的分叉点
                    new_line['end']=(point[0],point[1],point[2])
                    new_line['confidence'].append(point[6:] if len(point)>6 else []) 
                    # assert len(new_line['confidence'])==len(new_line['controls'])+2
                    
                    lines.append(new_line)
                    new_line={'start':None,'end':None,'controls':[],'confidence':[]}
                elif point[3]==14:
                    # 真正的汇聚点
                    new_line['end']=(point[0],point[1],point[2])
                    new_line['confidence'].append(point[6:] if len(point)>6 else [])
                    # assert len(new_line['confidence'])==len(new_line['controls'])+2
                    
                    lines.append(new_line)
                    new_line={'start':None,'end':None,'controls':[],'confidence':[]}
                    new_line['controls']=[]
                else:
                    continue
            return lines


def indexToXY(coarse_index, fine_index, big_h=15, big_w=30, small_h=8, small_w=8, resolution=0.2):
    """
    将粗粒度和细粒度转为x,y值
    :param coarse_index: 粗粒度索引
    :param fine_index: 细粒度索引
    :param big_h:
    :param big_w:
    :param small_h:
    :param small_w:
    :return:
    """
    # 先找到粗粒度的格子，沿着竖边索引,横为x,竖为y
    x_coarse = int(np.ceil(coarse_index / big_h))  # 计算y
    y_coarse = int(coarse_index - (x_coarse - 1) * big_h)
    # 再找细粒度的格子
    x_fine = int(np.ceil(fine_index / small_h))
    y_fine = int(fine_index - (x_fine - 1) * small_h)

    x_grid_index = (x_coarse - 1) * small_w + x_fine
    y_grid_index = (y_coarse - 1) * small_h + y_fine

    x = x_grid_index * resolution
    y = y_grid_index * resolution

    return x, y
def calc2Angles(p1,p2):
    # 计算向量p1到p2的相对于x轴的旋转角度
    x_diff=p2[0]-p1[0]
    y_diff=p2[1]-p1[1]

    angle=math.atan2(y_diff,x_diff)
    angle=math.degrees(angle) # 转角度
    return angle

if __name__ == '__main__':
    p1=[3,3]
    p2=[2,2]
    a=calc2Angles(p1,p2)
    print(a)


