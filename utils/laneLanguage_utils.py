import numpy as np
import copy
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
from scipy.interpolate import interp1d
from scipy.interpolate import griddata
from scipy import interpolate
from utils.inter_arc import interp_arc
from mpl_toolkits import mplot3d
import torch
from math import factorial
from utils.bezier import CustomParameterizeLane
from sklearn.preprocessing import StandardScaler,MinMaxScaler

from sklearn import preprocessing  



def sort_cp(lines,carlength=50,carwidth=25):
    newlines=[]
    for line in lines:
        if not line['start']:
            continue

        if len(line['controls']):
            # 存在控制点
            start=line['start']
            end=line['end']
            # 排序，按照与start点的距离排序
            # line['controls'].sort(key=lambda x:x[0],reverse=start[0]>end[0])
            line['controls'].sort(key=lambda x:(x[0]-start[0])**2+(x[1]-start[1])**2)

        
        # 调整坐标，将原点跳到图片中心
        # line['start']=(line['start'][0]-carlength,line['start'][1]-carwidth)
        # line['end']=(line['end'][0]-carlength,line['end'][1]-carwidth)
        # cps=[]
        # for cp in line['controls']:
        #     cps.append((cp[0]-carlength,cp[1]-carwidth))
        # line['controls']=cps

        # 增加一个键points
        # points[0] 为 起点
        # points[-1] 为终点
        points=[]
        points.append(np.array(line['start']))
        for cp in line['controls']:
            points.append(np.array(cp))
        points.append(np.array(line['end']))

        line['points']=[]
        for p in points:
            # 补一个0
            # p=np.pad(p,(0,1),'constant',constant_values=(0,0))
            line['points'].append(p)
        newlines.append(line)

    return newlines



def addPointsKey(arclines):
    for line in arclines:
        points=[]
        points.append(np.array(line['start']))
        for cp in line['controls']:
            points.append(np.array(cp))
        points.append(np.array(line['end']))
        line['points']=points
    return arclines


def sort_points(lines):
    newlines=[]
    for line in lines:
        start=line['start']
        end=line['end']
        # 排序，按照与start点的距离排序
        # line['controls'].sort(key=lambda x:x[0],reverse=start[0]>end[0])
        line['points'].sort(key=lambda x:(x[0]-start[0])**2+(x[1]-start[1])**2)
        newlines.append(line)

    return newlines


def plotLines2(lines,isdot=True,isid=False, isConfidence=False):
    fig = plt.figure(figsize=(14, 7))
    ax = fig.add_subplot(111)
    ax.set_xlim(xmin=-50, xmax=50)
    ax.set_ylim(ymin=-25, ymax=25)
    y_major_locator = MultipleLocator(5)
    ax.yaxis.set_major_locator(y_major_locator)
    ax.xaxis.set_major_locator(y_major_locator)
    
    i=0
    for line in lines:
        points=line['points']
        
        x=[p[0] for p in points]
        y=[p[1] for p in points]

        ax.plot(x,y,'o-')
        mid_index=len(x)//2
        mid=[points[mid_index][0],points[mid_index][1]]
        if isConfidence:
            anno=str(line['id'])+'( cf: ' + str(np.around(line['confidence'],3))+' )'
            ax.annotate(anno,(mid[0],mid[1]))
        else:
            ax.annotate(line['id'],(mid[0],mid[1]))
        i+=1

        if isdot:
            # 画点
            ax.scatter(line['start'][0],line['start'][1],color='green',marker='o')
            ax.scatter(line['end'][0],line['end'][1],color='gray',marker='o')
            if 'controls' in line.keys():
                for cp in line['controls']:
                    ax.scatter(cp[0],cp[1],color='black',marker='+')
             
    return fig, ax



def plot3dLines(lines,isdot=True):
    fig = plt.figure(figsize=(14, 7))
    ax = plt.axes(projection='3d')
    ax.set_xlim(xmin=-50, xmax=50)
    ax.set_ylim(ymin=-25, ymax=25)
    ax.set_zlim(zmin=-2,zmax=2)
    ax.set_box_aspect((2,1,0.5)) 
    xy_major_locator = MultipleLocator(5)
    ax.yaxis.set_major_locator(xy_major_locator)
    ax.xaxis.set_major_locator(xy_major_locator)
    z_major_locator = MultipleLocator(0.5)
    ax.zaxis.set_major_locator(z_major_locator)
    
    i=0
    for line in lines:
        points=line['points']
        
        x=[p[0] for p in points]
        y=[p[1] for p in points]
        z=[p[2] for p in points]

        ax.plot3D(x,y,z)
        # ax.annotate(i,(points[0][0],points[0][1]))
        i+=1

        if isdot:
            # 画点
            ax.scatter3D(line['start'][0],line['start'][1],line['start'][2],color='green',marker='o')
            ax.scatter3D(line['end'][0],line['end'][1],line['end'][2],color='gray',marker='o')
            for cp in line['controls']:
                ax.scatter3D(cp[0],cp[1],cp[2],color='black',marker='+')
    
    return fig, ax


def fitSplines(arclines,interval_nums=50):
    n=len(arclines)
    for i in range(n):
        points=arclines[i]['points']
        # 先对 x, y 进行3次样条拟合
        x=np.array([points[i][0] for i in range(len(points))])
        y=np.array([points[i][1] for i in range(len(points))])
        z=np.array([points[i][2] for i in range(len(points))])

        x,y,z=delCSame(x,y,z)

        # 首先使用 splprep 来获得S-pline的参数
        # 输入的x,y 不能有两个连续相同的数
        tck, u= interpolate.splprep([x,y,z],k=3,s=0)
        u=np.linspace(0,1,num=interval_nums,endpoint=True)
        out=interpolate.splev(u,tck)

        x_new=out[0]
        y_new=out[1]
        z_new=out[2]

        # 对于插值后的 x,y 坐标，生成新的z 轴坐标
        # z_new=griddata((np.array(x),np.array(y)),np.array(z),(x_new,y_new),method='cubic')

        points_new=[(x_new[i], y_new[i], z_new[i]) for i in range(len(x_new))]

        arclines[i]['points']=points_new

        arclines[i]['points'][0]=arclines[i]['start']
        arclines[i]['points'][-1]=arclines[i]['end']

    return arclines

def fitBezier(arclines):
    """首先转成原代码中的输入形状 [lane_nums, controlpoints_nums, 3]"""
    lanes=[]
    for line in arclines:
        # p=np.array(line['points']) # [5,3]
        # p=[line['points'][i].tolist() for i in range(len(line['points']))]
        p=[list(line['points'][i]) for i in range(len(line['points']))]
        lanes.append(p)
    lanes=np.array(lanes)

    def comb(n, k):
        return factorial(n) // (factorial(k) * factorial(n - k))
    
    n_points = 11
    n_control = lanes.shape[1] # 控制点的数量 lanes.shape[-1] // 3 =5
    A = np.zeros((n_points, n_control))
    t = np.arange(n_points) / (n_points - 1)
    
    for i in range(n_points):
        for j in range(n_control):
            A[i, j] = comb(n_control - 1, j) * np.power(1 - t[i], n_control - 1 - j) * np.power(t[i], j)
    
    bezier_A = torch.tensor(A, dtype=torch.float32)
    lanes = torch.tensor(lanes, dtype=torch.float32)
    lanes = torch.einsum('ij,njk->nik', bezier_A, lanes)
    lanes = lanes.numpy()
    
    lane_centerline=[]
    for i, lane in enumerate(lanes):
        lane_centerline.append({
            'id': i ,
            'points': lane.astype(np.float32),
            'confidence': arclines[i]['confidence'],
            'start':tuple(lane[0]),
            'end':tuple(lane[-1])
        })

    return lane_centerline
        

def tuple2np(arclines):
    """将 arclines 变换成 评价指标中需要的格式"""
    lane_centerline=[]

    for value in arclines :
        if isinstance(value['points'],list):
            points=np.array(value['points']).astype(np.float32)
        else:
            points=value['points']
        lane_centerline.append({
            'id': value['id'],
            'points': points ,
            'confidence': value['confidence'],
            'start':tuple(value['start']),
            'end':tuple(value['start'])
        })

    return lane_centerline
    


def inter_lines(arclines,t=5):
    for value in arclines:
        s=np.array(value['start'])
        e=np.array(value['end'])

        # 计算两者的距离
        dis=np.linalg.norm(s-e)
        
        points=np.array(value['points'])
        n=len(points)

        # 插值
        points=interp_arc(points,t)

        # 确保起点终点没变
        if (points[0]==s).any()==False:
            points[0]=s
        if (points[-1]==e).any()==False:
            points[-1]=e
        
        points = np.around(points,3)
        value['points']=list(map(tuple,points))
        # value['start']=value['points'][0]
        # value['end']=value['points'][-1]

    return arclines

def inter_lines_subline(arclines,t=5):

    for value in arclines:
        s=np.array(value['start'])
        e=np.array(value['end'])
        ps=np.array(value['points'])
        points=[]

        # 起点

        for i in range(ps.shape[0]-1):
            subpoints=np.vstack((ps[i],ps[i+1]))
            # 插值    
            new_subpoints=interp_arc(subpoints,t) # 每一小段之间插t个点
            new_subpoints=[tuple(new_subpoints[j]) for j in range(t)]
            points+=new_subpoints

        # 去重+转成 np array
        npoints=list(set(points))
        # set 变化
        npoints.sort(key=points.index)

        
        # 确保起点终点没变
        if (np.array(npoints[0])==s).any()==False:
            npoints[0]=tuple(s)
        if (np.array(npoints[-1])==e).any()==False:
            npoints[-1]=tuple(e)
        
        npoints = np.around(np.array(npoints),3)
        value['points']=list(map(tuple,npoints))


    return arclines


def getLineAdjMatrix(arclines):
    """获得线与线之间的邻接矩阵"""
    # 还是按照起点终点的联通关系来判断是否联通
    n=len(arclines)
    matrix=np.zeros((n,n))
    for i,line in enumerate(arclines):
        s1=line['start']
        e1=line['end']
        # 遍历剩下的线，判断是否有起点获得终点相同的
        for j, line2 in enumerate(arclines):
            if j==1:
                continue
            s2=line2['start']
            e2=line2['end']

            # 判断起点或者终点是否相同：
            if e1==s2:
                # 上一条线的终点连接着下一条线的起点
                matrix[i,j]=1
            elif s1==e2:
                # 上一条线的起点是下一条线的终点
                matrix[j,i]=1
    return matrix

def getLineAdjMatrixNoConstant(arclines,seqXY):
    """获得线与线之间的邻接矩阵"""
    # 还是按照起点终点的联通关系来判断是否联通
    n=len(arclines)
    matrix=np.zeros((n,n))
    for i,line in enumerate(arclines):
        s1=line['start']
        e1=line['end']
        # 找起点在 seqXY中的类型
        before_end=np.array([-10000,-10000,-10000])
        for ii in range(len(seqXY)):
            if (s1[0],s1[1],s1[2])==(seqXY[ii][0],seqXY[ii][1],seqXY[ii][2]):
                if seqXY[ii][3]==8 and ii>0 and seqXY[ii-1][3]==4:
                    # 为segment,且上一个点为 停止点
                    before_end=seqXY[ii-1][:3] # 一条线的before end 只有一个
                    
        
        # 遍历剩下的线，判断是否有起点获得终点相同的
        for j, line2 in enumerate(arclines):
            if j==1:
                continue
            s2=line2['start']
            e2=line2['end']

            # 判断起点或者终点是否相同：
            if e1==s2:
                # 上一条线的终点连接着下一条线的起点
                matrix[i,j]=1
            elif s1==e2 or e2==tuple(before_end):
                # 上一条线的起点是下一条线的终点
                matrix[j,i]=1
            
    
    return matrix

def add_keys(arclines):
    for i in range(len(arclines)):
        arclines[i]['id']=i
        arclines[i]['confidence']=1
    return arclines

    

def turn2bezier(arclines,n_control=5):
    method_para = dict(n_control=n_control)
    CPL = CustomParameterizeLane('bezier_Endpointfixed',method_para=method_para)
    LinesPoints={}

    for i in range(len(arclines)):
        line=arclines[i]
        points = np.array(line['points'])
        startP=tuple(points[0,:])
        endP=tuple(points[-1,:])
        line['start']=startP
        line['end']=endP
        # 增加控制点
        controls=CPL.fit_bezier_Endpointfixed(points,n_control=n_control)
        line['controls']=[tuple(controls[i]) for i in range(1,n_control-1)] # 0是起点，-1是终点
        arclines[i]['controls']=line['controls']
        arclines[i]['points']=[tuple(controls[i]) for i in range(n_control)] 

    return arclines

def delCSame(x,y,z):
    xyz=np.vstack((x,y,z))
    xyz=[(xyz[0][i],xyz[1][i],xyz[2][i]) for i in range(xyz.shape[1])]
    delist=[]
    for i in range(1,len(xyz)):
        if xyz[i]==xyz[i-1]:
            delist.append(i)
    xyz = [xyz[i] for i in range(0, len(xyz), 1) if i not in delist]
    x=np.array([xyz[i][0] for i in range(len(xyz))])
    y=np.array([xyz[i][1] for i in range(len(xyz))])
    z=np.array([xyz[i][2] for i in range(len(xyz))])
    return x,y,z

def directSample11(arclines):
    lanes=[]
    for line in arclines:
        p=[list(line['points'][i]) for i in range(len(line['points']))]
        p=p[::20]
        line['points']=np.array(p, dtype=np.float32)
        lanes.append(line)
    return lanes


def changeLineFormat(lines):
    newlines=[]
    for i, line in enumerate(lines):
        if not line['start']:
            continue
        
        # 首先得到每条线上的所有的点
        points=[]
        points.append(np.array(line['start']))
        for cp in line['controls']:
            points.append(np.array(cp))
        points.append(np.array(line['end']))

        line['points']=[]
        for p in points:
            line['points'].append(p)
        
        # 设置每条线的id, 避免重复
        line['id']= i
        
        
        
        # 确定每条线的 confidence
        if 'confidence' not in line.keys():
            line['confidence']=1
        elif len(line['confidence'][0])==0:
            # 说明也没有 confidence
            line['confidence']=1
        else:
            assert len(line['points'])==len(line['confidence'])
            line['confidence']=getLineConfidence(line)
            # print(line['confidence'])
        newlines.append(line)
         
    return newlines

def getLineConfidence(line):
    c = np.array(line['confidence']) # [N,5]
    c1 = c[:,0]
    c3 = c[:,2]
    
    c_combine=[]
    # 对每个点的c1 和 c3 加权
    for i in range(c1.shape[0]):
        c_combine.append(0.5*c1[i]+0.5*c3[i]) 

    # 对整条线加权，取平均
    c_combine=np.array(c_combine)
    line_c = np.mean(c_combine)

    return line_c
    
def preprocessUnc(unc):
      
    # 将不确定度转为置信度，并做归一化
    min_max_scaler = preprocessing.MinMaxScaler()
    c_norm =  1 - min_max_scaler.fit_transform(unc)
    
    return c_norm
    
    
def sortLinesByConfidence(arclines):
    # arclines=sorted(arclines,key= lambda x:-x['confidence'])
    for i in range(len(arclines)):
        arclines[i]['confidence']=1
    return arclines


def recordConfidence(arclines):
    r=[]
    for i in range(len(arclines)):
        r.append(arclines[i]['confidence'])
    return r

def delLineCon(arclines,threshold=0.9):
    r=[]
    for i in range(len(arclines)):
        if arclines[i]['confidence']>=threshold:
            r.append(arclines[i])
    return r