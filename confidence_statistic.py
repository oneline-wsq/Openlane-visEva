import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))		# 用两种路径都是可以的
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator

if __name__ =='__main__':
    zr=np.load('confidence.npy')
    tp_count_list=zr.tolist()
    # zr=zr.tolist() # 转成list
    # 画直方图
    # topology_len_hist
    topologoy_count_array = np.array(tp_count_list)
    # 范围从1开始
    topology_hist, bin_edges = np.histogram(topologoy_count_array, bins=10, range=(0,1)) 
    fig, ax = plt.subplots()
    # x=list(np.arange(0,1,0.05))
    x=['0.0','0.1','0.2','0.3','0.4','0.5','0.6','0.7','0.8','0.9']
    height=list(topology_hist)
    ax.bar(x,height=height)
    # ax.set_xlim(xmin=0,xmax=1)
    # ax.set_ylim(ymin=0,ymax=100000)
    # ax.legend()
    ax.set_title(f'min={topologoy_count_array.min()}, max={topologoy_count_array.max()}')
    plt.savefig('confidence'+'.jpg')
    print('end')

          


