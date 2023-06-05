"""测试将车道语言转回车道后的评价指标计算"""
import faulthandler
faulthandler.enable()
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))		# 用两种路径都是可以的
import numpy as np
import matplotlib.pyplot as plt
from openlanev2.io import io
from openlanev2.preprocessing import collect
from openlanev2.dataset import Collection, Frame
from openlanev2.visualization.utils import interp_arc, COLOR_DEFAULT, COLOR_DICT, THICKNESS
from utils.keyPointsGenerator import keyPointsGenerator 
from utils.laneLanguage_utils import *
from utils.readloadjson import *
from openlanev2.evaluation import evaluate
from openlanev2.utils import format_metric

import copy
import json
from tqdm import tqdm


BEV_SCALE = 10
BEV_RANGE = [-50, 50, -25, 25]
ROOT_PATH = '/mnt/share_disk/wsq/OpenLane-V2/data/OpenLane-V2'
COLLECTION = 'data_dict_subset_A_val'
PRED_PATH = "/mnt/map/zhaohp/MapTR_2d_mf_kp/test_results_aux100_mix100_mix400.json" # 预测结果
PRED_SAVEDIR = "results_test/imgs" # 预测结果可视化
GT_SAVEDIR = "results_test/imgs_gt" # 真值可视化
ISDRAW = True # 画图还是只测评价指标(False)


if __name__ == "__main__":
    if not os.path.exists(PRED_SAVEDIR):
        os.makedirs(PRED_SAVEDIR)
    if not os.path.exists(GT_SAVEDIR):
        os.makedirs(GT_SAVEDIR)
    """读取原始数据"""
    root_path = ROOT_PATH
    data_dict=io.json_load(f'{root_path}/data_dict_subset_A.json')
    collection = Collection(root_path, root_path, collection=COLLECTION)
    tokens=[collection.keys[i][1]+'_'+collection.keys[i][2] for i in range(len(collection.keys))]
    
    """读生成的车道语言"""
    pred_path=PRED_PATH
    with open(pred_path,encoding='utf-8') as a:
        language_seq_pred=json.load(a)
    nnn = 0
    
    """保存到字典中"""
    language_data={}
    gt_anno={}
    language_data['results']={}
    for full_key, value in language_seq_pred.items():
        # if nnn>=10:
        #     break

        print('---------------', nnn, '-', full_key, '---------------')
        
        seq_pred = np.array(value['res']) 
        kps_pred = keyPointsGenerator(seq_pred)

        """ 将它转成以线存储的方法 """
        arclines=kps_pred.getLinesNotMerge(kps_pred.seqXY)
        arclines=changeLineFormat(arclines) # 对线里面的 control points 进行排序
        
        """ 插值 """
        arclines=inter_lines(arclines,t=50)


        """得到邻接矩阵"""
        matrix=getLineAdjMatrix(arclines)
        # matrix=getLineAdjMatrixNoConstant(arclines,kps_pred.seqXY)

        arclines=fitSplines(arclines,interval_nums=50) # 等间隔201
        """直接等间隔采样11个点"""
        # arclines = directSample11(arclines)
        
        """按照原代码中的方法生成贝塞尔控制点；再利用贝塞尔控制点拟合回线"""
        arclines=turn2bezier(arclines)
        arclines=fitBezier(arclines)
        
        lane_centerline=tuple2np(arclines)
        
        if ISDRAW:
            fig, ax= plotLines2(lane_centerline,isdot=True)
            img_pred_dir=os.path.join(PRED_SAVEDIR,f'{full_key}.png')
            fig.savefig(img_pred_dir)

        """找到原来的真值"""
        # idx=tokens.index(full_key)
        # frame_info=collection.keys[idx]
        # frame=collection.get_frame_via_identifier(frame_info)
        # assert frame_info[1]+'_'+frame_info[2]==full_key
        # # 获得frame的数据
        # frame_anno=frame.get_annotations()
        
        # gt_anno[full_key]={}
        # gt_anno[full_key]['annotation']=frame_anno

        # # 画真值
        # if ISDRAW:
        #     fig,ax=plotLines2(frame_anno['lane_centerline'],isdot=False)
        #     img_gt_dir=os.path.join(GT_SAVEDIR,f'{full_key}.png')
        #     fig.savefig(img_gt_dir)
        
        plt.close()
        
        """修改格式"""
    #     predictions={}
    #     predictions['predictions']={}
    #     predictions['predictions']['lane_centerline']=lane_centerline
    #     predictions['predictions']['traffic_element']=frame_anno['traffic_element']
    #     predictions['predictions']['topology_lclc']=matrix
    #     if len(lane_centerline)==frame_anno['topology_lcte'].shape[0]:
    #         predictions['predictions']['topology_lcte']=frame_anno['topology_lcte'] #np.zeros((len(lane_centerline),len(frame_anno['traffic_element'])))
    #     else:
    #         predictions['predictions']['topology_lcte']=np.zeros((len(lane_centerline),len(frame_anno['traffic_element'])))
        
    #     for ele in predictions['predictions']['traffic_element']:
    #         ele['confidence']=1

    #     for i,ele in enumerate(predictions['predictions']['traffic_element']):
    #         ele['id']=str(i)+'10000'

    #     language_data['results'][full_key]=predictions

        nnn += 1
        
    # language_data['method']='lane language'
    # language_data['authors']=['oneline']
    # language_data['e-mail']='1111'
    # language_data['institution / company']='1111'
    # language_data['country / region']='China'

    # eva=evaluate(ground_truth=gt_anno,predictions=language_data)
    # format_metric(eva)
    
    print('end')