import cv2
import os
import os.path as osp
import mmcv

CAMS = ['CAM_FRONT_LEFT','CAM_FRONT','CAM_FRONT_RIGHT',
             'CAM_BACK_LEFT','CAM_BACK','CAM_BACK_RIGHT']

gt_path = 'results_bezier/imgs_gt'
pred_path = 'results_bezier/imgs'
result_path = 'results_bezier/GtPred'


if __name__ == '__main__':
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    
    file_list = os.listdir(gt_path)
    for file in file_list:
        bev_fig_path = osp.join(gt_path, file)
        # print(bev_fig_path)
        bev_fig = cv2.imread(bev_fig_path)
        bev_fig = cv2.resize(bev_fig, (1000, 500))
        # bev_fig = bev_fig[130:1180, 70:1150]

        v6_fig_path = osp.join(pred_path, file)
        # print(v6_fig_path)
        v6_fig = cv2.imread(v6_fig_path)
        v6_fig = cv2.resize(v6_fig, (1000, 500))

        # print(bev_fig.shape)
        # print(v6_fig.shape)

        fig = cv2.vconcat([bev_fig, v6_fig])

        cv2.putText(fig, 'GT', (480, 45), 0, 0.8,
                    color=(0, 0, 0), thickness=1, lineType=cv2.LINE_AA)

        cv2.putText(fig, 'Pred', (480, 555), 0, 0.8,
                    color=(0, 0, 0), thickness=1, lineType=cv2.LINE_AA)

        save_path = osp.join(result_path, file)
        
        cv2.imwrite(save_path, fig)
        print(save_path)
