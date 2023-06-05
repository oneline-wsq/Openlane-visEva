import json
import numpy as np
from PIL import Image

save_root0 = 'results/map_result'
save_root1 = 'results/kp_result'

# 读取第一个JSON文件
with open('/mnt/share_disk/wsq/OpenLane-V2/saved_gt_subset_A_val_border_t1/lane_gt.json', 'r') as f:
    dict1 = json.load(f)

# 读取第二个JSON文件
with open('/mnt/map/zhaohp/MapTR_2d_mf_kp/results_map.json', 'r') as f:
    dict2 = json.load(f)

with open('/mnt/map/zhaohp/MapTR_2d_mf_kp/results_kp.json', 'r') as f:
    dict3 = json.load(f)

print('load finished')

# def dice_loss(predictive, target, ep=1e-8):
#     intersection = 2 * np.sum(predictive * target) + ep
#     union = np.sum(predictive) + np.sum(target) + ep
#     loss = 1 - intersection / union
#     return loss

# dice_loss_total = 0.0
# num = 0

# 遍历所有键，将对应的二维列表转换为图像，并将它们合并成一张图像
for key in dict1.keys():
    # 将第一个字典中的值转换为NumPy数组
    array1 = np.array(dict1[key]['semantic_mask'])
    array2 = np.array(dict1[key]['KeyMatrix'][2])
    # 将第二个字典中的值转换为NumPy数组
    array3 = np.array(dict2[key])
    array4 = np.array(dict3[key])

    # loss = dice_loss(array2, array1)
    # dice_loss_total += loss
    # num += 1

    # 将值缩放到0到255之间，并将其转换为8位整数
    array1 = (array1 - np.min(array1)) / (np.max(array1) - np.min(array1)) * 255
    array1 = array1.astype(np.uint8)
    # 创建PIL图像对象
    image1 = Image.fromarray(array1).convert('RGB')

    # 将值缩放到0到255之间，并将其转换为8位整数
    array2 = (array2 - np.min(array2)) / (np.max(array2) - np.min(array2)) * 255
    array2 = array2.astype(np.uint8)
    # 创建PIL图像对象
    image2 = Image.fromarray(array2).convert('RGB')

    # 将值缩放到0到255之间，并将其转换为8位整数
    array3 = (array3 - np.min(array3)) / (np.max(array3) - np.min(array3)) * 255
    array3 = array3.astype(np.uint8)
    # 创建PIL图像对象
    image3 = Image.fromarray(array3).convert('RGB')

    # 将值缩放到0到255之间，并将其转换为8位整数
    array4 = (array4 - np.min(array4)) / (np.max(array4) - np.min(array4)) * 255
    array4 = array4.astype(np.uint8)
    # 创建PIL图像对象
    image4 = Image.fromarray(array4).convert('RGB')

    # 将两个图像水平连接
    merged_image = np.concatenate([np.array(image1), np.array(image3)], axis=1)

    # 将合并后的图像保存为PNG文件，以key命名
    image_filename = save_root0 + '/' + f"{key}.png"
    print(image_filename)
    merged_image = np.uint8(merged_image)
    merged_image = Image.fromarray(merged_image)
    merged_image.save(image_filename)

     # 将两个图像水平连接
    merged_image = np.concatenate([np.array(image2), np.array(image4)], axis=1)

    # 将合并后的图像保存为PNG文件，以key命名
    image_filename = save_root1 + '/' + f"{key}.png"
    print(image_filename)
    merged_image = np.uint8(merged_image)
    merged_image = Image.fromarray(merged_image)
    merged_image.save(image_filename)


# print(dice_loss_total / num)