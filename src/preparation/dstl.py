import os
import csv
import shapely.affinity
import shapely.wkt
import tifffile as tiff
import numpy as np
import cv2
from skimage import io
from tqdm import tqdm

from generate_data import generate_stat, generate_dataset

sys.path.append('./src')
from utils.imgformat import convert_label_to_vis

dstl_folder = 'dataset/dstl'
preprocessed_folder = os.path.join(dstl_folder, 'preprocessed')
train_folder = 'dataset/dstl-train'

# 原始分类到本项目的分类的映射
type_projection = [
    0,
    2,  # 1.Buildings
    2,  # 2.Misc. Manmade structures
    4,  # 3.Road
    4,  # 4.Track
    1,  # 5.Trees
    1,  # 6.Crops
    3,  # 7.Waterway
    3,  # 8.Standing water
    0,  # 9.Vehicle Large
    0   # 10.Vehicle Small
]

# 矢量多边形生成类型掩膜
def mask_for_polygons(poly_list, img_size):
    img_mask = np.zeros(img_size, np.uint8)
    if not poly_list:
        return img_mask
    int_coords = lambda x: np.array(x).round().astype(np.int32)
    for poly_type, polygons in poly_list:
        exteriors = [int_coords(poly.exterior.coords) for poly in polygons]
        interiors = [int_coords(pi.coords) for poly in polygons for pi in poly.interiors]
        cv2.fillPoly(img_mask, exteriors, type_projection[poly_type])
        cv2.fillPoly(img_mask, interiors, 0)
    return img_mask

# 使得遥感图像能正常查看
def scale_percentile(matrix):
    w, h, d = matrix.shape
    matrix = np.reshape(matrix, [w * h, d]).astype(np.float64)
    # Get 2nd and 98th percentile
    mins = np.percentile(matrix, 1, axis=0)
    maxs = np.percentile(matrix, 99, axis=0) - mins
    matrix = (matrix - mins[None, :]) / maxs[None, :]
    matrix = np.reshape(matrix, [w, h, d])
    matrix = matrix.clip(0, 1)
    return matrix

def preprocess_dstl():
    # 获取训练集的基本信息
    train_dict = {}
    with open(os.path.join(dstl_folder, 'train_wkt_v4.csv')) as csvfile:
        csv.field_size_limit(2**31-1)
        for i, (img_id, poly_type, poly) in enumerate(csv.reader(csvfile)):
            if i > 0:
                train_dict.setdefault(img_id, []).append((int(poly_type), shapely.wkt.loads(poly)))

    # 获取训练集每个图像的缩放大小
    size_dict = {}
    with open(os.path.join(dstl_folder, 'grid_sizes.csv')) as csvfile:
        for i, (img_id, x, y), in enumerate(csv.reader(csvfile)):
            if i > 0:
                size_dict[img_id]=(float(x), float(y))

    # 依次读取每个图像并进行预处理并输出
    print('total image number:', len(train_dict))
    print('preprocessing...')
    if not os.path.exists(preprocessed_folder):
        os.mkdir(preprocessed_folder)
        for img_id in tqdm(train_dict.keys()):
            #print(img_id, len(train_dict[img_id]))
            img = tiff.imread(os.path.join(
                dstl_folder, 'three_band/{}.tif'.format(img_id))).transpose([1, 2, 0])

            # 计算像素数
            img_size = img.shape[:2]
            h, w = img_size # 为什么是高宽的顺序？
            w_ = w * (w / (w + 1))
            h_ = h * (h / (h + 1))
            x_max, y_min = size_dict[img_id]
            x_scalar, y_scalar = w_/x_max, h_/y_min

            # 将多边形的坐标进行缩放
            poly_list = train_dict[img_id]
            scaled_poly_list = []
            for poly_type, poly in poly_list:
                scaled_poly_list.append((poly_type, shapely.affinity.scale(poly, xfact=x_scalar, yfact=y_scalar, origin=(0,0,0))))

            # 得到掩膜图像
            img_mask=mask_for_polygons(scaled_poly_list, img_size)

            # 将原始遥感图像转化为普通24位色图
            # 如果是乘256，会造成原值为1的分量溢出为0
            img_normal = np.uint8(scale_percentile(img) * 255)

            # 输出3张图
            io.imsave(os.path.join(preprocessed_folder, img_id + '.png'), img_normal)
            io.imsave(os.path.join(preprocessed_folder, img_id + '_class.png'), img_mask, check_contrast=False)
            convert_label_to_vis(os.path.join(preprocessed_folder, img_id + '_class.png'),
                os.path.join(preprocessed_folder, img_id + '_class_vis.png'))
            #io.imshow(img_normal)
            #io.imshow(img_mask)
    else:
        print('folder already exists, skip...')

def generate_dstl():
    # 获得图片id
    img_id_set = set()
    with open(os.path.join(dstl_folder, 'train_wkt_v4.csv')) as csvfile:
        csv.field_size_limit(2**31-1)
        for i, (img_id, poly_type, poly) in enumerate(csv.reader(csvfile)):
            if i > 0:
                img_id_set.add(img_id)

    # 生成文件名列表
    img_list = []
    label_list = []
    for img_id in img_id_set:
        img_list.append(os.path.join(preprocessed_folder, img_id + '.png'))
        label_list.append(os.path.join(preprocessed_folder, img_id + '_class.png'))

    # 统计每个类别的比例
    #stat = generate_stat(label_list)
    #print(np.array(stat)*1.0/np.min(stat[np.nonzero(stat)]))

    # 生成训练集数据
    print('generating...')
    if not os.path.exists(train_folder):
        os.mkdir(train_folder)
        generate_dataset(train_folder, 256, img_list, label_list)
    else:
        print('folder already exists, skip...')

preprocess_dstl()
generate_dstl()