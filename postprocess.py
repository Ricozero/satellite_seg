from skimage import io
import numpy as np

def edge1(img):
    '''
    绘制边界，上下左右不是同色的像素被标记为边界。
    使用逐个像素判断的方法。
    '''
    edge_img = np.zeros_like(img)
    for x in range(0, img.shape[0]):
        for y in range(0, img.shape[1]):
            if y - 1 >= 0:
                if img[x][y] != img[x][y - 1]:
                    edge_img[x][y] = img[x][y]
                    continue
            if y + 1 < img.shape[1]:
                if img[x][y] != img[x][y + 1]:
                    edge_img[x][y] = img[x][y]
                    continue
            if x - 1 >= 0:
                if img[x][y] != img[x - 1][y]:
                    edge_img[x][y] = img[x][y]
                    continue
            if x + 1 < img.shape[0]:
                if img[x][y] != img[x + 1][y]:
                    edge_img[x][y] = img[x][y]
                    continue
    return edge_img

def edge2(img):
    ''' 
    使用skimage.measure.find_contours
    '''
    # 坐标需要考虑0.5的偏移
    pass

labels = io.imread('results/test.png')
e = edge1(labels)
io.imshow(e)

io.imsave('results/test_edge.png', e)
save_gt_vis('results/test_edge.png', 'results/test_edge_vis.png')