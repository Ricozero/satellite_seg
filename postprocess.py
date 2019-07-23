from skimage import io
import numpy as np

from utils.preprocess import save_gt_vis

def label2edge(img, diffuse):
    '''
    绘制边界，上下左右不是同色的像素被标记为边界。
    使用逐个像素判断的方法。
    边界可以指定扩散的宽度。

    该方法可能需要一定的优化。

    还有一种方法是使用skimage.measure.find_contours，
    坐标需要考虑0.5的偏移。
    '''
    edge_img = np.zeros_like(img)
    for x in range(0, img.shape[0]):
        for y in range(0, img.shape[1]):
            if (y - 1 >= 0 and img[x][y] != img[x][y - 1] or
                y + 1 < img.shape[1] and img[x][y] != img[x][y + 1] or
                x - 1 >= 0 and img[x][y] != img[x - 1][y] or
                x + 1 < img.shape[0] and img[x][y] != img[x + 1][y]):
                edge_img[x][y] = img[x][y]

                # 标记为边缘的像素向外扩散diffuse个像素（围绕一圈，形成正方形）
                # 仅扩散到同色的像素
                # 考虑一圈都没有扩散之后，后面不再扩散
                for i in range(0, diffuse):
                    

    return edge_img

def find_label_edge(filename):
    labels = io.imread(filename)
    e = label2edge(labels)
    io.imshow(e)

    pos = filename.rfind('.')
    if pos == -1:
        efn = filename + '_edge'        # edge file name
        evfn = filename + '_edge_vis'   # edge visualised file name
    else:
        efn = filename[0:pos] + '_edge' + filename[pos:]
        evfn = filename[0:pos] + '_edge_vis' + filename[pos:]

    io.imsave(efn, e)
    save_gt_vis(efn, evfn)

find_label_edge('results/test_small.png')
