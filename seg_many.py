# -*-coding: utf-8 -*-

from PIL import Image
import cv2, os
import gdal
from graph import build_graph, segment_graph
from smooth_filter import gaussian_grid, filter_image
from random import random
import numpy as np
from numpy import sqrt

H, W = 650, 650

SIGMA = 0.5
K = 400.
MIN_SIZE = 100
neighbor = 8


def read_tif(path, H, W, band_list=[4, 2, 1]):
    '''
    band list is [1~8]. Among them, [2, 3, 5] is [B, G, R]
    raw_img [0, 1, 2, 3, 4, 5, 6, 7, 8]
    is [coastal, blue, green, yellow, red(4), red edge, near-IR1, near-IR2], dytpe is uint16
    [7] is the first NearIR
    :param path:
    :param band_list:
    :return:
    '''
    img_obj = gdal.Open(path)
    raw_img = img_obj.ReadAsArray(0, 0, H, W)  # 163 # [C, H, W]
    vis_bands = []
    # print("the raw dtype is :{} ...But change it to float64.".format(raw_img.dtype))

    raw_img = np.asarray(raw_img, dtype=np.float64)
    # print(raw_img.dtype)

    for band_id in band_list:
        tmp_band = raw_img[band_id]
        tmp_band = np.array(tmp_band*255.0/np.max(tmp_band), dtype=np.uint8)
        vis_bands.append(tmp_band)
    return raw_img, np.stack(vis_bands, axis=2)


def diff_geoTif(img, x1, y1, x2, y2):

    val_sum = 0.0
    for a_band in img:
        val_sum += (a_band[x1, y1] - a_band[x2, y2]) **2

    return sqrt(val_sum)

def diff_rgb(img, x1, y1, x2, y2):
    r = (img[0][x1, y1] - img[0][x2, y2]) ** 2
    g = (img[1][x1, y1] - img[1][x2, y2]) ** 2
    b = (img[2][x1, y1] - img[2][x2, y2]) ** 2
    return sqrt(r + g + b)


def diff_grey(img, x1, y1, x2, y2):
    v = (img[x1, y1] - img[x2, y2]) ** 2
    return sqrt(v)


def threshold(size, const):
    return (const / size)


def generate_image(forest, width, height):
    random_color = lambda: (int(random()*255), int(random()*255), int(random()*255))
    colors = [random_color() for i in xrange(width*height)]

    img = Image.new('RGB', (width, height))
    im = img.load()
    for y in xrange(height):
        for x in xrange(width):
            comp = forest.find(y * width + x)
            im[x, y] = colors[comp]

    return img.transpose(Image.ROTATE_270).transpose(Image.FLIP_LEFT_RIGHT)


def seg_a_img(img, img_name, img_mode="remote_sensing"):

    grid = gaussian_grid(SIGMA)

    if img_mode == "remote_sensing":
        c, h, w = img.shape[0], img.shape[1], img.shape[2]
        print ("we are working in geo-tif mode, img is read by gdal")
        img_bands = []
        for i in range(c):
            img_bands.append(filter_image(img[i], grid))
        smooth = tuple(img_bands)
        diff = diff_geoTif

    elif img_mode == "BGR":
        print ("we are working in BGR mode, the img is read by opencv")
        h, w, c = img.shape[0], img.shape[1], img.shape[2]

        smooth = []
        for i in range(3):
            smooth.append(filter_image(img[:, :, i], grid))
        smooth = tuple(smooth)
        diff = diff_rgb
    else:
        h, w = img.shape[0], img.shape[1]
        smooth = filter_image(img, grid)
        diff = diff_grey

    graph = build_graph(smooth, w, h, diff, neighbor == 8)
    forest = segment_graph(graph, h*w, K, MIN_SIZE, threshold)

    image = generate_image(forest, w, h)
    image.save("seg_res/%s.png" % img_name)
    # image.save(sys.argv[6])
    print 'Number of components: %d' % forest.num_sets



def seg_many(imglist_path):

    ROOT_PATH  = "/home/yjr/DataSet/SpaceNet"
    def get_name(all_name):
        name = all_name.strip().split("PanSharpen_")[1].split(".jpg")[0]
        root_name = name.split("_img")[0]

        return root_name, name
    with open(imglist_path) as f:
        img_list = f.readlines()

    for i, a_name in enumerate(img_list):
        print (i)
        dataset_name, name = get_name(a_name)
        img_path = os.path.join(ROOT_PATH, dataset_name+"_Train", "MUL-PanSharpen",
                                "MUL-PanSharpen_%s.tif" % name)
        # vis_some_results(img_path, img_name=name, save_res=True)
        raw_img, _ = read_tif(path=img_path, W=650, H=650)
        mbi = np.load("/home/yjr/PycharmProjects/MBI_win/MBI/data/res/raw_data/%s_mbi.npy" % name)
        mbi = np.array(mbi, dtype=float)

        combine_img = np.concatenate((raw_img, np.expand_dims(mbi, axis=0)), axis=0)
        seg_a_img(combine_img, name)
        print(20*"_")


if __name__ == '__main__':

    # img = cv2.imread("/home/yjr/PycharmProjects/MBI_win/MBI/data/res/viewed_data/Four_Vegas_img96_mbi.png")
    # name = 'Four_Vegas_img96_mbi'
    # # seg_a_img(img, name, img_mode="BGR")
    # img, _ = read_tif(path='/home/yjr/PycharmProjects/MBI_win/MBI/data/Four_Vegas_img96.tif', W=650, H=650)
    # mbi = np.load('/home/yjr/PycharmProjects/MBI_win/MBI/data/res/raw_data/Four_Vegas_img96_mbi.npy')
    # img = np.concatenate((img, np.expand_dims(mbi, axis=0)), axis=0)
    # name = "TIF"
    # seg_a_img(img, name)
    seg_many("/home/yjr/DataSet/SpaceNet/AOall_pascal/test_imgs_list.txt")