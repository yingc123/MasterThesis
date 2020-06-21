import numpy as np
import math
from skimage.io import imread, imsave
import matplotlib.pyplot as plt
from bresenham import bresenham
from collections import defaultdict
import argparse
import os
os.environ['OPENCV_IO_MAX_IMAGE_PIXELS']=str(2**64)
import cv2
from sklearn.linear_model import LinearRegression


def erode(img):
    kernel = np.ones((3,3),np.uint8)
    erosion = cv2.erode(img, kernel, iterations = 1)
    return (img-erosion)

def dilate(img):
    kernel = np.ones((3,3),np.uint8)
    dilation = cv2.dilate(img, kernel, iterations = 1)
    return (dilation-img)

def closing(img):
    kernel = np.ones((5, 5), np.uint8)
    closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    return closing

def opening(img):
    kernel = np.ones((5, 5), np.uint8)
    opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    return opening

def foreground_boundary(data_source):
    f_path = os.path.join(data_source, 'full_mask')
    b_path = os.path.join(data_source, 'boundary')
    for file_name in os.listdir(f_path):
        #print(file_name)
        img = imread(os.path.join(f_path, file_name), plugin='tifffile')
        new_img = np.zeros(np.squeeze(img).shape, dtype=np.uint16)
        for idx,slice in enumerate(img):
            #print(idx)
            opened_slice = opening(slice)
            closed_slice = closing(opened_slice)
            new_img[idx] = dilate(closed_slice)
        imsave(os.path.join(b_path, file_name), new_img, plugin='tifffile')

def linear_regression(tps, inner_source, outer_source, inner_save, outer_save, save_path, slice, num_points):
    #set the parameter of the image shape, time points
    linear_X = np.array([[i] for i in range(tps)])
    image_shape = (slice, num_points+1)
    print(image_shape)
    #new surface map
    inner_reg_score = np.zeros(image_shape[0]*image_shape[1], dtype=float)
    outer_reg_score = np.zeros(image_shape[0] * image_shape[1], dtype=float)
    #print(reg_score.shape)
    inner_surface_map = imread(inner_source, plugin='tifffile')
    outer_surface_map = imread(outer_source, plugin='tifffile')
    print(inner_surface_map.shape, outer_surface_map.shape)

    # do linear regression
    print('slice index is:')
    for i in range(image_shape[0]):
        print(i)
        for j in range(image_shape[1]):
            idx = i*image_shape[1]+j

            inner_linear_y = np.squeeze(inner_surface_map[:,i,j])
            inner_reg = LinearRegression().fit(linear_X, inner_linear_y)
            inner_reg_score[idx] = inner_reg.score(linear_X, inner_linear_y)

            outer_linear_y = np.squeeze(outer_surface_map[:,i,j])
            outer_reg = LinearRegression().fit(linear_X, outer_linear_y)
            outer_reg_score[idx] = outer_reg.score(linear_X, outer_linear_y)

            for idx in range(tps):

                inner_value = np.round(inner_reg.predict(linear_X[idx].reshape(-1, 1)), 0)
                inner_surface_map[idx, i, j] = int(inner_value)

                outer_value = np.round(outer_reg.predict(linear_X[idx].reshape(-1, 1)), 0)
                outer_surface_map[idx, i, j] = int(outer_value)

    np.save(os.path.join(save_path, 'inner_linear_score.npy'), inner_reg_score)
    np.save(os.path.join(save_path, 'outer_linear_score.npy'), outer_reg_score)
    imsave(inner_save, inner_surface_map.astype(np.uint16))
    imsave(outer_save, outer_surface_map.astype(np.uint16))

def smoothing(tps, inner_source, outer_source, inner_save, outer_save):
    kernel = np.ones((5, 5), np.float32) / 25
    inner_surface_map = imread(inner_source, plugin='tifffile')
    outer_surface_map = imread(outer_source, plugin='tifffile')

    for idx in range(tps):
        inner_slice = inner_surface_map[idx]
        inner_filter = cv2.filter2D(inner_slice, -1, kernel)
        inner_surface_map[idx] = inner_filter
        outer_slice = outer_surface_map[idx]
        outer_filter = cv2.filter2D(outer_slice, -1, kernel)
        outer_surface_map[idx] = outer_filter
    imsave(inner_save, inner_surface_map.astype(np.uint16))
    imsave(outer_save, outer_surface_map.astype(np.uint16))

def sequence_circle(num_points, dist_sequence, center):
    angle_array = np.linspace(0, 2*np.pi, num_points)
    x_array = np.zeros(num_points)
    y_array = np.zeros(num_points)
    for idx, angle in enumerate(angle_array):
        x_coor = dist_sequence[idx] * np.cos(angle)
        y_coor = dist_sequence[idx] * np.sin(angle)
        x_array[idx] = round(x_coor) + center[0]
        y_array[idx] = round(y_coor) + center[1]
    return x_array.astype(int), y_array.astype(int)

def surfacemap2volume(inner_surface_map, outer_surface_map, value, img_size, save_path, name_list):
    assert inner_surface_map.shape == outer_surface_map.shape, 'the shapes of the inner map and the outer map are not match'
    slices = inner_surface_map.shape[1]
    num_points = inner_surface_map.shape[2]
    center = (img_size / 2, img_size / 2)
    print('time points are: ')
    #each time frames
    for tp, file_name in enumerate(name_list):
        print(tp)
        volume_inner = np.zeros((slices, img_size, img_size), dtype=np.uint16)
        volume_outer = np.zeros((slices, img_size, img_size), dtype=np.uint16)
        #each slice
        for slice in range(slices):

            dist_inner = inner_surface_map[tp, slice]
            dist_inner = np.round(dist_inner/float(value)*(img_size/2-1),0).astype(int)
            rr_inner, cc_inner = sequence_circle(num_points, dist_sequence=dist_inner, center=center)
            volume_inner[slice, rr_inner, cc_inner] = value

            dist_outer = outer_surface_map[tp, slice]
            dist_outer = np.round(dist_outer/float(value)*(img_size/2-1),0).astype(int)
            rr_outer, cc_outer = sequence_circle(num_points, dist_sequence=dist_outer, center=center)
            volume_outer[slice, rr_outer, cc_outer] = value

        imsave(os.path.join(save_path, file_name), volume_inner+volume_outer, plugin='tifffile')



def calculateDistance(x1,y1,x2,y2):
     dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
     return round(dist, 8)

def initial_circle(num_points, radius, center):
    angle_array = np.linspace(0, 2*np.pi, num_points+1)
    x_array = np.zeros(num_points+1)
    y_array = np.zeros(num_points + 1)
    for idx, angle in enumerate(angle_array):
        x_coor = radius * np.cos(angle)
        y_coor = radius * np.sin(angle)
        x_array[idx] = round(x_coor) + center[0]
        y_array[idx] = round(y_coor) + center[1]
    return x_array.astype(int), y_array.astype(int)

def pad_or_truncate(some_list, target_len):
    return some_list[:target_len] + [(0,0)]*(target_len - len(some_list))

def radial_line(center, initial_rr, initial_cc):
    assert len(initial_rr) == len(initial_cc), 'Number of rows and columns does not match'

    bresenham_array = np.zeros((len(initial_cc), 128, 2), dtype=int)

    for point_idx in range(len(initial_rr)):
        x0 = initial_rr[point_idx]
        y0 = initial_cc[point_idx]
        bresenham_list = list(bresenham(x0, y0, center[0], center[1]))
        if len(bresenham_list)<128:
            bresenham_list = pad_or_truncate(bresenham_list, 128)
        b_array=np.array(bresenham_list)
        bresenham_array[point_idx] = b_array
    return bresenham_array

def test_closed(list_tuple):
    list.sort(list_tuple)
    list_tuple_new = []
    for i in range(len(list_tuple)-1):
        point_current = list_tuple[i]
        point_next = list_tuple[i+1]
        if (point_current[0] == point_next[0]) & (np.abs(point_current[1]-point_next[1])==1):
            pass
        elif (point_current[1] == point_next[1]) & (np.abs(point_current[0]-point_next[0])==1):
            pass
        else:
            list_tuple_new.append(point_current)
    list_tuple_new.append(list_tuple[-1])
    return list_tuple_new

def intersection_lists(center, initial_rr, initial_cc, point_list):
    assert len(initial_rr) == len(initial_cc), 'Number of rows and columns does not match'
    boundary_noise = [(0,0)]
    for point_idx in range(len(initial_rr)):
        x0 = initial_rr[point_idx]
        y0 = initial_cc[point_idx]
        bresenham_list = list(bresenham(x0,y0,center[0], center[1]))
        intersection_points = intersection(bresenham_list, point_list)
        for i in intersection_points:
            boundary_noise.append(i)
    return boundary_noise

def cal_bresenham(center, initial_rr, initial_cc, point_list):
    assert len(initial_rr) == len(initial_cc), 'Number of rows and columns does not match'

    outer_boundary = np.zeros((len(initial_rr), 2), dtype=int)
    inner_boundary = np.zeros((len(initial_rr), 2), dtype=int)
    d = []
    #calculating average distance
    for point_idx in range(len(initial_rr)):
        #print(point_idx)
        x0 = initial_rr[point_idx]
        y0 = initial_cc[point_idx]
        #bresenham_list = list(bresenham(x0,y0,center[0], center[1]))
        bresenham_list = bresenham(x0, y0, center[0], center[1])
        #intersection points of each radial line
        intersection_points = intersection(bresenham_list, point_list)
        # calculate relative distance
        for point in intersection_points:
            dist = calculateDistance(point[0], point[1], center[0], center[1])
            d.append(dist)

    average_dis=sum(d) / len(d)
    print(len(d), average_dis)

    for point_idx in range(len(initial_rr)):
        x0 = initial_rr[point_idx]
        y0 = initial_cc[point_idx]
        bresenham_list = list(bresenham(x0,y0,center[0], center[1]))
        intersection_points = intersection(bresenham_list, point_list)
        #print(intersection_points)
        intersection_points = test_closed(intersection_points)

        # but the intersection points could more than two, choose corresponding points belong to outer/inner boundary
        if len(intersection_points)>2:
            inner = 0
            inner_value = -center[0]
            outer = 0
            outer_value = center[1]
            radial_d = []
            # calculate relative distance
            for idx, point in enumerate(intersection_points):
                dist = calculateDistance(point[0], point[1], center[0], center[1])
                radial_d.append(dist-average_dis)
                if radial_d[idx]<0:
                    inner += 1
                else:
                    outer += 1

            if inner == 1:
                inner_idx = np.argmin(radial_d)
                inner_boundary[point_idx] = intersection_points[inner_idx]
            else:
                for i, dist in enumerate(radial_d):
                    if dist < 0:
                        if dist>inner_value:
                            inner_value = dist
                            inner_boundary[point_idx] = intersection_points[i]

            if outer == 1:
                outer_idx = np.argmax(radial_d)
                outer_boundary[point_idx] = intersection_points[outer_idx]
            else:
                for i, dist in enumerate(radial_d):
                    if dist > 0:
                        if dist<outer_value:
                            outer_value = dist
                            outer_boundary[point_idx] = intersection_points[i]

        elif len(intersection_points)==2:
            d1 = calculateDistance(intersection_points[0][0],intersection_points[0][1],center[0], center[1])
            d2 = calculateDistance(intersection_points[1][0],intersection_points[1][1],center[0], center[1])
            if d1>d2:
                outer_boundary[point_idx] = intersection_points[0]
                inner_boundary[point_idx] = intersection_points[1]
            else:
                inner_boundary[point_idx] = intersection_points[0]
                outer_boundary[point_idx] = intersection_points[1]

        else:
            print(point_idx,'did not has enough intersections, there might has a hole.')
            print('Check prediction images once again.')
    return outer_boundary, inner_boundary, average_dis

def cal_bresenham2(center, initial_rr, initial_cc, point_list):
    outer_boundary = np.zeros((len(initial_rr), 2), dtype=int)
    inner_boundary = np.zeros((len(initial_rr), 2), dtype=int)
    assert len(initial_rr) == len(initial_cc), 'Number of rows and columns does not match'
    #calculate average distance
    for point_idx in range(len(initial_rr)):
        x0 = initial_rr[point_idx]
        y0 = initial_cc[point_idx]
        bresenham_list = list(bresenham(x0,y0,center[0], center[1]))
        intersection_points = intersection(bresenham_list, point_list)
        #print(intersection_points)
        d = []
        # calculate relative distance
        for point in intersection_points:
            dist = calculateDistance(point[0], point[1], center[0], center[1])
            d.append(dist)
    average_dis=sum(d) / len(d)
    print(average_dis)

    #calculate average inner and outer distance
    for point_idx in range(len(initial_rr)):
        x0 = initial_rr[point_idx]
        y0 = initial_cc[point_idx]
        bresenham_list = list(bresenham(x0,y0,center[0], center[1]))
        intersection_points = intersection(bresenham_list, point_list)
        d_inner = []
        d_outer = []
        # calculate relative distance
        for point in intersection_points:
            dist = calculateDistance(point[0], point[1], center[0], center[1])
            if dist<average_dis:
                d_inner.append(dist)
            else:
                d_outer.append(dist)
    average_dis_inner = sum(d_inner) / len(d_inner)
    print(average_dis_inner)
    average_dis_outer = sum(d_outer) / len(d_outer)
    print(average_dis_outer)

    for point_idx in range(len(initial_rr)):
        x0 = initial_rr[point_idx]
        y0 = initial_cc[point_idx]
        bresenham_list = list(bresenham(x0,y0,center[0], center[1]))
        intersection_points = intersection(bresenham_list, point_list)
        #print(len(intersection_points))
        d_inner = []
        point_inner=[]
        d_outer = []
        point_outer=[]
        if len(intersection_points)>2:
            # assign inner class and outer class
            for point in intersection_points:
                dist = calculateDistance(point[0], point[1], center[0], center[1])
                #print(dist)
                if dist < average_dis:
                    d_inner.append(np.absolute(dist-average_dis_inner))
                    point_inner.append((point[0], point[1]))
                else:
                    d_outer.append(np.absolute(dist-average_dis_outer))
                    point_outer.append((point[0], point[1]))

            if len(d_inner) == 1:
                print('yes')
                inner_boundary[point_idx] = point_inner[0]
            else:
                inner_idx=np.argmin(d_inner)
                inner_boundary[point_idx] = point_inner[inner_idx]
            if len(d_outer) == 1:
                outer_boundary[point_idx] = point_outer[0]
            else:
                outer_idx = np.argmin(d_outer)
                outer_boundary[point_idx] = point_outer[outer_idx]

        elif len(intersection_points)==2:
            d1 = calculateDistance(intersection_points[0][0],intersection_points[0][1],center[0], center[1])
            d2 = calculateDistance(intersection_points[1][0],intersection_points[1][1],center[0], center[1])
            if d1>d2:
                outer_boundary[point_idx] = intersection_points[0]
                inner_boundary[point_idx] = intersection_points[1]
            else:
                inner_boundary[point_idx] = intersection_points[0]
                outer_boundary[point_idx] = intersection_points[1]

        else:
            print(point_idx,'did not has enough intersections, there might has a hole.')
            print('Check prediction images once again.')

    #print(outer_boundary, inner_boundary)
    return outer_boundary, inner_boundary

def tiff_modify():
    path = '/work/scratch/ychen/preprocessing/arabidopsis/plant18/512*512/full_mask'
    for x in os.listdir(path):
        im = imread(os.path.join(path, x), plugin='tifffile')
        for i in im:
            i[i > 1] = 65535
            i[i <= 1] = 0
        imsave(os.path.join(path, x), im, plugin='tifffile')

def intersection(lst1, lst2):
    return list(set(lst1) & set(lst2))

def slicemap(inner_boundary, outer_boundary, center):
    assert len(inner_boundary)==len(outer_boundary), 'Number of inner and outer points do not match'
    inner_line = np.zeros(len(inner_boundary), dtype=np.float64)
    for idx, inner_point in enumerate(inner_boundary):
        dist = calculateDistance(inner_point[0], inner_point[1], center[0], center[1])
        inner_line[idx] = dist
    outer_line = np.zeros(len(outer_boundary), dtype=np.float64)
    for idx, outer_point in enumerate(outer_boundary):
        dist = calculateDistance(outer_point[0], outer_point[1], center[0], center[1])
        outer_line[idx] = dist
    return inner_line, outer_line

def img2point(pred_img, value):
    point_list = []
    for x_value, array in enumerate(pred_img):
        for y_value,element in enumerate(array):
            if element==value:
                point_list.append((x_value, y_value))
    #print('len(point clouds) is: ', len(point_list))
    return point_list

def args_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_points', type=int, default=1000, help='choose how many points for initial circle')
    parser.add_argument('--bit_value', type=int, default=16, help='different intensity')
    parser.add_argument('--img_size', type=int, default=8, help='image size')
    parser.add_argument('--time_frame', type=int, default=21, help='time points')
    parser.add_argument('--num_slice', type=int, default=318, help='different intensity')
    parser.add_argument('--linear_first', type=bool, default=True, help='linear regression first, then spatial smoothing')
    parser.add_argument('--dataset', type=str, default='drosophila_new_1',
                        help='which dataset')

    return parser.parse_args()

def main():
    args = args_parse()
    value = 2**(args.bit_value) - 1
    img_size = 2 ** (args.img_size)
    data_source = os.path.join('/work/scratch/ychen/segmentation/smoothing', args.dataset)
    name_list=[]
    for name in os.listdir(os.path.join(data_source, 'full_mask')):
        name_list.append(name)
        name_list.sort()
    np.save(os.path.join(data_source, 'file_name.npy'), name_list)
    name_list = np.load(os.path.join(data_source, 'file_name.npy'), allow_pickle=True)
    print(name_list)
    ###############################################################################
    ''' full mask to boundary '''
    ###############################################################################
    step_one=True
    if step_one:
        foreground_boundary(data_source)

    ###############################################################################
    ''' boundary to surface map '''
    ###############################################################################
    step_two = False
    if step_two:
        two_D = False
        three_d = True
        center = (int(img_size / 2), int(img_size / 2))
        radius=img_size / 2 -1
        print('center is:', center, 'radius is:', radius)

        if two_D:
            fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(10, 10))
            img = np.zeros((256, 256), dtype=np.uint8)
            rr, cc = initial_circle(num_points=args.num_points, radius=radius, center=center)
            save_path = r'C:\Users\yingc\Desktop\Semester_5\MasterThesis_LFB\ThesisTemplete\ThesisTemplate_EN\images\img'

            #initial circle
            img[rr, cc] = value
            #imsave(os.path.join(save_path, 'cirlce.png'), img)
            axs[0][0].imshow(img, cmap=plt.cm.gray)
            axs[0][0].set_title('initial circle')
            axs[0][0].axis('off')

            #pred mask
            img_path=r'C:\Users\yingc\Desktop\Semester_5\MasterThesis_LFB\ThesisTemplete\ThesisTemplate_EN\images\img\origin.jpg'

            pred_img = imread(img_path, as_gray=True)
            #print(pred_img.max())
            pred_img[pred_img >= 0.5] = 1
            pred_img[pred_img < 0.5] = 0
            #imsave(os.path.join(save_path, 'origin.png'), pred_img)
            axs[0][1].imshow(pred_img, cmap=plt.cm.gray)
            axs[0][1].set_title('prediction')
            axs[0][1].axis('off')

            #surface fitting for both outer boundary and inner boundary
            point_list = img2point(pred_img, value)
            outer_boundary, inner_boundary = cal_bresenham2(center=center, initial_rr=rr, initial_cc=cc, point_list=point_list)

            bresenham_img = np.zeros((256, 256), dtype=np.uint8)
            bresenham_array = radial_line(center=center, initial_rr=rr, initial_cc=cc)
            for point_idx in range(args.num_points):
                radius_line = bresenham_array[point_idx]
                for i in radius_line:
                    bresenham_img[i[0], i[1]] = value
            bresenham_img[0][0]=0
            #imsave(os.path.join(save_path, 'radius.png'), bresenham_img)
            axs[1][0].imshow(bresenham_img, cmap=plt.cm.gray)
            axs[1][0].set_title('outer_boundary')
            axs[1][0].axis('off')

            boundary_noise_img = np.zeros((256, 256), dtype=np.uint8)
            boundary_list = intersection_lists(center=center, initial_rr=rr, initial_cc=cc, point_list=point_list)
            for i in boundary_list:
                boundary_noise_img[i[0], i[1]] = value
            boundary_noise_img[0][0] = 0
            # imsave(os.path.join(save_path, 'boundary_noise2.png'), boundary_noise_img)
            # axs[1][1].imshow(boundary_noise_img, cmap=plt.cm.gray)
            # axs[1][1].set_title('inner_boundary') #outer
            # axs[1][1].axis('off')


            outer_img = np.zeros((256, 256), dtype=np.uint8)
            for i in outer_boundary:
                outer_img[i[0], i[1]] = value
            inner_img = np.zeros((256, 256), dtype=np.uint8)
            for i in inner_boundary:
                inner_img[i[0], i[1]] = value
            #imsave(os.path.join(save_path, 'filtered_boundary2.png'), inner_img+outer_img)
            axs[1][1].imshow(inner_img+outer_img, cmap=plt.cm.gray)
            axs[1][1].set_title('inner_boundary')
            axs[1][1].axis('off')

            plt.show()
        if three_d:
            path = os.path.join(data_source,'boundary')
            save_path = os.path.join(data_source, 'filter_boundary')
            tps = args.time_frame
            image_shape = (args.num_slice, args.num_points+1)
            inner_surface_map = np.zeros((tps,image_shape[0],image_shape[1]), dtype=np.uint16)
            outer_surface_map = np.zeros((tps, image_shape[0], image_shape[1]), dtype=np.uint16)
            print('time frame idx:')
            avr_dist_array = np.zeros((args.time_frame, args.num_slice), dtype=float)
            for tp, file_name in enumerate(name_list):
                boundary_img = imread(os.path.join(path, file_name), plugin='tifffile')
                print(tp)
                inner_filtered_boundary = np.zeros((args.num_slice, img_size, img_size), dtype=np.int16)
                outer_filtered_boundary = np.zeros((args.num_slice, img_size, img_size), dtype=np.int16)
                for idx, img_slice in enumerate(boundary_img):
                    #print(idx)
                    # initial circle
                    rr, cc = initial_circle(num_points=args.num_points, radius=radius, center=center)
                    # surface fitting for both outer boundary and inner boundary
                    point_list = img2point(img_slice, value)
                    outer_boundary, inner_boundary, avr_dist = cal_bresenham(center=center, initial_rr=rr, initial_cc=cc,
                                                                   point_list=point_list)
                    avr_dist_array[tp,idx] = avr_dist
                    for i in outer_boundary:
                        outer_filtered_boundary[idx, i[0], i[1]] = value
                    for i in inner_boundary:
                        inner_filtered_boundary[idx, i[0], i[1]] = value
                    inner_line, outer_line = slicemap(inner_boundary, outer_boundary, center)
                    inner_surface_map[tp, idx] = inner_line/radius*value
                    outer_surface_map[tp, idx] = outer_line/radius*value

                imsave(os.path.join(save_path, 'inner_filter{}.tif'.format(tp)), inner_filtered_boundary.astype(np.uint16),
                       plugin='tifffile')
                imsave(os.path.join(save_path, 'outer_filter{}.tif'.format(tp)), outer_filtered_boundary.astype(np.uint16),
                       plugin='tifffile')
            imsave(os.path.join(data_source, 'inner_surface_map.tif'), inner_surface_map.astype(np.uint16),plugin='tifffile')
            imsave(os.path.join(data_source, 'outer_surface_map.tif'), outer_surface_map.astype(np.uint16),plugin='tifffile')
        np.save(os.path.join(data_source, 'avr_dist.npy'), avr_dist_array)
        print('Inner and outer surface map are saved!')

    ###############################################################################
    ''' linear regression and smoothing '''
    ###############################################################################
    step_three = False
    if step_three:
        inner_source = os.path.join(data_source, 'inner_surface_map.tif')
        outer_source = os.path.join(data_source, 'outer_surface_map.tif')
        if args.linear_first:
            print('linear regression first')
            dst_path = os.path.join(data_source, 'regression_smoothing')
            inner_regression = os.path.join(dst_path,'inner_regression.tif')
            outer_regression = os.path.join(dst_path, 'outer_regression.tif')
            inner_smoothing = os.path.join(dst_path,'inner_smoothing.tif')
            outer_smoothing = os.path.join(dst_path, 'outer_smoothing.tif')
            linear_regression(tps = args.time_frame, inner_source=inner_source, outer_source=outer_source,
                              inner_save=inner_regression, outer_save=outer_regression, save_path=dst_path,
                              slice=args.num_slice, num_points=args.num_points)
            smoothing(tps = args.time_frame, inner_source=inner_regression, outer_source=outer_regression,
                      inner_save=inner_smoothing, outer_save=outer_smoothing)

        else:
            print('spatial smoothing first')
            dst_path=os.path.join(data_source, 'smoothing_regression')
            inner_smoothing = os.path.join(dst_path,'inner_smoothing.tif')
            outer_smoothing = os.path.join(dst_path, 'outer_smoothing.tif')
            inner_regression = os.path.join(dst_path,'inner_regression.tif')
            outer_regression = os.path.join(dst_path, 'outer_regression.tif')
            smoothing(tps = args.time_frame, inner_source=inner_source, outer_source=outer_source,
                      inner_save=inner_smoothing, outer_save=outer_smoothing)
            linear_regression(tps = args.time_frame, inner_source=inner_smoothing, outer_source=outer_smoothing,
                              inner_save=inner_regression, outer_save=outer_regression, save_path=dst_path,
                              slice=args.num_slice, num_points=args.num_points)

    ###############################################################################
    ''' surface map to volumetrice data '''
    ###############################################################################
    step_four = False
    if step_four:
        print(value, img_size)
        if args.linear_first:
            result_path = os.path.join(data_source, 'result_1')
            inner_file = os.path.join(data_source, 'regression_smoothing/inner_smoothing.tif')
            outer_file = os.path.join(data_source, 'regression_smoothing/outer_smoothing.tif')
        else:
            result_path = os.path.join(data_source, 'result_2')
            inner_file = os.path.join(data_source, 'smoothing_regression/inner_smoothing.tif')
            outer_file = os.path.join(data_source, 'smoothing_regression/outer_smoothing.tif')
        inner_surface_map = imread(inner_file,plugin='tifffile')
        outer_surface_map = imread(outer_file,plugin='tifffile')
        surfacemap2volume(inner_surface_map, outer_surface_map, value, img_size, result_path, name_list)

if __name__=='__main__':
    main()