# -*- coding: utf-8 -*-
"""
Created on Thu Dec 13 16:45:56 2018

@author: eschweiler
Modification: YingChen
"""

import smtplib
import os
import socket
import warnings
import sys
import itertools
import numpy as np
import pandas as pd
import multiprocessing as mp

from functools import partial
from collections import Counter
from email.mime.text import MIMEText
from skimage import morphology, measure
from sklearn import metrics
from sklearn.cluster import DBSCAN
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.cluster.hierarchy import linkage
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance, Delaunay
from scipy.special import sph_harm
from scipy.ndimage import gaussian_filter
from dipy.core.sphere import HemiSphere, Sphere, disperse_charges
from dipy.core.geometry import sphere2cart, cart2sphere
###############################################################################
''' GENERAL STUFF '''


###############################################################################


# send mails for advanced remote control
def send_mail(subj=None, msg_text=None):
    '''
    INPUTS: subj: mail subject string     
            msg_text: mail body text
    '''

    msg = MIMEText(msg_text) if msg_text is not None else MIMEText('')
    sender = 'code_observer@lfb.rwth-aachen.de'
    receiver = 'dennis.eschweiler@lfb.rwth-aachen.de'
    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        subject = '[{} {}]'.format(socket.gethostname(), os.environ['CUDA_VISIBLE_DEVICES'])
    else:
        subject = '[{}]'.format(socket.gethostname())
    if subj is not None:
        subject += ' ' + subj
    msg['Subject'] = subject
    msg['From'] = sender
    msg['To'] = receiver
    s = smtplib.SMTP('smarthost.rwth-aachen.de', port=25)
    s.sendmail(sender, [receiver], msg.as_string())
    s.quit()


def print_continuous(msg='', msg_len=0):
    print('\r' * msg_len + msg, end='\r')
    return len(msg)


# sanity check for patch_params
def sanitycheck_patch_params(patch_params):
    # possible keys
    keys = ['shape', 'start', 'grayscale', 'load_fcn', 'axes_order', 'transforms', 'data_shape', \
            'scale_level', 'scales', 'rotation_count', 'mirror_x', 'mirror_y']

    # if patch_params is no dict, overwrite it as a dict
    if not type(patch_params) is dict:
        # warnings.warn('Patch parameters were not passed as a dict and will be overwritten.')
        patch_params = {}

    # fill in keys
    for key in keys:
        if not key in patch_params: patch_params[key] = None

    return patch_params


# adapt paths to the current os
def sanitycheck_path(path, use_local=True):
    path = path.replace('\\', '/')
    path = os.path.normpath(path)

    if sys.platform == 'win32':
        path = path.replace(os.path.join(os.sep, 'work', 'scratch', 'eschweiler'), 'I:')
        path = path.replace(os.path.join(os.sep, 'home', 'temp', 'eschweiler'), 'K:')
        path = path.replace(os.path.join(os.sep, 'home', 'staff', 'eschweiler'), 'Y:')
        path = path.replace(os.path.join(os.sep, 'images'), 'V:')

    if socket.gethostname() == 'pc53' and use_local:
        path = path.replace(os.path.join('V:' + os.sep, 'BiomedicalImageAnalysis'),
                            os.path.join('D:' + os.sep, 'Daten'))
    else:
        path = path.replace(os.path.join('D:' + os.sep, 'Daten'),
                            os.path.join('V:' + os.sep, 'BiomedicalImageAnalysis'))

    if sys.platform != 'win32':
        path = path.replace('I:', os.path.join(os.sep, 'work', 'scratch', 'eschweiler'))
        path = path.replace('K:', os.path.join(os.sep, 'home', 'temp', 'eschweiler'))
        path = path.replace('Y:', os.path.join(os.sep, 'home', 'staff', 'eschweiler'))
        path = path.replace('V:', os.path.join(os.sep, 'images'))

    return os.path.normpath(path)


###############################################################################
''' TOOLS '''


###############################################################################

def sphere_intersection_poolhelper(instance_indices, point_coords=None, radii=None):
    # get radiii, positions and distance
    r1 = radii[instance_indices[0]]
    r2 = radii[instance_indices[1]]
    p1 = point_coords[instance_indices[0]]
    p2 = point_coords[instance_indices[1]]
    d = np.sqrt(np.sum((np.array(p1) - np.array(p2)) ** 2))

    # calculate individual volumes
    vol1 = 4 / 3 * np.pi * r1 ** 3
    vol2 = 4 / 3 * np.pi * r2 ** 3

    # calculate intersection of volumes

    # Smaller sphere inside the bigger sphere
    if d <= np.abs(r1 - r2):
        intersect_vol = 4 / 3 * np.pi * np.minimum(r1, r2) ** 3
    # No intersection at all
    elif d > r1 + r2:
        intersect_vol = 0
    # Partially intersecting spheres
    else:
        intersect_vol = np.pi * (r1 + r2 - d) ** 2 * (
                    d ** 2 + 2 * d * r2 - 3 * r2 ** 2 + 2 * d * r1 + 6 * r2 * r1 - 3 * r1 ** 2) / (12 * d)

    return (intersect_vol, vol1, vol2)


def harmonic_non_max_suppression(point_coords, point_probs, shape_descriptors, overlap_thresh=0.5, dim_scale=(1, 1, 1),
                                 num_kernel=1, **kwargs):
    if len(point_coords) > 7500:

        print('Too many points, aborting NMS')
        nms_coords = point_coords[:2000]
        nms_probs = point_probs[:2000]
        nms_shapes = shape_descriptors[:2000]

    elif len(point_coords) > 1:

        dim_scale = [d / np.min(dim_scale) for d in dim_scale]
        point_coords_uniform = []
        for point_coord in point_coords:
            point_coords_uniform.append(tuple([p * d for p, d in zip(point_coord, dim_scale)]))

        # calculate upper and lower volumes
        r_upper = [r.max() for r in shape_descriptors]
        r_lower = [r.min() for r in shape_descriptors]

        # Calculate intersections of lower and upper spheres
        # instance_indices = list(itertools.combinations(range(len(point_coords)), r=2))
        r_max = np.max(r_upper)
        instance_indices = [(i, j) for i in range(len(point_coords))
                            for j in range(i + 1, len(point_coords))
                            if
                            np.sum(np.sqrt(np.abs(np.array(point_coords[i]) - np.array(point_coords[j])))) < r_max * 2]
        with mp.Pool(processes=num_kernel) as p:
            vol_upper = p.map(partial(sphere_intersection_poolhelper, point_coords=point_coords_uniform, radii=r_upper),
                              instance_indices)
            vol_lower = p.map(partial(sphere_intersection_poolhelper, point_coords=point_coords_uniform, radii=r_lower),
                              instance_indices)

        instances_keep = np.ones((len(point_coords),), dtype=np.bool)

        # calculate overlap measure
        for inst_idx, v_up, v_low in zip(instance_indices, vol_upper, vol_lower):

            # average intersection with smaller sphere
            overlap_measure_up = v_up[0] / np.minimum(v_up[1], v_up[2])
            overlap_measure_low = v_low[0] / np.minimum(v_low[1], v_low[2])
            overlap_measure = (overlap_measure_up + overlap_measure_low) / 2

            if overlap_measure > overlap_thresh:
                # Get min and max probable indice
                inst_min = inst_idx[np.argmin([point_probs[i] for i in inst_idx])]
                inst_max = inst_idx[np.argmax([point_probs[i] for i in inst_idx])]

                # If there already was an instance with higher probability, don't add the current "winner"
                if instances_keep[inst_max] == 0:
                    # Mark both as excluded
                    instances_keep[inst_min] = 0
                    instances_keep[inst_max] = 0
                else:
                    # Exclude the loser
                    instances_keep[inst_min] = 0
                    # instances_keep[inst_max] = 1

        # Mark remaining indices for keeping
        # instances_keep = instances_keep != -1

        nms_coords = [point_coords[i] for i, v in enumerate(instances_keep) if v]
        nms_probs = [point_probs[i] for i, v in enumerate(instances_keep) if v]
        nms_shapes = [shape_descriptors[i] for i, v in enumerate(instances_keep) if v]

    else:
        nms_coords = point_coords
        nms_probs = point_probs
        nms_shapes = shape_descriptors

    return nms_coords, nms_probs, nms_shapes


# hierarchical clustering of point detections
def agglomerative_clustering(point_coords, point_probs, shape_descriptors=None, max_dist=6, max_points=20000,
                             dim_scale=(1, 1, 1), use_nms=False, **kwargs):
    if len(point_coords) > max_points:
        warnings.warn('Too many objects detected! Due to memory limitations, the clustering will be aborted.',
                      stacklevel=0)
        cluster_coords = point_coords
        cluster_probs = point_probs
        cluster_shapes = shape_descriptors

    elif len(point_coords) > 1:

        # scale coordinates to a uniform voxel grid
        dim_scale = [d / np.min(dim_scale) for d in dim_scale]
        point_coords_uniform = []
        for point_coord in point_coords:
            point_coords_uniform.append(tuple([p * d for p, d in zip(point_coord, dim_scale)]))

            # calculate linkages
        links = linkage(point_coords_uniform, method='ward', metric='euclidean')

        # define recursive cluster merging
        def recursive_cluster_merging(links, points, clusters_used, cluster_id, num_points):
            clusters_used.append(cluster_id)
            for i in range(2):
                if links[cluster_id, i] < num_points:
                    points.append(int(links[cluster_id, i]))
                else:
                    points, clusters_used = recursive_cluster_merging(links, points, clusters_used,
                                                                      int(links[cluster_id, i] % num_points),
                                                                      num_points)
            return points, clusters_used

        # start from the top of the dendogram and merge clusters until all points
        # are assigned to their corresponding cluster
        clusters = []
        clusters_used = np.zeros((links.shape[0],), dtype='bool')
        for cluster_id in range(links.shape[0] - 1, -1, -1):
            if links[cluster_id, 2] <= max_dist and not clusters_used[cluster_id]:
                points, id_used = recursive_cluster_merging(links, [], [], cluster_id, len(point_coords))
                clusters.append(points)
                clusters_used[id_used] = True

        # sanity check
        clustered_points = [idx for c in clusters for idx in c]
        assert len(clustered_points) == len(np.unique(clustered_points)), \
            'Some points are assigned to multiple clusters'

        # determine centroids based on determined clusters and probabilities
        cluster_coords = []
        cluster_probs = []
        cluster_shapes = []
        for cluster in clusters:
            cluster_prob = [point_probs[c] for c in cluster]
            # determine weighted cluster centroid
            cluster_points = [point_coords[c] for c in cluster]
            if use_nms:
                cluster_centroid = cluster_points[np.argmax(cluster_prob)]
            else:
                cluster_centroid = np.average(cluster_points, axis=0, weights=cluster_prob)
                cluster_centroid = [int(np.round(p)) for p in cluster_centroid]
            # determine cluster descriptor
            if not shape_descriptors is None:
                cluster_descriptors = [shape_descriptors[c] for c in cluster]
                if use_nms:
                    cluster_descriptors = cluster_descriptors[np.argmax(cluster_prob)]
                else:
                    cluster_descriptors = np.average(cluster_descriptors, axis=0, weights=cluster_prob)
                cluster_shapes.append(cluster_descriptors)
            # determine cluster certainity
            if use_nms:
                cluster_prob = cluster_prob[np.argmax(cluster_prob)]
            else:
                cluster_prob = np.average(cluster_prob, axis=0, weights=cluster_prob)
            # add cluster
            cluster_coords.append(tuple(cluster_centroid))
            cluster_probs.append(float(cluster_prob))

        # add points, which were not clustered
        for idx in list(set(range(len(point_coords))) - set(clustered_points)):
            cluster_coords.append(point_coords[idx])
            cluster_probs.append(float(point_probs[idx]))
            if not shape_descriptors is None:
                cluster_shapes.append(shape_descriptors[idx])

    else:
        cluster_coords = point_coords
        cluster_probs = point_probs
        cluster_shapes = shape_descriptors

    if not shape_descriptors is None:
        return cluster_coords, cluster_probs, cluster_shapes
    else:
        return cluster_coords, cluster_probs


# density-based clustering of point detections
def dbscan_clustering(point_coords, point_probs, shape_descriptors=None, max_dist=15, min_count=1, max_points=10000,
                      dim_scale=(1, 1, 1)):
    if len(point_coords) > max_points:
        warnings.warn('Too many objects detected! Due to memory limitations, the clustering will be aborted.',
                      stacklevel=0)
        cluster_coords = point_coords
        cluster_probs = point_probs
        cluster_shapes = shape_descriptors

    elif len(point_coords) > 1:

        # scale coordinates to a uniform voxel grid
        dim_scale = [d / np.min(dim_scale) for d in dim_scale]
        point_coords_uniform = []
        for point_coord in point_coords:
            point_coords_uniform.append(tuple([p * d for p, d in zip(point_coord, dim_scale)]))

        # calculate linkages
        clustering = DBSCAN(eps=max_dist, min_samples=min_count).fit_predict(point_coords_uniform)

        # determine centroids based on determined clusters and probabilities
        cluster_coords = []
        cluster_probs = []
        cluster_shapes = []
        for cluster_idx in np.unique(clustering):

            cluster = np.where(clustering == cluster_idx)[0]

            cluster_prob = [point_probs[c] for c in cluster]
            # determine weighted cluster centroid
            cluster_points = [point_coords[c] for c in cluster]
            cluster_centroid = np.average(cluster_points, axis=0, weights=cluster_prob)
            cluster_centroid = [int(np.round(p)) for p in cluster_centroid]
            # determine cluster descriptor
            if not shape_descriptors is None:
                cluster_descriptors = [shape_descriptors[c] for c in cluster]
                cluster_descriptors = np.average(cluster_descriptors, axis=0, weights=cluster_prob)
                cluster_shapes.append(cluster_descriptors)
            # determine cluster certainity
            cluster_prob = np.average(cluster_prob, axis=0, weights=cluster_prob)
            # add cluster
            cluster_coords.append(tuple(cluster_centroid))
            cluster_probs.append(float(cluster_prob))

    else:
        cluster_coords = point_coords
        cluster_probs = point_probs
        cluster_shapes = shape_descriptors

    if not shape_descriptors is None:
        return cluster_coords, cluster_probs, cluster_shapes
    else:
        return cluster_coords, cluster_probs


def scatter_3d(coords1, coords2, coords3, cartesian=True):
    # (x, y, z) or (r, theta, phi)

    if not cartesian:
        coords1, coords2, coords3 = sphere2cart(coords1, coords2, coords3)

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(coords1, coords2, coords3, depthshade=True)
    plt.show()


def triplot_3d(coords1, coords2, coords3, tri_mesh, cartesian=True):
    # (x, y, z) or (r, theta, phi)

    if not cartesian:
        coords1, coords2, coords3 = sphere2cart(coords1, coords2, coords3)

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.plot_trisurf(coords1, coords2, coords3, triangles=tri_mesh.simplices.copy())
    plt.show()


def get_sampling_sphere(num_sample_points=500, num_iterations=5000, plot_sampling=False):
    # get angular sampling
    theta = np.pi * np.random.rand(num_sample_points // 2)
    phi = 2 * np.pi * np.random.rand(num_sample_points // 2)

    # get initial and updated hemisphere
    hsph_initial = HemiSphere(theta=theta, phi=phi)
    hsph_updated, potential = disperse_charges(hsph_initial, num_iterations)

    # get the full sphere
    sph = Sphere(xyz=np.vstack((hsph_updated.vertices, -hsph_updated.vertices)))

    # plot the resulting sample distribution
    if plot_sampling:
        scatter_3d(sph.x, sph.y, sph.z, cartesian=True)

    return sph


def samples2delaunay(sample_points, cartesian=True):
    # sample points as [r,theta,phi] list

    if not cartesian:
        sample_points = sphere2cart(sample_points[0], sample_points[1], sample_points[2])

    sample_points = np.transpose(np.array(sample_points))
    delaunay_tri = Delaunay(sample_points)

    return delaunay_tri


def delaunay2indices(delaunay_tri, idx):
    # -1 indicates indices outside the region
    voxel_idx = np.nonzero(delaunay_tri.find_simplex(idx) + 1)

    return voxel_idx


def spherical_instance_sampling(data, theta_phi_sampling, bg_values=[0], verbose=False, **kwargs):
    # theta_phi_sampling as [(theta_1,phi_1), (theta_2,phi_2),...] list

    # get all instances
    instances = np.unique(data)
    # exclude background instances
    instances = list(set(instances) - set(bg_values))

    r_sampling = []
    centroids = []

    for num_instance, instance in enumerate(instances):

        if verbose:
            print('\r' * 22 + 'Progress {0:0>2d}% ({1:0>3d}/{2:0>3d})'.format(int(num_instance / len(instances) * 100),
                                                                              num_instance, len(instances)), end='\r')

        # get the mask of the current cell
        instance_mask = data == instance
        # extract the boundary of the current cell
        instance_boundary = np.logical_xor(instance_mask, morphology.binary_erosion(instance_mask))
        # ensure there are no holes created at the image boundary
        instance_boundary[..., 0] = instance_mask[..., 0]
        instance_boundary[..., -1] = instance_mask[..., -1]
        instance_boundary[:, 0, :] = instance_mask[:, 0, :]
        instance_boundary[:, -1, :] = instance_mask[:, -1, :]
        instance_boundary[0, ...] = instance_mask[0, ...]
        instance_boundary[-1, ...] = instance_mask[-1, ...]

        # get coordinates and centroid of the current cell
        mask_coords = np.nonzero(instance_boundary)
        centroid = np.array([np.mean(dim_coords) for dim_coords in mask_coords])
        centroids.append(centroid)
        # set centroid as coordinate origin and get spherical coordinates
        mask_coords = [dim_coords - c for dim_coords, c in zip(mask_coords, centroid)]
        r_mask, theta_mask, phi_mask = cart2sphere(*mask_coords)

        # find closest matches to each sampling point
        distances = distance.cdist(theta_phi_sampling, list(zip(theta_mask, phi_mask)), metric='euclidean',
                                   p=1)  # minkowski??
        closest_matches = np.argmin(distances, axis=1)
        r_sampling.append(r_mask[closest_matches])

    return instances, r_sampling, centroids


class sampling2harmonics():

    def __init__(self, sh_order, theta_phi_sampling, lb_lambda=0.006):
        super(sampling2harmonics, self).__init__()
        self.sh_order = sh_order
        self.theta_phi_sampling = theta_phi_sampling
        self.lb_lambda = lb_lambda
        self.num_samples = len(theta_phi_sampling)
        self.num_coefficients = np.int((self.sh_order + 1) ** 2)

        b = np.zeros((self.num_samples, self.num_coefficients))
        l = np.zeros((self.num_coefficients, self.num_coefficients))

        for num_sample in range(self.num_samples):
            num_coefficient = 0
            for num_order in range(self.sh_order + 1):
                for num_degree in range(-num_order, num_order + 1):

                    theta = theta_phi_sampling[num_sample][0]
                    phi = theta_phi_sampling[num_sample][1]

                    y = sph_harm(np.abs(num_degree), num_order, phi, theta)

                    if num_degree < 0:
                        b[num_sample, num_coefficient] = np.real(y) * np.sqrt(2)
                    elif num_degree == 0:
                        b[num_sample, num_coefficient] = np.real(y)
                    elif num_degree > 0:
                        b[num_sample, num_coefficient] = np.imag(y) * np.sqrt(2)

                    l[num_coefficient, num_coefficient] = self.lb_lambda * num_order ** 2 * (num_order + 1) ** 2
                    num_coefficient += 1

        b_inv = np.linalg.pinv(np.matmul(b.transpose(), b) + l)
        self.convert_mat = np.matmul(b_inv, b.transpose()).transpose()

    def convert(self, r_sampling):
        converted_samples = []
        for r_sample in r_sampling:
            r_converted = np.matmul(r_sample[np.newaxis], self.convert_mat)
            converted_samples.append(np.squeeze(r_converted))
        return converted_samples


class harmonics2sampling():

    def __init__(self, sh_order, theta_phi_sampling):
        super(harmonics2sampling, self).__init__()
        self.sh_order = sh_order
        self.theta_phi_sampling = theta_phi_sampling
        self.num_samples = len(theta_phi_sampling)
        self.num_coefficients = np.int((self.sh_order + 1) ** 2)

        convert_mat = np.zeros((self.num_coefficients, self.num_samples))

        for num_sample in range(self.num_samples):
            num_coefficient = 0
            for num_order in range(self.sh_order + 1):
                for num_degree in range(-num_order, num_order + 1):

                    theta = theta_phi_sampling[num_sample][0]
                    phi = theta_phi_sampling[num_sample][1]

                    y = sph_harm(np.abs(num_degree), num_order, phi, theta)

                    if num_degree < 0:
                        convert_mat[num_coefficient, num_sample] = np.real(y) * np.sqrt(2)
                    elif num_degree == 0:
                        convert_mat[num_coefficient, num_sample] = np.real(y)
                    elif num_degree > 0:
                        convert_mat[num_coefficient, num_sample] = np.imag(y) * np.sqrt(2)

                    num_coefficient += 1

        self.convert_mat = convert_mat

    def convert(self, r_harmonic):
        converted_harmonics = []
        for r_sample in r_harmonic:
            r_converted = np.matmul(r_sample[np.newaxis], self.convert_mat)
            converted_harmonics.append(np.squeeze(r_converted))
        return converted_harmonics


def scale_detections_csv(filelist, x_scale=1, y_scale=1, z_scale=1, **kwargs):
    for filepath in filelist:
        with open(filepath, 'r') as fh:
            data = pd.read_csv(fh, sep=';')

        data['xpos'] = data['xpos'].div(x_scale)
        data['ypos'] = data['ypos'].div(y_scale)
        data['zpos'] = data['zpos'].div(z_scale)

        data = data.astype(int)

        data.to_csv(filepath, sep=';', index_label='id')


###############################################################################
''' METRICS '''


###############################################################################

# calculation of detection accuracy
def detection_accuracy(true_indices, pred_indices, dist_thresh=10, intens_thresh=0.5, calculate_indices=False,
                       return_indices=False, **kwargs):
    if calculate_indices:
        # Get true indices
        true_labeled = measure.label(true_indices)
        true_regions = measure.regionprops(true_labeled)
        true_indices = [r.centroid for r in true_regions]
        # Get predicted indices
        pred_labeled = measure.label(pred_indices > intens_thresh)
        pred_regions = measure.regionprops(pred_labeled)
        pred_indices = [r.centroid for r in pred_regions]

    if len(true_indices) > 0 and len(pred_indices) > 0:

        # set up distance measurements
        dist_mat = distance.cdist(true_indices, pred_indices, metric='euclidean')

        # calculate cost matrix and find cheapest assignments
        cost_mat = dist_mat.copy()
        cost_mat[dist_mat > dist_thresh] = 1e18
        row_ind, col_ind = linear_sum_assignment(cost_mat)

        # create the assignment matrix
        assign_mat = np.zeros_like(dist_mat, dtype=np.uint8)
        assign_mat[row_ind, col_ind] = 1

        # remove assignments above the distance threshold
        assign_mat[dist_mat > dist_thresh] = 0
        row_ind = np.nonzero(np.sum(assign_mat, axis=1))[0]
        col_ind = np.nonzero(np.sum(assign_mat, axis=0))[0]

        # calculate detection accuracy
        tp = np.sum(assign_mat)
        fn = assign_mat.shape[0] - tp
        fp = assign_mat.shape[1] - tp
        det_acc = tp / (tp + fp + fn) if tp != 0 else 0
        det_prec = tp / (tp + fp)
        det_rec = tp / (tp + fn)

        # get indices
        tp_ind = [pred_indices[i] for i in col_ind]
        fp_ind = [pred_indices[i] for i in list(set(range(len(pred_indices))) - set(col_ind))]
        fn_ind = [true_indices[i] for i in list(set(range(len(true_indices))) - set(row_ind))]

    else:
        det_acc = 0
        det_prec = 0
        det_rec = 0
        tp_ind = []
        fp_ind = pred_indices
        fn_ind = true_indices

    if return_indices:
        return det_acc, det_prec, det_rec, tp_ind, fp_ind, fn_ind
    else:
        return det_acc, det_prec, det_rec


def multiclass_f1_score(true_mask, pred_mask, class_threshs=(0.5,), **kwargs):
    # Check if the number of classes match
    assert true_mask.shape[-1] == pred_mask.shape[-1], 'Number of classes do not match!'

    # Get number of classes
    num_classes = true_mask.shape[-1]

    # Set one threshold for each class
    if len(class_threshs) != num_classes:
        class_threshs = (class_threshs[0],) * num_classes

    # Calculate scores
    f1_scores = []
    for num_class, class_thresh in zip(range(num_classes), class_threshs):
        f1_scores.append(
            metrics.f1_score(true_mask[..., num_class].flatten(), pred_mask[..., num_class].flatten() > class_thresh,
                             average=None)[1])

    # Limit to 4 decimals
    f1_scores = np.round(f1_scores, decimals=4)

    return f1_scores


def boundary_score(true_mask, pred_mask, safety_margin=2, thresh=0.5, **kwargs):
    pred_mask = pred_mask > thresh

    # Get safety margins of boundaries
    pred_dilated = morphology.binary_dilation(pred_mask, selem=np.ones((safety_margin,) * pred_mask.ndim))
    true_dilated = morphology.binary_dilation(true_mask, selem=np.ones((safety_margin,) * true_mask.ndim))

    # Calculate scores
    tp = np.sum(np.logical_and(true_dilated.flatten(), pred_mask.flatten()))
    fp = np.sum(np.logical_and(~true_dilated.flatten(), pred_mask.flatten()))
    fn = np.sum(np.logical_and(true_mask.flatten(), ~pred_dilated.flatten()))

    return tp, fp, fn

# calculate recall, precision, BF score
class membrane_score_calculator(object):
    def __init__(self, class_thresh=0.5, image_safety_margin=None):
        self.class_thresh = class_thresh
        self.image_safety_margin = image_safety_margin
        self.TP = 0
        self.TN = 0
        self.FP = 0
        self.FN = 0

    def add_example(self, pred_mask, true_mask):
        # Remove boundary regions for more accurate results
        if not self.image_safety_margin is None:
            pred_mask = pred_mask[self.image_safety_margin:-self.image_safety_margin,
                        self.image_safety_margin:-self.image_safety_margin,
                        self.image_safety_margin:-self.image_safety_margin, :]
            true_mask = true_mask[self.image_safety_margin:-self.image_safety_margin,
                        self.image_safety_margin:-self.image_safety_margin,
                        self.image_safety_margin:-self.image_safety_margin, :]

        # Calculate background Dice
        self.TP += np.count_nonzero(
            np.logical_and(pred_mask[..., 0] >= self.class_thresh, true_mask[..., 0] >= self.class_thresh))
        self.TN += np.count_nonzero(
            np.logical_and(pred_mask[..., 0] < self.class_thresh, true_mask[..., 0] < self.class_thresh))
        self.FP += np.count_nonzero(
            np.logical_and(pred_mask[..., 0] >= self.class_thresh, true_mask[..., 0] < self.class_thresh))
        self.FN += np.count_nonzero(
            np.logical_and(pred_mask[..., 0] < self.class_thresh, true_mask[..., 0] >= self.class_thresh))

    def get_scores(self):
        scores = {}
        scores['TP'] = self.TP
        scores['FN'] = self.FN
        scores['FP'] = self.FP
        scores['Recall'] = self.TP / (self.TP + self.FN)
        scores['Precision'] = self.TP / (self.TP + self.FP)
        scores['BFscore'] = 2 * scores['Precision'] * scores['Recall'] / (scores['Recall'] + scores['Precision'])
        scores['F1score'] = 2 * scores['TP']/ (2 * scores['TP'] + scores['FN'] + scores['FP'])
        return scores


class seg_score_accumulator(object):

    def __init__(self, thresh=None, safety_margin=None):
        self.thresh = np.arange(start=0.5, stop=0.95, step=0.05) if thresh is None else thresh
        self.safety_margin = safety_margin
        self.TP = 0
        self.FN = 0
        self.FP = 0
        self.iouScores = 0
        self.diceScores = 0
        self.eps = 1e-10

    def add_example(self, pred, gt):
        # Remove boundary regions for more accurate results
        if not self.safety_margin is None:
            pred = pred[self.safety_margin:-self.safety_margin, self.safety_margin:-self.safety_margin,
                   self.safety_margin:-self.safety_margin, :]
            gt = gt[self.safety_margin:-self.safety_margin, self.safety_margin:-self.safety_margin,
                 self.safety_margin:-self.safety_margin, :]

        # Set up lists
        IoU_list = []
        dice_list = []

        # Determine areas and unique gt labels
        pred_area = self.get_area_dict(pred)
        gt_area = self.get_area_dict(gt)
        unique_gt = np.unique(gt)
        false_count_gt = 0

        for label in unique_gt:

            # Skip background label
            if label == 0:
                false_count_gt += 1
                continue

            # Get the labels and pixel counts for the predicted regions overlapping the gt region
            u, c = np.unique(pred[gt == label], return_counts=True)
            ind = np.argsort(c, kind='mergesort')
            # take the gt label with the largest overlap
            if len(u) == 1 and u[ind[-1]] == 0:
                # only background contained
                IoU_list.append(0)
                dice_list.append(0)
            else:
                # take the gt label with the largest overlap
                i = ind[-2] if u[ind[-1]] == 0 else ind[-1]
                intersect = c[i]
                IoU_list.append(intersect / (gt_area[label] + pred_area[u[i]] - intersect))
                dice_list.append(2*intersect / (gt_area[label] + pred_area[u[i]]))
        IoU_list = np.array(IoU_list)
        dice_list = np.array(dice_list)

        tp = np.sum(IoU_list >= self.thresh)
        self.TP += tp
        self.FP += len(IoU_list) - tp
        self.FN += len(unique_gt) - false_count_gt - tp
        self.diceScores += np.sum(dice_list[IoU_list >= self.thresh])

    def get_area_dict(self, label_map):
        props = measure.regionprops(np.squeeze(label_map).astype(np.int))
        return {p.label: p.area for p in props}

    def get_scores(self):
        scores = {}
        scores['iou_threshs'] = self.thresh
        scores['precisions'] = (self.TP + self.eps) / (self.TP + self.FN + self.FP + self.eps)
        #scores['avg_precision'] = np.nanmean(scores['precisions'])
        scores['dice_scores'] = self.diceScores / self.TP
        #scores['avg_dice'] = np.nanmean(scores['dice_scores'])

        return scores


###############################################################################
''' KARLSRUHE METRICS '''


###############################################################################


def intersection_hist_col(result_list):
    """ Calculation of all intersections of a predicted nucleus with the ground truth nuclei and background. This
    results in a column of the intersection histogram needed to calculate some metrics.

    :param result_list: List containing the prediction, the ground_truth, a list of the ids of the predicted nuclei and
        a list of the ids of the ground truth nuclei (no explicit color channels for prediction and ground truth). The
        maximum label in the prediction and the ground truth is the maximum in these images.
        :type result_list: list
    :return: List of intersection histogram columns and the corresponding prediction nucleus ids.
    """

    # Unpack result list
    prediction, ground_truth, nucleus_ids_prediction, nucleus_ids_ground_truth = result_list

    intersection_hist_cols = []

    # Calculate for each predicted nucleus and the background the intersections with the ground truth nuclei and
    # background
    for nucleus_id_prediction in nucleus_ids_prediction:
        # Select predicted nucleus
        nucleus_prediction = (prediction == nucleus_id_prediction)

        # Intensity-coded intersections
        intersections = nucleus_prediction * ground_truth

        # Sum intersection with every ground truth nucleus
        hist = np.histogram(intersections,
                            bins=range(1, nucleus_ids_ground_truth[-1] + 2),
                            range=(1, nucleus_ids_ground_truth[-1] + 1))
        hist = hist[0]
        # Move background to front
        hist = hist[[len(hist) - 1] + list(range(len(hist) - 1))]
        intersection_hist_cols.append([nucleus_id_prediction, hist.astype(np.uint64)])

    return intersection_hist_cols


def aggregated_iou_score(result_list):
    """ Best intersection, the corresponding unions and the best intersection over union score for each ground truth
    nucleus. It is assumed that the biggest intersection corresponds to the best intersection over union score. There
    may be cases in which this does not hold. However, in that cases, the IoU is below 0.5 and it is insignificant for
    the piou metric.

    :param result_list:
    :return:
    """

    pred, gt, intersections, hist = result_list

    aggregated_intersection, aggregated_union, used_nuclei_pred, iou = 0, 0, [], []

    for i in hist:  # start from 1 to exclude the background matches

        if i != 0:
            best_intersection_nucleus = np.argmax(intersections[i, 1:]) + 1
            best_intersection = intersections[i, best_intersection_nucleus]
            aggregated_intersection += best_intersection
            union = np.sum((gt == i) | (pred == best_intersection_nucleus))
            aggregated_union += np.sum((gt == i) | (pred == best_intersection_nucleus))
            used_nuclei_pred.append(best_intersection_nucleus)
            iou.append(best_intersection / union)

    return [aggregated_intersection, aggregated_union, used_nuclei_pred, iou]


def metric_collection(prediction, ground_truth, num_threads=8):
    """ Calculation of Rand-Index, Jaccard-Index, mean average precision at different intersection over union
    thresholds (P_IoU), precision, recall, F-score and split/merged/missing/spurious objects.

    :param prediction: Prediction with intensity coded nuclei.
        :type prediction:
    :param ground_truth: Ground truth image with intensity coded nuclei.
        :type ground_truth:
    :param num_threads: Number of threads to speeden up the computation.
        :type num_threads: int
    :return: Dictionary containing the metric scores.
    """

    # Create copy of the prediction and ground truth to avoid changing them
    pred, gt = np.copy(prediction), np.copy(ground_truth)

    # Find intensity coded nuclei in the ground truth image and the prediction (simply looking for the maximum is not
    # possible because in the post-processing numbered seeds can vanish, additionally for tracking data some nuclei
    # may not appear at that time point)
    nucleus_ids_ground_truth = np.unique(gt)
    nucleus_ids_prediction = np.unique(pred)

    # Number of cell nuclei in the ground truth image and in the prediction
    num_nuclei_ground_truth, num_nuclei_prediction = len(nucleus_ids_ground_truth), len(nucleus_ids_prediction)

    # Check for empty predictions
    if num_nuclei_prediction == 0:
        return {'Rand_index': 0, 'Jaccard_index': 0, 'Aggregated_Jaccard_index': 0, 'P_IoU': 0, 'Precision': 0,
                'Recall': 0, 'F-Score': 0, 'Split': 0, 'Merged': 0, 'Missing': num_nuclei_ground_truth, 'Spurious': 0
                }, 0

    # Check for missing nuclei ids in the prediction. To build the intersection histogram the nuclei_ids should range
    # from 1 to the number of nuclei.
    if num_nuclei_prediction != pred.max():

        hist = np.histogram(pred, bins=range(1, pred.max() + 2), range=(1, pred.max() + 1))

        # Find missing values
        missing_values = np.where(hist[0] == 0)[0]

        # Decrease the ids of the nucleus with higher id than the missing. Reverse the list to avoid problems in case
        # of multiple missing objects
        for th in reversed(missing_values):
            pred[pred > th] = pred[pred > th] - 1

    # Check for missing nuclei ids in the ground truth. To build the intersection histogram the nuclei_ids should range
    # from 1 to the number of nuclei.
    if num_nuclei_ground_truth != gt.max():

        hist = np.histogram(gt, bins=range(1, gt.max() + 2), range=(1, gt.max() + 1))

        # Find missing values
        missing_values = np.where(hist[0] == 0)[0]

        # Decrease the ids of the nucleus with higher id than the missing. Reverse the list to avoid problems in case
        # of multiple missing objects
        for th in reversed(missing_values):
            gt[gt > th] = gt[gt > th] - 1

    # Change the background label from 0 to num_nuclei + 1. This enables to calculate the intersection with the
    # background efficiently.
    bg_gt, bg_pred = num_nuclei_ground_truth + 1, num_nuclei_prediction + 1
    pred[pred == 0] = bg_pred
    gt[gt == 0] = bg_gt
    nucleus_ids_ground_truth = np.unique(gt)
    nucleus_ids_prediction = np.unique(pred)

    # Preallocate arrays for the intersection histogram
    intersections = np.zeros(shape=(num_nuclei_ground_truth + 1, num_nuclei_prediction + 1), dtype=np.uint64)

    # Create list to calculate the histogram entries in parallel
    result_list = []

    if (num_nuclei_prediction + 1) > num_threads:

        fraction = (num_nuclei_prediction + 1) / num_threads  # + 1 because the background is added

        for i in range(num_threads):
            result_list.append([pred,
                                gt,
                                nucleus_ids_prediction[int(i * fraction):int((i + 1) * fraction)],
                                nucleus_ids_ground_truth])
    else:

        result_list.append([pred, gt, nucleus_ids_prediction, nucleus_ids_ground_truth])

    # Calculate the intersection histogram entries in parallel
    pool = mp.Pool(num_threads)
    intersection_hist_entries = pool.map(intersection_hist_col, result_list)
    pool.close()

    # Pack the intersection histogram column lists into a single list
    for i in range(len(intersection_hist_entries)):
        for j in range(len(intersection_hist_entries[i])):
            col = intersection_hist_entries[i][j][0]
            if col == bg_pred:  # Move background column to the first
                col = 0
            intersections[:, col] = intersection_hist_entries[i][j][1]

    # Calculate Rand index and Jaccard index
    a, b, c, n = 0, 0, 0, len(prediction.flatten())

    for i in range(intersections.shape[0]):
        row_sum = np.sum(intersections[i, :], dtype=np.uint64)
        b += row_sum * (row_sum - 1) / 2
        for j in range(intersections.shape[1]):
            if i == 0:
                col_sum = np.sum(intersections[:, j], dtype=np.uint64)
                c += col_sum * (col_sum - 1) / 2
            a += intersections[i, j].astype(np.float64) * (intersections[i, j].astype(np.float64) - 1) / 2
    b -= a
    c -= a
    d = n * (n - 1) / 2 - a - b - c
    rand, jaccard = (a + d) / (a + b + c + d), (a + d) / (b + c + d)

    # Match objects with maximum intersections to detect split, merged, missing and spurious objects
    gt_matches, pred_matches, merged, missing, split, spurious = [], [], 0, 0, 0, 0
    for i in range(intersections.shape[0]):
        gt_matches.append(np.argmax(intersections[i, :]))
    for j in range(intersections.shape[1]):
        pred_matches.append(np.argmax(intersections[:, j]))
    gt_matches_counts, pred_matches_counts = Counter(gt_matches), Counter(pred_matches)
    for nucleus in gt_matches_counts:
        if nucleus == 0 and gt_matches_counts[nucleus] > 1:
            missing = gt_matches_counts[nucleus] - 1
        elif nucleus != 0 and gt_matches_counts[nucleus] > 1:
            merged += gt_matches_counts[nucleus] - 1
    for nucleus in pred_matches_counts:
        if nucleus == 0 and pred_matches_counts[nucleus] > 1:
            spurious = pred_matches_counts[nucleus] - 1
        elif nucleus != 0 and pred_matches_counts[nucleus] > 1:
            split += pred_matches_counts[nucleus] - 1

    # Aggregated Jaccard index and P_IoU (for the best IoU it does not matter if the predictions are matched to ground
    # truth nuclei or the other way around since the lowest threshold used later is 0.5, for the Jaccard-index it does).
    result_list = []  # Create list to find the best intersections and the corresponding unions in parallel

    if len(gt_matches) > num_threads:

        fraction = len(gt_matches) / num_threads

        for i in range(num_threads):
            result_list.append([pred, gt, intersections, list(range(int(i * fraction), int((i + 1) * fraction)))])

    else:
        result_list.append([pred, gt, intersections, list(range(1, len(gt_matches)))])

    pool = mp.Pool(num_threads)
    best_intersections_unions = pool.map(aggregated_iou_score, result_list)
    pool.close()

    aggregated_intersection, aggregated_union, used_nuclei_pred, iou = 0, 0, [], []
    for i in range(len(best_intersections_unions)):
        aggregated_intersection += best_intersections_unions[i][0]
        aggregated_union += best_intersections_unions[i][1]
        used_nuclei_pred = used_nuclei_pred + best_intersections_unions[i][2]
        iou = iou + best_intersections_unions[i][3]

    for nucleus in nucleus_ids_prediction[:-1]:  # Exclude background
        if nucleus not in used_nuclei_pred:
            aggregated_union += np.sum(pred == nucleus)

    aggregated_jaccard_index = aggregated_intersection / aggregated_union

    # Preallocate arrays for true positives, false negatives and true positives for each IoU threshold
    tp = np.zeros(shape=(10,), dtype=np.uint16)
    fp = np.zeros(shape=(10,), dtype=np.uint16)
    fn = np.zeros(shape=(10,), dtype=np.uint16)

    # Count true positives, false positives and false negatives for different IoU-thresholds th
    for i, th in enumerate(np.arange(0.5, 1.0, 0.05)):
        matches = iou > th

        # True positive: IoU > threshold
        tp[i] = np.count_nonzero(matches)

        # False negative: ground truth object has no associated predicted object
        fn[i] = num_nuclei_ground_truth - tp[i]

        # False positive: predicted object has no associated ground truth object
        fp[i] = num_nuclei_prediction - tp[i]

    # Precision for each threshold
    prec = np.divide(tp, (tp + fp + fn))

    # Precision for IoU-threshold = 0.5
    precision = np.divide(tp[0], tp[0] + fp[0])

    # Recall for IoU-threshold = 0.5
    recall = np.divide(tp[0], tp[0] + fn[0])

    # F-Score for IoU-threshold = 0.5
    f_score = 2 * np.divide(recall * precision, recall + precision)

    # Mean precision (10 thresholds)
    piou = np.mean(prec)

    # Result dictionary
    results = {'Rand_index': rand,
               'Jaccard_index': jaccard,
               'Aggregated_Jaccard_index': aggregated_jaccard_index,
               'P_IoU': piou,
               'Precision': precision,
               'Recall': recall,
               'F-Score': f_score,
               'Split': split,
               'Merged': merged,
               'Missing': missing,
               'Spurious': spurious
               }

    return results, intersections


