import cv2
import numpy as np
import random
import os, sys 
import torch
import torch.nn.functional as F
from tqdm import tqdm
import glob
import re
import math
from datetime import datetime
import pytz
from torch.utils.data import Dataset
from PIL import Image
import warnings

submodule_path = ( "/share/phoenix/nfs05/S8/gc492/scene_gen/Depth-Anything" )
assert os.path.exists(submodule_path)
sys.path.insert(0, submodule_path)
import depth_anything.dpt
from depth_anything.dpt import DepthAnything
from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet
from torchvision.transforms import Compose


def create_sparse_depth_map(points_3d, intrinsic_matrix, extrinsic_matrix, image_shape):
    # points_3d shape: N,3
    # Convert points to homogeneous coordinates
    points_homogeneous = np.hstack((points_3d, np.ones((points_3d.shape[0], 1)))).T
    
    # Apply the extrinsic transformation (world to camera coordinates)
    camera_coords = extrinsic_matrix @ points_homogeneous

    # Convert to non-homogeneous camera coordinates
    camera_coords_non_homogeneous = camera_coords[:3, :]

    # Project the points onto the 2D image plane (camera to image coordinates)
    projected_points = intrinsic_matrix @ camera_coords_non_homogeneous

    # Normalize the coordinates and round them to get pixel indices
    x_pixels = np.round(projected_points[0, :] / projected_points[2, :]).astype(int)
    y_pixels = np.round(projected_points[1, :] / projected_points[2, :]).astype(int)
    depths = projected_points[2, :]

    # Initialize the depth map with infinity (or a very large number)
    depth_map = np.full(image_shape, np.inf)

    # Update the depth map with the nearest depth at each pixel
    for x, y, depth in zip(x_pixels, y_pixels, depths):
        if 0 <= x < image_shape[1] and 0 <= y < image_shape[0]:
            depth_map[y, x] = min(depth_map[y, x], depth)

    # Replace infinities with zeros or an appropriate background value
    # later zeros are replaced by mask
    depth_map[depth_map == np.inf] = 0

    return depth_map

# crashes for some reason, debug later
# def create_sparse_depth_map_new(points_3d, intrinsic_matrix, extrinsic_matrix, image_shape, inverse=False):
#     # Convert points to homogeneous coordinates
#     points_homogeneous = np.hstack((points_3d, np.ones((points_3d.shape[0], 1)))).T
    
#     # Apply the extrinsic transformation (world to camera coordinates)
#     camera_coords = extrinsic_matrix @ points_homogeneous

#     # Convert to non-homogeneous camera coordinates
#     camera_coords_non_homogeneous = camera_coords[:3, :]

#     # Project the points onto the 2D image plane (camera to image coordinates)
#     projected_points = intrinsic_matrix @ camera_coords_non_homogeneous

#     # Normalize the coordinates and round them to get pixel indices
#     x_pixels = np.round(projected_points[0, :] / projected_points[2, :]).astype(int)
#     y_pixels = np.round(projected_points[1, :] / projected_points[2, :]).astype(int)
#     depths = projected_points[2, :]

#     # Create a full 3D grid of coordinates and depths
#     xx, yy, dd = np.meshgrid(np.arange(image_shape[1]), np.arange(image_shape[0]), depths, indexing='xy')
#     xx, yy = xx - x_pixels, yy - y_pixels
    
#     # Create a mask for the minimum depths
#     mask = (xx == 0) & (yy == 0)
#     min_depths = np.where(mask, dd, np.inf).min(axis=2)
    
#     # Initialize the depth map and update it
#     depth_map = np.full(image_shape, np.inf)
#     depth_map = np.minimum(depth_map, min_depths)
    
#     # Replace infinities with zeros or an appropriate background value
#     depth_map[depth_map == np.inf] = 0

#     # zeros for masking but could also convert to inverse depth
#     if inverse:
#         depth_map[depth_map != 0] = 1.0/depth_map[depth_map != 0]

#     return depth_map




# load depth model: https://pytorch.org/hub/intelisl_midas_v2/
# def load_depth_model(depth_model_name='midas'):
#     if depth_model_name == "midas":
#         depth_repo = "intel-isl/MiDaS"
#         #depth_repo = "/home/jupyter/.cache/torch/hub/isl-org_MiDaS_master"
#         depth_arch = "DPT_Large" #"DPT_Hybrid" #"DPT_SwinV2_T_256"
#         depth_model = torch.hub.load( depth_repo, depth_arch )  #source="local"
#         depth_model = depth_model.eval().cuda()
#         return depth_model

def load_depth_model():
    # load depth model
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        encoder = 'vitl' # can also be 'vitb' or 'vitl'
        depth_model = DepthAnything.from_pretrained('LiheYoung/depth_anything_{:}14'.format(encoder)).cuda().eval()
        dtransform = Compose([
                Resize(
                    width=518,
                    height=518,
                    resize_target=False,
                    keep_aspect_ratio=True,
                    ensure_multiple_of=14,
                    resize_method='lower_bound',
                    image_interpolation_method=cv2.INTER_CUBIC,
                ),
                NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                PrepareForNet(),
            ])
        return depth_model, dtransform


def invert_depth(depth_map):
    inv = depth_map.clone()
    # disparity_max = 1000
    disparity_min = 0.001
    # inv[inv > disparity_max] = disparity_max
    inv[inv < disparity_min] = disparity_min
    inv = 1.0 / inv
    return inv

    
def ransac_pc_depth_alignment(estimated_dense_disparity, colmap_depth, ransac_iters=100, device='cuda'):
    if device == 'cpu':
        return cpu_ransac_alignment(estimated_dense_disparity, colmap_depth, ransac_iters=ransac_iters)
    else:
        return gpu_ransac_alignment(estimated_dense_disparity, colmap_depth, ransac_iters=ransac_iters, device='cuda')


def cpu_ransac_alignment(estimated_dense_disparity, colmap_depth, ransac_iters=100):
    disparity_max = 10000
    disparity_min = 0.0001
    depth_max = 1 / disparity_min
    depth_min = 1 / disparity_max

    #print(colmap_depth.shape, estimated_dense_disparity.shape)
    assert colmap_depth.shape == estimated_dense_disparity.shape, ( colmap_depth.shape, estimated_dense_disparity.shape )

    colmap_depth = colmap_depth.float()
    estimated_dense_disparity = estimated_dense_disparity.float()

    mask = colmap_depth != 0

    target = colmap_depth
    target = torch.clip(target, depth_min, depth_max)

    prediction = estimated_dense_disparity

    # zeros where colmap_depth is zero, 1/depth for disparity where colmap_depth is not zero
    # target[mask == 1] equal to colmap_depth[mask == 1]
    target_disparity = torch.zeros_like(target)
    target_disparity[mask == 1] = 1.0 / target[mask == 1] # convert to inverse depth

    nonzero_indices = torch.nonzero(mask[0], as_tuple=False)
    num_to_zero_out = len(nonzero_indices)-2  #int(percent * len(nonzero_indices))
    
    min_error = 1.0
    best_aligned = 0.0

    for _ in range(ransac_iters):
        mask1 = mask.clone()
        target_disparity1 = target_disparity.clone()
        rand_indices = torch.randperm(len(nonzero_indices))[:num_to_zero_out]
        coords = nonzero_indices[rand_indices].long()
        rows, cols = coords[:, 0], coords[:, 1]
        mask1[0, rows, cols] = 0
        target_disparity1[0, rows, cols] = 0.0
    
        scale, shift = compute_scale_and_shift(prediction, target_disparity1, mask1)
            
        prediction_aligned = scale.view(-1, 1, 1) * prediction + shift.view(-1, 1, 1)
        prediction_aligned[prediction_aligned > disparity_max] = disparity_max
        prediction_aligned[prediction_aligned < disparity_min] = disparity_min
        prediction_depth = (1.0 / prediction_aligned).float()

        # calculate errors
        threshold=1.05
        # bad pixel
        err = torch.zeros_like(prediction_depth, dtype=torch.float)
        
        # target is ground truth sparse depth map, 0 where no points and 1 where there is a 3d reconstruction point
        # prediction_depth / target or target / prediction_depth is the error...should ideally be 1 (i.e. prediction after alignment == ground truth)
        err[mask == 1] = torch.max(
            prediction_depth[mask == 1] / target[mask == 1],
            target[mask == 1] / prediction_depth[mask == 1],
        )
        err[mask == 1] = (err[mask == 1] > threshold).float()
        
        # err is 0 where no points or prediction == ground truth
        # err is 1 where prediction != ground truth
        # mask is 1 where there is a point
        p = torch.sum(err, (1, 2)) / torch.sum(mask, (1, 2))

        if p.squeeze().item() < min_error:
            min_error = p.squeeze().item()
            best_aligned = prediction_depth

    return best_aligned, min_error

def gpu_ransac_alignment(estimated_dense_disparity, colmap_depth, ransac_iters=100, mask_percent=0.9, device='cuda'):
    disparity_max = 10000
    disparity_min = 0.0001

    depth_max = 1 / disparity_min
    depth_min = 1 / disparity_max

    #print(colmap_depth.shape, estimated_dense_disparity.shape)
    assert colmap_depth.shape == estimated_dense_disparity.shape, ( colmap_depth.shape, estimated_dense_disparity.shape )

    mask = colmap_depth != 0

    target = colmap_depth
    target = torch.clip(target, depth_min, depth_max)

    prediction = estimated_dense_disparity

    # zeros where colmap_depth is zero, 1/depth for disparity where colmap_depth is not zero
    # target[mask == 1] equal to colmap_depth[mask == 1]
    target_disparity = torch.zeros_like(target)
    target_disparity[mask == 1] = 1.0 / target[mask == 1] # convert to inverse depth

    # ransac
    #ipdb.set_trace()
    prediction1 = prediction.repeat(ransac_iters,1,1).to(device)
    target_disparity1 = target_disparity.repeat(ransac_iters,1,1).to(device)
    mask1 = mask.repeat(ransac_iters,1,1).to(device)
    target_depth = target.repeat(ransac_iters,1,1).to(device)
    original_mask = mask.repeat(ransac_iters,1,1).to(device)

    percent = mask_percent # how much to mask out --> changing to only using 2 points
    nonzero_indices = torch.nonzero(mask1[0], as_tuple=False)
    num_to_zero_out = len(nonzero_indices)-2  #int(percent * len(nonzero_indices))

    all_indices = torch.stack([torch.randperm(len(nonzero_indices)) for _ in range(ransac_iters)])
    rand_indices = all_indices[:, :num_to_zero_out]

    range_tensor = torch.arange(ransac_iters).view(-1, 1)
    nonzeros = nonzero_indices.unsqueeze(0).repeat(ransac_iters,1,1)
    masked_indices = nonzeros[range_tensor, rand_indices].long()

    # Extract batch indices (which is just a range from 0 to B-1)
    batch_indices = torch.arange(mask1.shape[0])[:, None]

    # Use advanced indexing to set the specified pixels to 0
    mask1[batch_indices, masked_indices[..., 0], masked_indices[..., 1]] = 0
    target_disparity1[batch_indices, masked_indices[..., 0], masked_indices[..., 1]] = 0.0

    scale, shift = compute_scale_and_shift(prediction1, target_disparity1, mask1)
        
    prediction_aligned = scale.view(-1, 1, 1) * prediction1 + shift.view(-1, 1, 1)

    prediction_aligned[prediction_aligned > disparity_max] = disparity_max
    prediction_aligned[prediction_aligned < disparity_min] = disparity_min
    prediction_depth = 1.0 / prediction_aligned

    # calculate errors
    threshold=1.05
    prediction_depth = prediction_depth.float()
    original_mask = original_mask.float()
    target_depth = target_depth.float()
    # bad pixel
    err = torch.zeros_like(prediction_depth, dtype=torch.float)

    # target is ground truth sparse depth map, 0 where no points and 1 where there is a 3d reconstruction point
    # prediction_depth / target or target / prediction_depth is the error...should ideally be 1 (i.e. prediction after alignment == ground truth)
    err[original_mask == 1] = torch.max(
        prediction_depth[original_mask == 1] / target_depth[original_mask == 1],
        target_depth[original_mask == 1] / prediction_depth[original_mask == 1],
    )
    err[original_mask == 1] = (err[original_mask == 1] > threshold).float()

    # err is 0 where no points or prediction == ground truth
    # err is 1 where prediction != ground truth
    # mask is 1 where there is a point
    p = torch.sum(err, (1, 2)) / torch.sum(original_mask, (1, 2))
    min_value, min_index = torch.min(p, 0)

    return prediction_depth[min_index].unsqueeze(0).cpu(), min_value.cpu().item() # aligned depth, error



def get_aligned_monodepths(estimated_dense_disparity, colmap_depth, return_error=False):

    disparity_max = 10000
    disparity_min = 0.0001

    depth_max = 1 / disparity_min
    depth_min = 1 / disparity_max

    #print(colmap_depth.shape, estimated_dense_disparity.shape)
    assert colmap_depth.shape == estimated_dense_disparity.shape, ( colmap_depth.shape, estimated_dense_disparity.shape )

    mask = colmap_depth != 0

    target = colmap_depth
    target = torch.clip(target, depth_min, depth_max)

    prediction = estimated_dense_disparity

    # zeros where colmap_depth is zero, 1/depth for disparity where colmap_depth is not zero
    # target[mask == 1] equal to colmap_depth[mask == 1]
    target_disparity = torch.zeros_like(target)
    target_disparity[mask == 1] = 1.0 / target[mask == 1] # convert to inverse depth

    scale, shift = compute_scale_and_shift(prediction, target_disparity, mask)

    prediction_aligned = scale.view(-1, 1, 1) * prediction + shift.view(-1, 1, 1)

    #print(scale,shift)
    #printinfo(prediction, prediction_aligned)

    prediction_aligned[prediction_aligned > disparity_max] = disparity_max
    prediction_aligned[prediction_aligned < disparity_min] = disparity_min
    prediction_depth = 1.0 / prediction_aligned

    if return_error:
        threshold=1.05
        prediction_depth = prediction_depth.float()
        mask = mask.float()
        target = target.float()
        # bad pixel
        err = torch.zeros_like(prediction_depth, dtype=torch.float)
        
        # target is ground truth sparse depth map, 0 where no points and 1 where there is a 3d reconstruction point
        # prediction_depth / target or target / prediction_depth is the error...should ideally be 1 (i.e. prediction after alignment == ground truth)
        err[mask == 1] = torch.max(
            prediction_depth[mask == 1] / target[mask == 1],
            target[mask == 1] / prediction_depth[mask == 1],
        )
        err[mask == 1] = (err[mask == 1] > threshold).float()
        
        # err is 0 where no points or prediction == ground truth
        # err is 1 where prediction != ground truth
        # mask is 1 where there is a point
        p = torch.sum(err, (1, 2)) / torch.sum(mask, (1, 2))
        #print("error: ", 100 * torch.mean(p)) # % of points that have error compared to ground truth sparse depth map
        return prediction_depth, p.item()
    
    return prediction_depth


def compute_scale_and_shift(prediction, target, mask):
    '''
    https://gist.github.com/ranftlr/45f4c7ddeb1bbb88d606bc600cab6c8d
    '''
    # system matrix: A = [[a_00, a_01], [a_10, a_11]]
    a_00 = torch.sum(mask * prediction * prediction, (1, 2))
    a_01 = torch.sum(mask * prediction, (1, 2))
    a_11 = torch.sum(mask, (1, 2)) # shape of B, where each b is the sum of every element at index b

    # right hand side: b = [b_0, b_1]
    b_0 = torch.sum(mask * prediction * target, (1, 2))
    b_1 = torch.sum(mask * target, (1, 2))

    # solution: x = A^-1 . b = [[a_11, -a_01], [-a_10, a_00]] / (a_00 * a_11 - a_01 * a_10) . b
    x_0 = torch.zeros_like(b_0)
    x_1 = torch.zeros_like(b_1)

    det = a_00 * a_11 - a_01 * a_01
    # A needs to be a positive definite matrix.
    valid = det > 0

    x_0[valid] = (a_11[valid] * b_0[valid] - a_01[valid] * b_1[valid]) / det[valid]
    x_1[valid] = (-a_01[valid] * b_0[valid] + a_00[valid] * b_1[valid]) / det[valid]

    return x_0, x_1


# fx,fy should be of the target image!
def warp_image(original_image, target_image, depth_map, fx1, fy1,  fx2, fy2, original_pose, new_pose):
    # assuming depth_map is aligned with the original_image and in the same scale.
    
    # initialize the warped image with black pixels
    warped_image = np.zeros_like(target_image)

    # intrinsic camera parameters
    cxref, cyref = original_image.shape[1] / 2, original_image.shape[0] / 2
    cxtar, cytar = target_image.shape[1] / 2, target_image.shape[0] / 2

    # create a meshgrid of pixel coordinates
    height, width = original_image.shape[:2]
    heighttar, widthtar = target_image.shape[:2]
    x = np.linspace(0, width - 1, width)
    y = np.linspace(0, height - 1, height)
    xv, yv = np.meshgrid(x, y)

    # depth map to 3D point cloud in original camera coordinates
    z = depth_map
    x3d = (xv - cxref) * z / fx1
    y3d = (yv - cyref) * z / fy1

    # original camera pose to world coordinates
    points_3d = np.dstack((x3d, y3d, z))
    points_3d_homogeneous = np.hstack((points_3d.reshape(-1, 3), np.ones((height * width, 1))))
    points_world = np.linalg.inv(original_pose) @ points_3d_homogeneous.T

    # 4,N; 4,4; 4,N; N,4
    #print("shapes: ", points_world.shape, original_pose.shape, points_3d_homogeneous.T.shape, points_3d_homogeneous.shape)

    # world coordinates to new camera coordinates
    points_new_camera = new_pose @ points_world

    # Project points back onto the new image plane
    x_new = fx2 * points_new_camera[0, :] / (points_new_camera[2, :]+1e-10) + cxtar
    y_new = fy2 * points_new_camera[1, :] / (points_new_camera[2, :]+1e-10) + cytar

    valid_mask = (~np.isnan(x_new)) & (~np.isnan(y_new))
    x_new_int = x_new.astype(int)
    y_new_int = y_new.astype(int)
    
    # refine mask for coordinates within image boundaries
    in_bounds_mask = (0 <= x_new_int) & (x_new_int < width) & (0 <= y_new_int) & (y_new_int < height)
    in_bounds_mask2 = (0 <= x_new_int) & (x_new_int < widthtar) & (0 <= y_new_int) & (y_new_int < heighttar)
    total_mask = valid_mask & in_bounds_mask & in_bounds_mask2
    
    # apply mask and copy pixels
    xv_flat = xv.flatten()
    yv_flat = yv.flatten()
    
    warped_image[y_new_int[total_mask], x_new_int[total_mask]] = original_image[yv_flat[total_mask].astype(int), xv_flat[total_mask].astype(int)]

    return warped_image

def get_warp_9d_motion(original_image, depth_map, fx, fy, original_pose, new_pose):
    # Assuming depth_map is aligned with the original_image and in the same scale.
    
    # Initialize the warped image with black pixels
    warped_image = np.zeros((original_image.shape[0], original_image.shape[1], 9)) # RGB + 6d coords

    # Intrinsic camera parameters
    cx, cy = original_image.shape[1] / 2, original_image.shape[0] / 2

    # Create a meshgrid of pixel coordinates
    height, width = original_image.shape[:2]
    x = np.linspace(0, width - 1, width)
    y = np.linspace(0, height - 1, height)
    xv, yv = np.meshgrid(x, y)

    # Convert depth map to 3D point cloud in original camera coordinates
    z = depth_map
    x3d = (xv - cx) * z / fx
    y3d = (yv - cy) * z / fy

    # Convert 3D points from original camera pose to world coordinates
    points_3d = np.dstack((x3d, y3d, z))
    points_3d_homogeneous = np.hstack((points_3d.reshape(-1, 3), np.ones((height * width, 1))))
    points_world = np.linalg.inv(original_pose) @ points_3d_homogeneous.T

    # 4,N; 4,4; 4,N; N,4
    #print("shapes: ", points_world.shape, original_pose.shape, points_3d_homogeneous.T.shape, points_3d_homogeneous.shape)

    # Convert points from world coordinates to new camera coordinates
    points_new_camera = new_pose @ points_world

    # Project points back onto the new image plane
    x_new = fx * points_new_camera[0, :] / points_new_camera[2, :] + cx
    y_new = fy * points_new_camera[1, :] / points_new_camera[2, :] + cy

    cam2x, cam2y, cam2z = points_new_camera[0, :], points_new_camera[1, :], points_new_camera[2, :]

    valid_mask = (~np.isnan(x_new)) & (~np.isnan(y_new))
    x_new_int = x_new.astype(int)
    y_new_int = y_new.astype(int)

    # Further refine mask for coordinates within image boundaries
    in_bounds_mask = (0 <= x_new_int) & (x_new_int < width) & (0 <= y_new_int) & (y_new_int < height)
    total_mask = valid_mask & in_bounds_mask

    # Apply mask and copy pixels
    xv_flat = xv.flatten()
    yv_flat = yv.flatten()

    # copy over the RGB and the 6d coords (x3d, y3d, z; cam2x, cam2y, cam2z)
    warped_image[y_new_int[total_mask], x_new_int[total_mask], :3] = original_image[yv_flat[total_mask].astype(int), xv_flat[total_mask].astype(int)]
    warped_image[y_new_int[total_mask], x_new_int[total_mask], 3:6] = points_3d[yv_flat[total_mask].astype(int), xv_flat[total_mask].astype(int)]
    warped_image[y_new_int[total_mask], x_new_int[total_mask], 6:] = np.vstack((cam2x[total_mask], cam2y[total_mask], cam2z[total_mask])).T  # equivalent to: points_new_camera.transpose(1,0)[:,:3].reshape(192,256,3)[yv_flat[total_mask].astype(int), xv_flat[total_mask].astype(int)]
    return warped_image



def get_6d_pc_motion(points_3d, original_pose, new_pose, samplepts=50):
    # Convert points to homogeneous coordinates
    points_homogeneous = np.hstack((points_3d, np.ones((points_3d.shape[0], 1)))).T
    
    ref_camera_coords = (original_pose @ points_homogeneous)[:3, :]
    new_camera_coords = (new_pose @ points_homogeneous)[:3, :]

    # concatenate along first dimension: 6xN
    motion = np.concatenate((ref_camera_coords, new_camera_coords), axis=0)
    # transpose to Nx6 then randomly select samplepts points from N
    motion = motion.transpose(1,0)
    #motion = motion[np.random.choice(motion.shape[0], samplepts, replace=False), :]

    # N = motion.shape[0]
    # num_to_select = min(300, N)  # Choose 300 or N, whichever is smaller
    # selected_indices = np.random.choice(N, num_to_select, replace=False)  # Randomly select indices
    # selected = motion[selected_indices]

    return motion

# instead of doing warp depth, take the pixel locations in each image; no need for projecting the 3d points!
# i.e. use imgname_dict['pointids'] to find overlap, then imgname_dict['pointxy'] to get the pixel locations
def warp_image_from_pc(original_image, points_3d, fx, fy, new_pose):
    # Assuming depth_map is aligned with the original_image and in the same scale.
    
    # Initialize the warped image with black pixels
    warped_image = np.zeros_like(original_image)

    # Intrinsic camera parameters
    cx, cy = original_image.shape[1] / 2, original_image.shape[0] / 2

    # Create a meshgrid of pixel coordinates
    height, width = original_image.shape[:2]
    x = np.linspace(0, width - 1, width)
    y = np.linspace(0, height - 1, height)
    xv, yv = np.meshgrid(x, y)

    points_world = np.hstack((points_3d, np.ones((points_3d.shape[0], 1)))).T

    # 4,N; 4,4; 4,N; N,4
    #print("shapes: ", points_world.shape, original_pose.shape, points_3d_homogeneous.T.shape, points_3d_homogeneous.shape)

    # Convert points from world coordinates to new camera coordinates
    points_new_camera = new_pose @ points_world

    # Project points back onto the new image plane
    x_new = fx * points_new_camera[0, :] / points_new_camera[2, :] + cx
    y_new = fy * points_new_camera[1, :] / points_new_camera[2, :] + cy


    valid_mask = (~np.isnan(x_new)) & (~np.isnan(y_new))
    x_new_int = x_new.astype(int)
    y_new_int = y_new.astype(int)
    
    # Further refine mask for coordinates within image boundaries
    in_bounds_mask = (0 <= x_new_int) & (x_new_int < width) & (0 <= y_new_int) & (y_new_int < height)
    total_mask = valid_mask & in_bounds_mask
    
    # Apply mask and copy pixels
    xv_flat = xv.flatten()
    yv_flat = yv.flatten()
    
    warped_image[y_new_int[total_mask], x_new_int[total_mask]] = original_image[yv_flat[total_mask].astype(int), xv_flat[total_mask].astype(int)]

    return warped_image

    

def visualize_sparse_depth(depthmap, radius=3):
    non_zero_coordinates = np.nonzero(depthmap)
    mask = np.zeros((depthmap.shape[0], depthmap.shape[1], 1), dtype=np.uint8)
    for y, x in zip(*non_zero_coordinates):
        mask[y-radius:y+radius,x-radius:x+radius,:] = depthmap[y,x]  #np.array([255,0,0])
    return mask