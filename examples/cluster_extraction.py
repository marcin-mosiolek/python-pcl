'''
    This file is a python port of the c++ tutorial from:
    http://pointclouds.org/documentation/tutorials/cluster_extraction.php

    You will also need the file, which was used in that tutorial. It might
    be found here:
    https://raw.github.com/PointCloudLibrary/data/master/tutorials/table_scene_lms400.pcd

    The output of this code, are cloud_cluster_X.pcd files, that might be
    viewed with the help of pcl_viewer.
'''
import pcl

# Read in the cloud data
my_pcl = pcl.load("table_scene_lms400.pcd")
print("PointCloud before filtering has: {} data points".format(my_pcl.size))

# Create the filtering object: downsample
# the dataset using a leaf size of 1cm
vg_filter = my_pcl.make_voxel_grid_filter()
vg_filter.set_leaf_size(0.01, 0.01, 0.01)
cloud_filtered = vg_filter.filter()
print("PointCloud after filtering has:"
      "{} data points".format(cloud_filtered.size))

nr_points = cloud_filtered.size

while cloud_filtered.size > 0.3 * nr_points:
    # Create the segmentation object for the planar model
    # and set all the parameters
    seg = cloud_filtered.make_segmenter()
    seg.set_optimize_coefficients(True)
    seg.set_model_type(pcl.SACMODEL_PLANE)
    seg.set_method_type(pcl.SAC_RANSAC)
    seg.set_max_iterations(100)
    seg.set_distance_threshold(0.02)

    indices, model = seg.segment()
    cloud_plane = cloud_filtered.extract(indices, False)
    print("PointCloud representing the planar"
          "component: {} data points".format(cloud_plane.size))
    cloud_filtered = cloud_filtered.extract(indices, True)

# Creating the KdTree object for the search method of the extraction
kdtree = pcl.KdTree(cloud_filtered)

# Creating euclidean cluster extraction
ec = cloud_filtered.make_euclidean_cluster_extractor()
ec.set_cluster_tolerance(0.02)
ec.set_min_cluster_size(100)
ec.set_max_cluster_size(25000)
ec.set_search_method(kdtree)
ec.set_input_cloud(cloud_filtered)
cluster_indices = ec.extract()

for i, indices in enumerate(cluster_indices):
    cluster_points = [cloud_filtered[ind] for ind in indices]
    pc = pcl.PointCloud(cluster_points)
    # Setting width and height explicitly is not required, as
    # its already done in from_list method in class PointCloud,
    # however for completness with c++ example, let's do it again..
    pc.width = pc.size
    pc.height = 1
    pc.is_dense = True
    print("PointCloud representing the cluster:"
          "{} data points.".format(pc.size))
    file_name = str.encode("cloud_cluster_{}.pcd".format(i))
    pc.to_file(file_name)
