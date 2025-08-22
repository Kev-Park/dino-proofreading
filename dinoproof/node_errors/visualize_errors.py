import neurerrors
import caveclient
import numpy as np
import torch
import torch_geometric
import tqdm
import pandas

### Caveclient initialization ###
datastack_name = "flywire_fafb_public"
voxel_resolution = np.array([16,16,40])

client = caveclient.CAVEclient(datastack_name)
client.materialize.version = 783 # FlyWire public version


dataset_client = neurerrors.Dataset_Client(client)


input_path = 'ID_tests.txt'
root_to_old_seg_ids, old_seg_ids = dataset_client.get_old_seg_ids_from_txt_file(input_path, min_l2_size=10, show_progress=True)
print(root_to_old_seg_ids)

#old_seg_ids = [720575940620826450] # Example with one old segment ID
attributes_list = ['rep_coord_nm', 'size_nm3', 'area_nm2', 'max_dt_nm', 'mean_dt_nm', 'pca_val']
graph_dataset = dataset_client.build_graph_dataset(old_seg_ids, attributes_list, show_progress=True, verbose=False)

dataset_client.associate_error_to_graph_dataset(graph_dataset, voxel_resolution=voxel_resolution, show_progress=True) # you can add find_operation_weights=True to find the weights of the operations, but this adds a big time cost
#dataset_client.normalize_features(graph_dataset, flywire_normalization=True) #if you want to normalize the features for more stable training, normalizes data.x (features) to have a mean of 0 and a std of 1. 
# Standard normalization over 20,000 neurons of Flywire public dataset.

# FlyWire
data_point = graph_dataset[0]
print(data_point)
url = dataset_client.get_url_for_visualization(seg_id=data_point.metadata['seg_id'], error_features=data_point.error_features, voxel_resolution=voxel_resolution, local_host=True)
# CA3
# url = dataset_client.get_url_for_visualization(seg_id=data_point.metadata['seg_id'], em_data_url=em_data_url_ca3, segmentation_url=segmentation_url_ca3, error_features=data_point.error_features, voxel_resolution=voxel_resolution, local_host=True)
print(url)