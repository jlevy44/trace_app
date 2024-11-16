import json
import os
import numpy as np
import pandas as pd
import geopandas
import cv2
import tifffile as tiff
import scipy.ndimage as ndimage
from collections import OrderedDict
from dash import dash_table
from dash.exceptions import PreventUpdate
from skimage import draw
from dask.diagnostics import ProgressBar
import dask
import plotly.express as px
from skimage import draw

def poly2mask(vertex_row_coords, vertex_col_coords, shape):
    fill_row_coords, fill_col_coords = draw.polygon(vertex_row_coords, vertex_col_coords, shape)
    mask = np.zeros(shape, dtype=np.bool_)
    mask[fill_row_coords, fill_col_coords] = True
    return mask

def polygon_to_path(polygon, 
                    im_shape,
                   ):
    coordinates = polygon.astype(float).tolist()#replace("POLYGON ((", "").replace("))", "").split(", ")
    path_points = []
    for x,y in coordinates:
        path_points.append(f"{x},{y}")# path_points.append(f"{x},{(im_shape[0] - y)}")
        # path_points.append(f"{x/im_shape[1]},{(im_shape[0] - y)/im_shape[0]}")
    path_string = "M" + "L".join(path_points) + "Z"  # Add "M" at the beginning and "Z" at the end
    
    return path_string

def path_to_indices(path, im_shape):#rdim, scale_factor):
    """From SVG path to numpy array of coordinates, each row being a (row, col) point
    """
    indices_str = [
        el.replace("M", "").replace("Z", "").split(",") for el in path.split("L")
    ]
    path=np.rint(np.array(indices_str, dtype=float)*np.array(im_shape[::-1])).astype(np.int)[:,::-1]
    path[:,0]=im_shape[0]-path[:,0]
    return path[:,::-1]

def path_to_mask_old(path, shape, scale_factor, ):
    path_work = path_to_indices(path, shape)#[0], scale_factor)
    cols, rows = path_work.T
    rr, cc = draw.polygon(rows, cols)
    rr_new, cc_new = [], []
    for one_rr in rr:
        if one_rr >= shape[1]:
            rr_new.append(shape[1]-1)
        else:
            rr_new.append(one_rr)
    for one_cc in cc:
        if one_cc >= shape[0]:
            cc_new.append(shape[0]-1)
        else:
            cc_new.append(one_cc)
    mask = np.zeros(shape, dtype=np.bool)

    mask[rr_new, cc_new] = True
    mask = ndimage.binary_fill_holes(mask)
    return mask

def path_to_mask(path, shape, scale_shape=False, scale_factor=1., flip_y=True):
    path=np.array([np.array(x.split(",")) for x in path[1:-1].split("L")]).astype(float)[:,::-1]
    if scale_shape: path*=np.array(shape)
    if scale_factor!=1: path/=scale_factor
    if flip_y: path[:,0]=shape[0]-path[:,0]
    mask=poly2mask(path[:,0],path[:,1],shape)
    mask=ndimage.binary_fill_holes(mask)
    return mask

def update_data_table(config,previous_data, current_data):
    if previous_data is None:
        raise PreventUpdate

    if len(previous_data) > len(current_data):
        index_count = 1
        for one_dict in current_data:
            one_dict['index'] = index_count
            index_count += 1
        return current_data
    
    return current_data


def update_table_callback(config, n_clicks, boxplot_dropdown, boxplot_log_scale, boxplot_metal_type, selected_elements, relayout_data, table_children):
    if n_clicks is None:
        return table_children, px.box()
    
    if n_clicks == 0:
        return table_children, px.box()
    
    all_type_list = []
    for one_shape in config.all_relayout_data['shapes']:
        all_type_list.append(config.color_to_type_dict[one_shape['line']['color']])
    all_type_list = list(set(all_type_list))
    
    all_type_area_list_dict = {}
    for one_type in all_type_list:
        all_type_area_list_dict[one_type] = []
    
    for one_shape in config.all_relayout_data['shapes']:
        all_type_area_list_dict[config.color_to_type_dict[one_shape['line']['color']]].append(one_shape['path'])
        
    all_type_area_dict = {}
        
    for one_type in list(all_type_area_list_dict.keys()):
        for one_path in all_type_area_list_dict[one_type]:
            if one_type not in list(all_type_area_dict.keys()):
                all_type_area_dict[one_type] = path_to_mask(one_path, 
                                                            config.metal_data['metals']['All'].shape,
                                                            flip_y=False,
                                                            # config.scale_factor, 
                                                           )
            else:
                all_type_area_dict[one_type] += path_to_mask(one_path, 
                                                             config.metal_data['metals']['All'].shape,
                                                             flip_y=False,
                                                            # config.scale_factor, 
                                                            )
    type_df_list = []
    metal_df_list = []
    mean_df_list = []
    std_df_list = []
    median_df_list = []
    q1_df_list = []
    q3_df_list = []
    
    if "All" in selected_elements:
        metals_to_process = list(config.metal_data['metals'].keys())
    else:
        metals_to_process = selected_elements
    
    for one_type in list(all_type_area_dict.keys()):
        one_area = all_type_area_dict[one_type]
        for one_metal in list(config.metal_data['metals'].keys()):
            original_metal_image = config.metal_data['metals'][one_metal]
            padded_metal_image = original_metal_image
            padded_metal_image = np.nan_to_num(padded_metal_image, nan=0.000001)
            metal_values = padded_metal_image[one_area]
            type_df_list.append(one_type)
            metal_df_list.append(one_metal)
            mean_df_list.append(round(metal_values.mean(), 3))
            std_df_list.append(round(metal_values.std(), 3))
            median_df_list.append(round(np.median(metal_values), 3))
            q1_df_list.append(round(np.percentile(metal_values, 25), 3))
            q3_df_list.append(round(np.percentile(metal_values, 75), 3))
            
    result_df = pd.DataFrame({
        'type': type_df_list, 
        'metal': metal_df_list, 
        'mean': mean_df_list, 
        'std': std_df_list,
        'median': median_df_list,
        'Q1': q1_df_list,
        'Q3': q3_df_list
    })
    
    new_type_df_list = []
    new_metal_df_list = []
    new_value_df_list = []
    
    for one_type in list(all_type_area_dict.keys()):
        one_area = all_type_area_dict[one_type]
        for one_metal in list(config.metal_data['metals'].keys()):
            original_metal_image = config.metal_data['metals'][one_metal]
            padded_metal_image = original_metal_image
            # padded_metal_image = np.nan_to_num(padded_metal_image, nan=0.000001)
            new_values = padded_metal_image[np.logical_and(one_area, ~np.isnan(padded_metal_image))].tolist()
            new_value_df_list.extend(new_values)
            new_metal_df_list.extend([one_metal]*len(new_values))
            new_type_df_list.extend([one_type]*len(new_values))
                
    if boxplot_log_scale:
        # new_value_df_list = [0.0001 if x <= 0.0001 else x for x in new_value_df_list]
        new_value_df_list = list(np.log1p(new_value_df_list))
    config.box_df = pd.DataFrame({'type': new_type_df_list, 'metal': new_metal_df_list, 
                              'value': new_value_df_list})
    config.newest_result_df = result_df

    annotation_result_table = dash_table.DataTable(
                                data=result_df[result_df['metal'].isin(metals_to_process) if isinstance(selected_elements, list) else result_df['metal'] == selected_elements].to_dict('records'),
                                columns=[{'id': c, 'name': c} for c in result_df.columns],
                                fixed_rows={'headers': True},
                                style_table={'height': 400, 'overflowY': 'auto'}  # defaults to 500
                            )
    if boxplot_metal_type:
        if isinstance(boxplot_dropdown, str):
            box_df_test = config.box_df[config.box_df['type'] == boxplot_dropdown]
        else:
            box_df_test = config.box_df[config.box_df['type'].isin(boxplot_dropdown)]
        boxplot_fig = px.box(box_df_test, x='metal', y='value', color='type', template='plotly_white')
    else:
        if isinstance(boxplot_dropdown, str):
            box_df_test = config.box_df[config.box_df['metal'] == boxplot_dropdown]
        else:
            box_df_test = config.box_df[config.box_df['metal'].isin(boxplot_dropdown)]
        boxplot_fig = px.box(box_df_test, x='type', y='value', color='metal', template='plotly_white')
    return annotation_result_table, boxplot_fig

def update_boxplot_dropdown(config, boxplot_metal_type, type_dropdown_annotation):
    if boxplot_metal_type:
        return type_dropdown_annotation[0], type_dropdown_annotation
    else:
        return list(config.metal_data['metals'].keys())[0], list(config.metal_data['metals'].keys())
