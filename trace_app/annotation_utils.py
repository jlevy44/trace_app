import copy
import json
import os
from collections import OrderedDict

import cv2
import geopandas
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import tifffile as tiff
from dash import html
from dash.exceptions import PreventUpdate
from dash_canvas.utils import array_to_data_url
from matplotlib.colors import Normalize as Colors_Normalize
from skimage import draw
import matplotlib.pyplot as plt
from skimage import exposure
import pysnooper
from .file_utils import parse_contents
from .data_processing import polygon_to_path, path_to_mask
from .image_processing import setup_base_figure, add_image_to_figure, add_click_grid

def output_blank_hne(im_small_crop_annotation_tab):
    fig = setup_base_figure(im_small_crop_annotation_tab, add_clickmode=True)
    fig.update_layout(
        autosize=True,
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
        dragmode='drawclosedpath',
        newshape=dict(
                line=dict(color='blue', width=2),
                fillcolor='rgba(0,0,0,0)',
                opacity=1
            ),
            modebar=dict(
                add=['drawclosedpath']
            )
        )
    return fig

def add_annotation_type(config, n_clicks, 
                        contents, project_name,
                        type_input, previous_types, 
                        filename, last_modified):
    if n_clicks > 0 and type_input and filename == config.file_name_history_1:
        new_option = type_input
        if new_option not in previous_types:
            previous_types.append(new_option)
            already_types_num = len(list(config.type_to_color_dict.keys()))
            config.type_to_color_dict[str(type_input)] = config.all_color_list[already_types_num]
            config.color_to_type_dict[config.all_color_list[already_types_num]] = str(type_input)
            
            color_type_match_list = []
            for one_color in list(config.color_to_type_dict.keys()):
                new_line = html.P(
                    [
                        str(config.color_to_type_dict[one_color]),': ',
                        html.Span("", style={"background-color": one_color, "width": "15px", "height": "15px", "display": "inline-block", "vertical-align": "middle"}),
                        f" {one_color}."
                    ],
                    style={"display": "flex", "align-items": "center"}
                )
                color_type_match_list.append(new_line)
        return previous_types, color_type_match_list, #previous_types
    if filename != None and filename != config.file_name_history_1:
        annotation_data = parse_contents(config,filename)
        # print(annotation_data)
        colname = "name"
        if colname not in annotation_data.columns:
            annotation_data['annot'] = annotation_data['classification' if 'classification' in annotation_data.columns else 'properties'].map(get_name)
        else:
            annotation_data['annot'] = annotation_data[colname]
        gp2 = annotation_data.copy()
        gp2['geometry'] = gp2['geometry'].scale(config.compression_annotation_tab*config.compression_value_again,#*config.compression_value_again,
                                                config.compression_annotation_tab*config.compression_value_again,#*config.compression_value_again,
                                                origin=(0, 0))
        all_types = list(set(list(gp2['annot'])))
        for one_type in all_types:
            new_option = one_type
            if new_option not in previous_types:
                previous_types.append(new_option)
                already_types_num = len(list(config.type_to_color_dict.keys()))
                config.type_to_color_dict[str(one_type)] = config.all_color_list[already_types_num]
                config.color_to_type_dict[config.all_color_list[already_types_num]] = str(one_type)

        color_type_match_list = []
        for one_color in list(config.color_to_type_dict.keys()):
            new_line = html.P(
                [
                    str(config.color_to_type_dict[one_color]),': ',
                    html.Span("", style={"background-color": one_color, "width": "15px", "height": "15px", "display": "inline-block", "vertical-align": "middle"}),
                    f" {one_color}."
                ],
                style={"display": "flex", "align-items": "center"}
            )
            color_type_match_list.append(new_line)
        config.file_name_history_1 = filename
        return previous_types, color_type_match_list, #previous_types
    elif n_clicks > 0 and type_input:
        new_option = type_input
        if new_option not in previous_types:
            previous_types.append(new_option)
            already_types_num = len(list(config.type_to_color_dict.keys()))
            config.type_to_color_dict[str(type_input)] = config.all_color_list[already_types_num]
            config.color_to_type_dict[config.all_color_list[already_types_num]] = str(type_input)
            
            color_type_match_list = []
            for one_color in list(config.color_to_type_dict.keys()):
                new_line = html.P(
                    [
                        str(config.color_to_type_dict[one_color]),': ',
                        html.Span("", style={"background-color": one_color, "width": "15px", "height": "15px", "display": "inline-block", "vertical-align": "middle"}),
                        f" {one_color}."
                    ],
                    style={"display": "flex", "align-items": "center"}
                )
                color_type_match_list.append(new_line)
        return previous_types, color_type_match_list, #previous_types

def process_metal_image(metal_image, threshold, vmin, vmax, colormap):
    padded_metal_image = np.maximum(metal_image, 0.000001)
    padded_metal_image = np.nan_to_num(padded_metal_image, nan=0.000001)
    padded_metal_image = np.log1p(padded_metal_image)
    
    mask = padded_metal_image > threshold
    padded_metal_image_eq = exposure.equalize_hist(padded_metal_image, mask=mask)
    
    if vmin and vmax:
        c_norm = Colors_Normalize(vmin=np.percentile(padded_metal_image_eq, vmin), 
                                    vmax=np.percentile(padded_metal_image_eq, vmax), clip=True)
        padded_metal_image_normalized = c_norm(padded_metal_image_eq)
    else:
        padded_metal_image_normalized = padded_metal_image_eq
    
    cmap_jet = plt.cm.get_cmap(colormap)
    padded_metal_image_rgb = cmap_jet(padded_metal_image_normalized)[:, :, :3]
    
    target_color_to_white = padded_metal_image_rgb[0, 0, :]
    mask_to_white = np.all(padded_metal_image_rgb == target_color_to_white, axis=-1)
    padded_metal_image_rgb[mask_to_white] = [1, 1, 1]
    
    return padded_metal_image_rgb

def create_figure(image_rgb):
    figure = setup_base_figure(image_rgb, add_clickmode=True)

    # figure = go.Figure()
    # figure.update_xaxes(visible=False, range=[0, 1])
    # figure.update_yaxes(visible=False, range=[0, 1], scaleanchor="x", autorange="reversed")
    
    # figure.add_layout_image(
    #     dict(
    #         source=array_to_data_url((image_rgb * 255).astype(np.uint8)),
    #         x=0, sizex=1, y=1, sizey=1,
    #         xref="x", yref="y",
    #         opacity=1.0,
    #         layer="below",
    #         sizing="stretch"
    #     )
    # )
    
    figure.update_layout(
        autosize=True,
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        showlegend=False,
    )
    
    return figure

def get_name(x):
    try:
        return eval(x).get("name","none")
    except:
        return "none"

# @pysnooper.snoop()
def update_annotation_callback(config,
                               selected_metal, 
                               relayout_data,
                               annotation_type,
                               colormap,
                               vmin_vmax,
                               n_clicks,
                               n_clicks_load_data,
                               contents,
                               filename,
                               last_modified,
                               threshold
                              ):
    vmin, vmax = vmin_vmax[0], vmin_vmax[1]

    if n_clicks > 0 and n_clicks != config.history_co_transport_n_clicks:
        metal_image = config.warped_metals[selected_metal]
        config.history_co_transport_n_clicks = n_clicks
        config.all_relayout_data = {'shapes': [], 'dragmode': 'drawclosedpath'}
    else:
        if n_clicks_load_data > 0 and n_clicks_load_data != config.n_clicks_load_data:
            config.n_clicks_load_data = n_clicks_load_data
            config.all_relayout_data = {'shapes': [], 'dragmode': 'drawclosedpath'}
        metal_image = config.metal_data['metals'][selected_metal]

    padded_metal_image_rgb = process_metal_image(metal_image, threshold, vmin, vmax, colormap)
    blank_figure = create_figure(padded_metal_image_rgb)
    blank_hne_image = output_blank_hne(config.im_small_crop_annotation_tab)

    if (selected_metal != config.history_selected_metal or
        colormap != config.history_colormap or
        vmin != config.history_vmin or
        vmax != config.history_vmax):
        config.history_selected_metal = selected_metal
        config.history_colormap = colormap
        config.history_vmin = vmin
        config.history_vmax = vmax
        config.all_relayout_data['dragmode'] = 'drawclosedpath'
    elif filename and filename != config.file_name_history_2:
        annotation_data = parse_contents(config, filename)
        # print(annotation_data)
        colname = "name"
        if colname not in annotation_data.columns:
            annotation_data['annot'] = annotation_data['classification' if 'classification' in annotation_data.columns else 'properties'].map(get_name)
        else:
            annotation_data['annot'] = annotation_data[colname]
        gp2 = annotation_data.copy()
        gp2['geometry'] = gp2['geometry'].scale(config.compression_annotation_tab*config.compression_value_again,#compression_value_again,#config.compression_annotation_tab,#*config.compression_value_again,
                                                config.compression_annotation_tab*config.compression_value_again,#compression_value_again,#config.compression_annotation_tab,#*config.compression_value_again,
                                                origin=(0, 0))
        
        for _, row in gp2.iterrows():
            one_polygon = np.vstack(row.geometry.exterior.coords.xy).T
            one_path = polygon_to_path(one_polygon, config.im_small_crop_annotation_tab.shape)
            append_relayout = {
                'editable': True, 
                'fillcolor': 'rgba(0, 0, 0, 0)', 
                'fillrule': 'evenodd', 
                'layer': 'above', 
                'line': {'color': config.type_to_color_dict[row.annot], 'dash': 'solid', 'width': 2}, 
                'opacity': 1, 'type': 'path', 
                'xref': 'x', 'yref': 'y', 
                'path': one_path
            }
            if append_relayout not in config.all_relayout_data['shapes']:
                config.all_relayout_data['shapes'].append(append_relayout)
        
        config.all_relayout_data['dragmode'] = 'drawclosedpath'
        config.file_name_history_2 = filename
        config.all_annot_data.update({"shapes": config.all_relayout_data["shapes"]})
    elif relayout_data:
        if 'shapes' in relayout_data:
            if len(relayout_data['shapes']) > len(config.all_relayout_data['shapes']):
                append_relayout = relayout_data['shapes'][-1]
                append_relayout['line']['color'] = config.type_to_color_dict[annotation_type]
                config.all_relayout_data['shapes'].append(append_relayout)
            elif len(relayout_data['shapes']) < len(config.all_relayout_data['shapes']):
                config.all_relayout_data['shapes'] = relayout_data['shapes']
        if 'dragmode' in relayout_data:
            config.all_relayout_data['dragmode'] = relayout_data['dragmode']
        for one_shape in config.all_relayout_data['shapes']:
            one_shape['line']['width'] = 2
        config.all_annot_data.update({"shapes": config.all_relayout_data["shapes"]})

    blank_figure['layout'].update(config.all_relayout_data)
    blank_hne_image['layout'].update(config.all_relayout_data)
    return blank_figure, blank_hne_image

