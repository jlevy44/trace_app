import os
import copy
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from dash import html, dash_table
from dash.exceptions import PreventUpdate
from dash_canvas.utils import array_to_data_url
from matplotlib.colors import Normalize as Colors_Normalize
from skimage import measure, morphology
import pickle
import plotly.express as px
import tifffile as tiff
from skimage import exposure
from shapely.geometry import Polygon
from .data_processing import path_to_mask
import pysnooper
from scipy.ndimage import generic_filter, binary_dilation
from .file_utils import generate_files_df_records
from .utils import warp_metals_new, replace_nan_with_median


# Add these helper functions at the top of the file
def add_click_grid(fig, shape):
    """Add invisible grid points for precise clicking"""
    x = []#np.linspace(0, 1, shape[1])# [0, 1] #
    y = []#np.linspace(0, 1, shape[0])# [0, 1] #
    # xx, yy = np.meshgrid(x, y)
    fig.add_scatter(
        x=x,#xx.flatten(), #x, #xx.flatten(),
        y=y,#yy.flatten(), #y, #yy.flatten(),
        mode='markers',
        # marker_color="black",
        marker_opacity=0,
        marker_size=25,
        hoverinfo='none',
        showlegend=False
    )
    return fig

def add_points_trace(fig, x_list, y_list, index_list):
    """Add points with labels to figure"""
    fig.add_scatter(
        x=x_list,
        y=y_list,
        mode='markers+text',
        marker_color='black',
        marker_size=5,
        text=index_list,
        textposition='top center',
        textfont=dict(color='black')
    )
    return fig

def setup_base_figure(img=None,add_clickmode=False):
    """Create base figure with normalized coordinates and click settings"""
    if img is not None:
        fig=px.imshow(img, binary_string=True)
    else:   
        fig = go.Figure()
    fig.update_xaxes(visible=False)#, range=[0, 1])
    fig.update_yaxes(visible=False)#, range=[0, 1], scaleanchor="x")
    fig.update_layout(
        autosize=True,
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        showlegend=False,
    )
    if add_clickmode:
        fig.update_layout(clickmode='event+select', dragmode=False)
    return fig

def add_image_to_figure(fig, img, as_uint8=False):
    """Add image to figure with normalized coordinates"""
    if as_uint8:
        img = (img * 255).astype(np.uint8)
    fig.add_layout_image(
        dict(
            source=array_to_data_url(img),
            x=0, sizex=1,
            y=1, sizey=1,
            xref="x", yref="y",
            opacity=1.0,
            layer="below",
            sizing="stretch"
        )
    )
    return fig

def output_blank_hne(im_small_crop_annotation_tab):
    fig = setup_base_figure()
    fig = add_image_to_figure(fig, im_small_crop_annotation_tab)
    fig = add_click_grid(fig, im_small_crop_annotation_tab.shape[:2])
    return fig

# @pysnooper.snoop()
def generate_tissue_mask_func(config, n_clicks, threshold, old_selected_rows, metal_dropdown_pre, metal_colormap_pre):

    if n_clicks > 0:
        # Combine selected metals
        if "All" in metal_dropdown_pre:
            all_metals = config.metal_data['metals']["All"]
        elif len(metal_dropdown_pre) == 1:
            all_metals = config.metal_data['metals'][metal_dropdown_pre[0]]
        else:
            all_metals = np.sum([config.metal_data['metals'][i] for i in metal_dropdown_pre], axis=0)
        
        # Process image
        image = replace_nan_with_median(all_metals)#generic_filter(image, np.nanmean, size=3, mode='constant', cval=np.nan)#replace_nan_with_mean(all_metals)
        blurred = cv2.GaussianBlur(np.log1p(image), (1, 1), 0)

        # Generate mask
        tissue_mask = np.logical_and(blurred >= threshold,~np.isnan(all_metals))
        selem = morphology.disk(1)
        tissue_mask = morphology.opening(~tissue_mask, selem)
        tissue_mask = morphology.closing(tissue_mask, selem) > 0   
        final_mask = morphology.remove_small_objects(
            morphology.remove_small_holes(tissue_mask, 10000), 50000
        )
        final_mask = ~final_mask
        # pd.to_pickle(dict(image=image, final_mask=final_mask, blurred=blurred,all_metals=all_metals,tissue_mask=tissue_mask),"./test.pkl")

        # # Process metal image for display
        # padded_metal_image = np.log1p(np.maximum(all_metals, 0.000001))
        
        # # Apply histogram equalization with mask
        # padded_metal_image_eq = exposure.equalize_hist(padded_metal_image, mask=np.logical_and(final_mask, ~np.isnan(padded_metal_image)))
        
        # # Normalize the equalized image
        # # padded_metal_image_normalized = (padded_metal_image_eq - padded_metal_image_eq.min()) / (padded_metal_image_eq.max() - padded_metal_image_eq.min())
        
        # # Generate RGB image
        # cmap_jet = plt.cm.get_cmap(metal_colormap_pre)
        padded_metal_image_rgb = config.padded_metal_image_rgb.copy()#cmap_jet(padded_metal_image_eq)[:, :, :3] # padded_metal_image_normalized
        # get edges of final_mask, dilate, then set to blue within rgb
        edges = np.logical_xor(final_mask, binary_dilation(final_mask, morphology.disk(20)))
        padded_metal_image_rgb[edges] = [0, 0, 255]
        # Set corners to white
        # corners = [(0,0), (-1,0), (0,-1), (-1,-1)]
        # for corner in corners:
        #     mask_to_white = np.all(padded_metal_image_rgb == padded_metal_image_rgb[corner], axis=-1)
        #     padded_metal_image_rgb[mask_to_white] = [1, 1, 1]

        # # Convert to 8-bit image
        # padded_metal_image_rgb = (padded_metal_image_rgb * 255).astype(np.uint8)

        # Generate contour image directly
        # fig, ax = plt.subplots()
        # ax.imshow(padded_metal_image_rgb)
        # for contour in measure.find_contours(final_mask, 0.5):
        #     ax.plot(contour[:, 1], contour[:, 0], linewidth=2, color="blue")
        # ax.axis('off')
        # fig.tight_layout(pad=0)
        
        # # Convert matplotlib figure to image array
        # fig.canvas.draw()
        # contour_image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        # contour_image = contour_image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        # plt.close(fig)

        blank_figure_mask = go.Figure()

        blank_img_width = padded_metal_image_rgb.shape[1]
        blank_img_height = padded_metal_image_rgb.shape[0]
        scale_factor = 0.3

    
        blank_figure_mask.update_xaxes(
            visible=False,
            range=[0, blank_img_width * scale_factor]
        )
        blank_figure_mask.update_yaxes(
            visible=False,
            range=[0, blank_img_height * scale_factor],
            scaleanchor="x",
        )
        blank_figure_mask.add_layout_image(
            dict(
                x=0,
                sizex=blank_img_width * scale_factor,
                y=blank_img_height * scale_factor,
                sizey=blank_img_height * scale_factor,
                xref="x",
                yref="y",
                opacity=1.0,
                layer="below",
                sizing="stretch",
                source=array_to_data_url((padded_metal_image_rgb * 255).astype(np.uint8)),)
        )

        # Add empty scatter trace with mode=markers+text
        blank_figure_mask.add_trace(go.Scatter(
            x=[],
            y=[],
            mode='markers+text',
            marker=dict(color='black', size=5),
            text=[],
            textposition="top center"
        ))

        blank_figure_mask.update_layout(
            autosize=True,
            margin={"l": 0, "r": 0, "t": 0, "b": 0},
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            showlegend=False,
        )

        # # Create Plotly figure
        # blank_figure_mask = go.Figure()

        # blank_figure_mask_height, blank_figure_mask_width = contour_image.shape[:2]
        # scale_factor = (config.metal_data['metals'][list(config.metal_data['metals'].keys())[0]].shape[1]/contour_image.shape[1])*0.32


        # blank_figure_mask.update_xaxes(
        #     visible=False,
        #     range=[0, blank_figure_mask_width * scale_factor]
        # )
        # blank_figure_mask.update_yaxes(
        #     visible=False,
        #     range=[0, blank_figure_mask_height * scale_factor],
        #     scaleanchor="x",
        # )
        # blank_figure_mask.add_layout_image(
        #     dict(
        #         x=0,
        #         sizex=blank_figure_mask_width * scale_factor,
        #         y=blank_figure_mask_height * scale_factor,
        #         sizey=blank_figure_mask_height * scale_factor,
        #         xref="x",
        #         yref="y",
        #         opacity=1.0,
        #         layer="below",
        #         sizing="stretch",
        #         source=array_to_data_url(contour_image),
        #     )
        # )

        # # Add empty scatter trace with mode=markers+text
        # blank_figure_mask.add_trace(go.Scatter(
        #     x=[],
        #     y=[],
        #     mode='markers+text',
        #     marker=dict(color='black', size=5),
        #     text=[],
        #     textposition="top center"
        # ))

        # blank_figure_mask.update_layout(
        #     autosize=True,
        #     margin={"l": 0, "r": 0, "t": 0, "b": 0},
        #     paper_bgcolor='rgba(0,0,0,0)',
        #     plot_bgcolor='rgba(0,0,0,0)',
        #     showlegend=False,
        # )

        # Apply mask to metal data
        for one_key in config.metal_data_preprocess['metals'].keys():
            config.metal_data_preprocess['metals'][one_key][~final_mask] = np.nan

        # Save preprocessed data
        dump_path = f'./data/{config.selected_project}/preprocess_metals.pkl'
        with open(dump_path, 'wb') as handle:
            pickle.dump(config.metal_data_preprocess, handle)

        # Update file list
        files_df, new_selected_rows = generate_files_df_records(config, old_selected_rows)
        config.upload_files_df_data = files_df
        display_files_table = dash_table.DataTable(
            id='files_table',
            selected_rows=new_selected_rows,
            columns=[{"name": i, "id": i} for i in files_df.columns],
            data=files_df,
            row_selectable='multi'
        )

        return blank_figure_mask, [display_files_table]
    else:
        raise PreventUpdate
        
def calculate_mask_threshold(config, selected_metal, relayout_data):
    if not relayout_data or 'shapes' not in relayout_data or not relayout_data['shapes']:
        return np.nan
        
    # Get shape path
    shape = relayout_data['shapes'][-1]  # Use the last shape
    if 'path' not in shape:
        return np.nan
    
    # Get image dimensions from the first metal
    metal_shape = next(iter(config.metal_data['metals'].values())).shape
    

    if selected_metal == []:
        selected_metal = ['All']
    append_count = 0
    for i_index in selected_metal:
        i = config.metal_data['metals'][i_index]
        if append_count == 0:
            all_metals = i
        else:
            all_metals += i
        append_count += 1
    original_metal_image = all_metals

    image = replace_nan_with_median(all_metals)#generic_filter(all_metals, np.nanmean, size=3, mode='constant', cval=np.nan)#replace_nan_with_mean(all_metals)
    blurred = cv2.GaussianBlur(np.log1p(image), (1, 1), 0)
    
    # Create mask from path_
    mask = np.zeros(metal_shape, dtype=bool)
    for shape in relayout_data['shapes']:
        if 'path' not in shape:
            continue
        mask = np.logical_or(mask, path_to_mask(shape['path'], metal_shape, scale_shape=False, scale_factor=0.3))
    mask = np.logical_and(mask,~np.isnan(original_metal_image))
    # pd.to_pickle(dict(all_metals=all_metals,image=image, blurred=blurred, mask=mask, shapes=relayout_data['shapes']),"./test.pkl")
    mean_conc = float(np.nanmean(blurred[mask]))
    return mean_conc

def update_pre_metals(config, colormap, selected_metal, vmin_vmax, threshold, all_relayout_data):
    vmin, vmax = vmin_vmax[0], vmin_vmax[1]
    if selected_metal == []:
        selected_metal = ['All']
    append_count = 0
    for i_index in selected_metal:
        i = config.metal_data['metals'][i_index]
        if append_count == 0:
            all_metals = i
        else:
            all_metals += i
        append_count += 1
    original_metal_image = all_metals
    padded_metal_image = original_metal_image
    padded_metal_image[padded_metal_image <= 0] = 0.000001
    padded_metal_image = np.nan_to_num(padded_metal_image, nan=0.000001)
    
    # Apply log1p transformation
    padded_metal_image = np.log1p(padded_metal_image)
    
    # Create mask using the provided threshold
    mask = padded_metal_image > threshold
    
    # Apply histogram equalization with mask
    padded_metal_image_eq = exposure.equalize_hist(padded_metal_image, mask=mask)
    
    if vmin and vmax:
        c_norm = Colors_Normalize(vmin=np.percentile(padded_metal_image_eq, vmin), 
                                  vmax=np.percentile(padded_metal_image_eq, vmax), clip=True)
        padded_metal_image_normalized = c_norm(padded_metal_image_eq)
    else:
        padded_metal_image_normalized = padded_metal_image_eq

    cmap_jet = plt.cm.get_cmap(colormap)
    padded_metal_image_rgb = cmap_jet(padded_metal_image_normalized)
    padded_metal_image_rgb = padded_metal_image_rgb[:, :, :3]
    target_color_to_white = padded_metal_image_rgb[0, 0, :]
    replacement_color_to_white = np.array([1, 1, 1])
    mask_to_white = np.all(padded_metal_image_rgb == target_color_to_white, axis=-1)
    padded_metal_image_rgb[mask_to_white] = replacement_color_to_white
    config.padded_metal_image_rgb = padded_metal_image_rgb
    blank_figure = go.Figure()

    blank_img_width = original_metal_image.shape[1]
    blank_img_height = original_metal_image.shape[0]
    scale_factor = 0.3

   
    blank_figure.update_xaxes(
        visible=False,
        range=[0, blank_img_width * scale_factor]
    )
    blank_figure.update_yaxes(
        visible=False,
        range=[0, blank_img_height * scale_factor],
        scaleanchor="x",
    )
    blank_figure.add_layout_image(
        dict(
            x=0,
            sizex=blank_img_width * scale_factor,
            y=blank_img_height * scale_factor,
            sizey=blank_img_height * scale_factor,
            xref="x",
            yref="y",
            opacity=1.0,
            layer="below",
            sizing="stretch",
            source=array_to_data_url((padded_metal_image_rgb * 255).astype(np.uint8)),)
    )

    # Add empty scatter trace with mode=markers+text
    blank_figure.add_trace(go.Scatter(
        x=[],
        y=[],
        mode='markers+text',
        marker=dict(color='black', size=5),
        text=[],
        textposition="top center"
    ))

    blank_figure_right = copy.deepcopy(blank_figure)

    blank_figure.update_layout(
        autosize=True,
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        showlegend=False,
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
    blank_figure_right.update_layout(
        autosize=True,
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        showlegend=False,
    )

    # Add shapes from all_relayout_data to blank_figure
    if (all_relayout_data is not None) and 'shapes' in all_relayout_data:
        for shape in all_relayout_data['shapes']:
            blank_figure.add_shape(shape)

    return blank_figure, blank_figure_right

# @pysnooper.snoop()
def change_two_images_and_clean_point_table(config, n_clicks, selected_rows):
    if n_clicks > 0 and selected_rows:
        project_df_list, file_name_df_list, file_type_list = [], [], []
        all_files_list = [f for f in os.listdir('./data/') if f != '.DS_Store']
        
        for project in all_files_list:
            data_files = [f for f in os.listdir(f'./data/{project}') if f != '.DS_Store' and 'small' not in f]
            for data_file in data_files:
                ext = os.path.splitext(data_file)[1]
                if ext in config.file_extensions:
                    file_type_list.append('WSI')
                elif ext == ".pkl":
                    file_type_list.append('Metals')
                elif ext == ".zarr":
                    file_type_list.append("Exported Metals")
                elif ext in ['.xml', ".json", ".geojson"]:
                    file_type_list.append('Annotations')
                project_df_list.append(project)
                file_name_df_list.append(data_file)
        
        for idx in selected_rows:
            file_name = file_name_df_list[idx]
            project_name = project_df_list[idx]
            config.selected_project = project_name
            
            file_path = f'./data/{project_name}/{file_name}'
            if os.path.splitext(file_path)[1] in config.file_extensions:
                if not file_name.endswith("_small.tiff"):
                    file_path = f"{os.path.splitext(file_path)[0]}_small.tiff"
                config.im_small_crop_annotation_tab = tiff.imread(file_path)
                config.im_small_crop_co = tiff.imread(file_path)
                config.hne_shape = config.im_small_crop_co.shape
                select_tiff_file = file_name
            else:
                with open(file_path, "rb") as input_file:
                    config.metal_data = pickle.load(input_file)
                config.select_pkl_file = file_name
    else:
        raise PreventUpdate

    if config.select_pkl_file != config.last_select_pkl_file or select_tiff_file != config.last_select_tiff_file:
        if select_tiff_file != config.last_select_tiff_file and config.metal_data:
            x_metal, y_metal = config.metal_data['metals']['All'].shape
            x_hne, y_hne = config.im_small_crop_co.shape[:2]
            compression_value = (max(x_hne, y_hne) / max(x_metal, y_metal) + min(x_hne, y_hne) / min(x_metal, y_metal)) / 2
            config.compression_value_again = 1 / compression_value
            config.im_small_crop_co = cv2.resize(config.im_small_crop_co, None, fx=1/compression_value, fy=1/compression_value)
            config.im_small_crop_annotation_tab = cv2.resize(config.im_small_crop_annotation_tab, None, fx=1/compression_value, fy=1/compression_value)
            config.hne_shape = config.im_small_crop_co.shape

        selected_metal = list(config.metal_data['metals'].keys())[0]
        all_metals_list = list(config.metal_data['metals'].keys())
        colormap = 'jet'
        vmin_vmax = [0, 100]

        update_df = pd.DataFrame({'index': [], 'hne x': [], 'hne y': [], 'metals x': [], 'metals y': []})
        xy_coords_table = dash_table.DataTable(
            id='datatable_coord_co',
            columns=[{"name": i, "id": i} for i in update_df.columns],
            data=update_df.to_dict('records'),
            editable=True, row_deletable=True,
        )

        white_fig_co = setup_base_figure(np.ones((100, 100, 3), dtype=np.uint8) * 255,add_clickmode=True)

        # white_fig_co = add_image_to_figure(white_fig_co, np.ones((100, 100, 3), dtype=np.uint8) * 255)

        # Add invisible trace with grid points
        white_fig_co = add_click_grid(white_fig_co, (100, 100))

        white_fig_co.update_layout(
            autosize=True,
            margin={"l": 0, "r": 0, "t": 0, "b": 0},
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            showlegend=False,
        )

        vmin, vmax = vmin_vmax

        original_metal_image = config.metal_data['metals'][selected_metal]
        # padded_metal_image = np.maximum(np.log1p(original_metal_image), 0.000001)
        
        # if vmin and vmax:
        #     c_norm = Colors_Normalize(vmin=np.percentile(padded_metal_image, vmin), 
        #                               vmax=np.percentile(padded_metal_image, vmax), clip=True)
        #     padded_metal_image_normalized = c_norm(padded_metal_image)
        # else:
        #     padded_metal_image_normalized = (padded_metal_image - padded_metal_image.min()) / (padded_metal_image.max() - padded_metal_image.min())
        
        cmap_jet = plt.cm.get_cmap(colormap)
        # padded_metal_image_rgb = cmap_jet(padded_metal_image_normalized)[:, :, :3]
        # padded_metal_image_rgb[np.all(padded_metal_image_rgb == padded_metal_image_rgb[0, 0], axis=-1)] = [1, 1, 1]

        blank_figure = setup_base_figure(add_clickmode=True)
        blank_figure = add_image_to_figure(blank_figure, config.im_small_crop_annotation_tab)
        blank_figure = add_click_grid(blank_figure, config.im_small_crop_annotation_tab.shape[:2])

        blank_figure.update_layout(
            autosize=True,
            margin={"l": 0, "r": 0, "t": 0, "b": 0},
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            showlegend=False,
        )

        blank_hne_image = setup_base_figure(add_clickmode=True)
        blank_hne_image = add_image_to_figure(blank_hne_image, config.im_small_crop_annotation_tab)
        blank_hne_image = add_click_grid(blank_hne_image, config.im_small_crop_annotation_tab.shape[:2])

        blank_hne_image.update_layout(
            autosize=True,
            margin={"l": 0, "r": 0, "t": 0, "b": 0},
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            showlegend=False
        )

        # config.all_relayout_data = {
        #     'shapes': [], 
        #     'dragmode': None #'drawclosedpath',
        # }
        # blank_figure['layout'].update(config.all_relayout_data)
        # blank_hne_image['layout'].update(config.all_relayout_data)

        all_metals = np.sum([config.metal_data['metals'][metal] for metal in config.metal_data['metals']], axis=0)
        padded_all_metals = np.maximum(np.log1p(all_metals), 0.000001)
        
        if vmin and vmax:
            c_norm = Colors_Normalize(vmin=np.percentile(padded_all_metals, vmin), 
                                      vmax=np.percentile(padded_all_metals, vmax), clip=True)
            padded_all_metals_normalized = c_norm(padded_all_metals)
        else:
            padded_all_metals_normalized = (padded_all_metals - padded_all_metals.min()) / (padded_all_metals.max() - padded_all_metals.min())
        
        padded_all_metals_rgb = cmap_jet(padded_all_metals_normalized)[:, :, :3]
        padded_all_metals_rgb[np.all(padded_all_metals_rgb == padded_all_metals_rgb[0, 0], axis=-1)] = [1, 1, 1]

        blank_figure_pre = setup_base_figure((padded_all_metals_rgb * 255).astype(np.uint8), add_clickmode=True)
        blank_figure_pre = add_click_grid(blank_figure_pre, padded_all_metals_rgb.shape[:2])

        blank_figure_pre_right = copy.deepcopy(blank_figure_pre)
        blank_figure_pre.update_layout(
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
        blank_figure_pre_right.update_layout(
            autosize=True,
            margin={"l": 0, "r": 0, "t": 0, "b": 0},
        )

        fig_hne_co = setup_base_figure(config.im_small_crop_annotation_tab, add_clickmode=True)
        fig_hne_co = add_click_grid(fig_hne_co, config.im_small_crop_annotation_tab.shape[:2])
        # fig_hne_co = add_image_to_figure(fig_hne_co, config.im_small_crop_annotation_tab)

        fig_hne_co.update_layout(
            autosize=True,
            margin={"l": 0, "r": 0, "t": 0, "b": 0},
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            showlegend=False,
        )

        config.file_name_history_2 = ''
        
        return [], white_fig_co, xy_coords_table, list(config.metal_data['metals'].keys()), selected_metal, blank_figure, blank_hne_image, fig_hne_co, selected_metal, all_metals_list, colormap, vmin_vmax, [selected_metal], all_metals_list, colormap, vmin_vmax, ['All'], ['All']+all_metals_list, colormap, vmin_vmax, blank_figure_pre, blank_figure_pre_right, selected_metal, list(config.metal_data['metals'].keys()), 0, 0, 0

# @pysnooper.snoop()
def update_back_to_image(config, table_data, metal_colormap_co, metal_dropdown_co, vmin_vmax_input_co, threshold):
    vmin, vmax = vmin_vmax_input_co[0], vmin_vmax_input_co[1]
    
    if len(table_data) == 0:
        fig_hne_co = setup_base_figure(config.im_small_crop_annotation_tab, add_clickmode=True)
        # fig_hne_co = add_image_to_figure(fig_hne_co, config.im_small_crop_annotation_tab)
        fig_hne_co = add_click_grid(fig_hne_co, config.im_small_crop_annotation_tab.shape[:2])
        
        padded_rows_co = 1000
        padded_columns_co = 1600

        append_count = 0

        for i_index in metal_dropdown_co:
            i = config.metal_data['metals'][i_index]
            if append_count == 0:
                original_metal_image_co = i
            else:
                original_metal_image_co += i
            append_count += 1

        
        padded_metal_image_co = original_metal_image_co
        padded_metal_image_co[padded_metal_image_co <= 0] = 0.000001
        padded_metal_image_co = np.nan_to_num(padded_metal_image_co, nan=0.000001)
        padded_metal_image_co = np.log1p(padded_metal_image_co)
        
        # Create mask using the provided threshold
        mask = padded_metal_image_co > threshold
        
        # Apply histogram equalization with mask
        padded_metal_image_eq = exposure.equalize_hist(padded_metal_image_co, mask=mask)
        
        if vmin and vmax:
            c_norm = Colors_Normalize(vmin=np.percentile(padded_metal_image_eq, vmin/100), 
                                      vmax=np.percentile(padded_metal_image_eq, vmax/100), clip=True)
            padded_metal_image_normalized_co = c_norm(padded_metal_image_eq)
        else:
            padded_metal_image_normalized_co = padded_metal_image_eq
        
        cmap_jet = plt.cm.get_cmap(metal_colormap_co)
        padded_metal_image_rgb_co = cmap_jet(padded_metal_image_normalized_co)
        padded_metal_image_rgb_co = padded_metal_image_rgb_co[:, :, :3]
        target_color_to_white_co = padded_metal_image_rgb_co[0, 0, :]
        replacement_color_to_white_co = np.array([1, 1, 1])
        mask_to_white_co = np.all(padded_metal_image_rgb_co == target_color_to_white_co, axis=-1)
        padded_metal_image_rgb_co[mask_to_white_co] = replacement_color_to_white_co

        fig_metal_co = setup_base_figure((padded_metal_image_rgb_co * 255).astype(np.uint8), add_clickmode=True)
        fig_metal_co = add_click_grid(fig_metal_co, padded_metal_image_rgb_co.shape[:2])
        
        fig_metal_co.update_layout(
            autosize=True,
            margin={"l": 0, "r": 0, "t": 0, "b": 0},
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            showlegend=False,
            # dragmode='select',
            # clickmode='event+select'
        )
    else:
        index_list = []
        hne_x = []
        hne_y = []
        metal_x = []
        metal_y = []
        
        index_count = 1
        for one_dict in table_data:
            index_list.append(index_count)
            hne_x.append(one_dict['hne x'])
            hne_y.append(one_dict['hne y'])
            metal_x.append(one_dict['metals x'])
            metal_y.append(one_dict['metals y'])
            index_count += 1 

        index_list_to_hne_image = []
        x_list_to_hne_image = []
        y_list_to_hne_image = []
        index_list_to_metal_image = []
        x_list_to_metal_image = []
        y_list_to_metal_image = []
        for one_row_index in range(len(index_list)):
            if hne_x[one_row_index] != '-' and hne_y[one_row_index] != '-':
                index_list_to_hne_image.append(index_list[one_row_index])
                x_list_to_hne_image.append(hne_x[one_row_index])
                y_list_to_hne_image.append(hne_y[one_row_index])
            if metal_x[one_row_index] != '-' and metal_y[one_row_index] != '-':
                index_list_to_metal_image.append(index_list[one_row_index])
                x_list_to_metal_image.append(metal_x[one_row_index])
                y_list_to_metal_image.append(metal_y[one_row_index])
        
        fig_hne_co = setup_base_figure(config.im_small_crop_annotation_tab, add_clickmode=True)
        # fig_hne_co = add_image_to_figure(fig_hne_co, config.im_small_crop_annotation_tab)
        fig_hne_co = add_click_grid(fig_hne_co, config.im_small_crop_annotation_tab.shape[:2])
        fig_hne_co = add_points_trace(fig_hne_co, x_list_to_hne_image, y_list_to_hne_image, index_list_to_hne_image)

        fig_hne_co.update_layout(
            autosize=True,
            margin={"l": 0, "r": 0, "t": 0, "b": 0},
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            showlegend=False,
            # dragmode='select',
            # clickmode='event+select'
        )
        
        padded_rows_co = 1000
        padded_columns_co = 1600

        append_count = 0
        for i_index in metal_dropdown_co:
            i = config.metal_data['metals'][i_index]
            if append_count == 0:
                original_metal_image_co = i
            else:
                original_metal_image_co += i
            append_count += 1
        padded_metal_image_co = original_metal_image_co
        
        padded_metal_image_co[padded_metal_image_co <= 0] = 0.000001
        padded_metal_image_co = np.nan_to_num(padded_metal_image_co, nan=0.000001)
        padded_metal_image_co = np.log1p(padded_metal_image_co)
        
        # Create mask using the provided threshold
        mask = padded_metal_image_co > threshold
        
        # Apply histogram equalization with mask
        padded_metal_image_eq = exposure.equalize_hist(padded_metal_image_co, mask=mask)
        
        if vmin and vmax:
            c_norm = Colors_Normalize(vmin=np.percentile(padded_metal_image_eq, vmin), 
                                      vmax=np.percentile(padded_metal_image_eq, vmax), clip=True)
            padded_metal_image_normalized_co = c_norm(padded_metal_image_eq)
        else:
            padded_metal_image_normalized_co = padded_metal_image_eq
        
        cmap_jet = plt.cm.get_cmap(metal_colormap_co)
        padded_metal_image_rgb_co = cmap_jet(padded_metal_image_normalized_co)
        
        padded_metal_image_rgb_co = padded_metal_image_rgb_co[:, :, :3]
        target_color_to_white_co = padded_metal_image_rgb_co[0, 0, :]
        replacement_color_to_white_co = np.array([1, 1, 1])
        mask_to_white_co = np.all(padded_metal_image_rgb_co == target_color_to_white_co, axis=-1)
        padded_metal_image_rgb_co[mask_to_white_co] = replacement_color_to_white_co

        fig_metal_co = setup_base_figure((padded_metal_image_rgb_co * 255).astype(np.uint8), add_clickmode=True)
        fig_metal_co = add_click_grid(fig_metal_co, padded_metal_image_rgb_co.shape[:2])
        fig_metal_co = add_points_trace(fig_metal_co, x_list_to_metal_image, y_list_to_metal_image, index_list_to_metal_image)


        fig_metal_co.update_layout(
            autosize=True,
            margin={"l": 0, "r": 0, "t": 0, "b": 0},
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            showlegend=False,
            # dragmode='select',
            # clickmode='event+select'
        )
        fig_hne_co['data'][1].update(x=x_list_to_hne_image)
        fig_hne_co['data'][1].update(y=y_list_to_hne_image)
        fig_hne_co['data'][1].update(text=index_list_to_hne_image)
        fig_metal_co['data'][1].update(x=x_list_to_metal_image)
        fig_metal_co['data'][1].update(y=y_list_to_metal_image)
        fig_metal_co['data'][1].update(text=index_list_to_metal_image)
    return fig_hne_co, fig_metal_co

def update_two_image_and_table(config, clickData_hne, clickData_metal, table_data):
    if (not clickData_hne) and (not clickData_metal):
        update_df = pd.DataFrame({'index': [], 
                           'hne x': [], 'hne y': [], 
                        'metals x': [], 'metals y': [], })
        xy_coords_table = dash_table.DataTable(
                                        id='datatable_coord_co',
                                        columns=[
                                            {"name": i, "id": i,} for i in update_df.columns
                                        ],
                                        data=update_df.to_dict('records'),
                                        editable=True, row_deletable=True,)
        return xy_coords_table
    
    index_list = []
    hne_x = []
    hne_y = []
    metal_x = []
    metal_y = []
    index_count = 1
    
    for one_dict in table_data:
        index_list.append(index_count)
        hne_x.append(one_dict['hne x'])
        hne_y.append(one_dict['hne y'])
        metal_x.append(one_dict['metals x'])
        metal_y.append(one_dict['metals y'])
        index_count += 1
    if clickData_hne and clickData_hne != config.last_clickData_hne:
        config.last_clickData_hne = clickData_hne
        points = clickData_hne.get('points')[0]
        x_hne_point = points.get('x')
        y_hne_point = points.get('y')
        
        hne_point_exist = 0
        for one_dict in table_data:
            if (one_dict['hne x'] == x_hne_point) and (one_dict['hne y'] == y_hne_point):
                hne_point_exist = 1
        if hne_point_exist == 0:
            if ('-' in hne_x) or ('-' in hne_y):
                for one_index in range(len(index_list)):
                    if hne_x[one_index] == '-' or hne_y[one_index] == '-':
                        hne_x[one_index] = x_hne_point
                        hne_y[one_index] = y_hne_point
                        break
            else:
                if len(index_list) == 0:
                    index_list.append(1)
                else:
                    index_list.append(index_list[-1]+1)
                hne_x.append(x_hne_point)
                hne_y.append(y_hne_point)
                metal_x.append('-')
                metal_y.append('-')
            update_df = pd.DataFrame({'index': index_list, 
                           'hne x': hne_x, 'hne y': hne_y, 
                        'metals x': metal_x, 'metals y': metal_y, })
            xy_coords_table = dash_table.DataTable(
                                            id='datatable_coord_co',
                                            columns=[
                                                {"name": i, "id": i,} for i in update_df.columns
                                            ],
                                            data=update_df.to_dict('records'),
                                            editable=True, row_deletable=True,)
            return xy_coords_table
        else:
            points = clickData_metal.get('points')[0]
            x_metal_point = points.get('x')
            y_metal_point = points.get('y')

            metal_point_exist = 0
            for one_dict in table_data:
                if (one_dict['metals x'] == x_metal_point) and (one_dict['metals y'] == y_metal_point):
                    metal_point_exist = 1

            if metal_point_exist == 0:
                if ('-' in metal_x) or ('-' in metal_y):
                    for one_index in range(len(index_list)):
                        if metal_x[one_index] == '-' or metal_y[one_index] == '-':
                            metal_x[one_index] = x_metal_point
                            metal_y[one_index] = y_metal_point
                            break
                else:
                    if len(index_list) == 0:
                        index_list.append(1)
                    else:
                        index_list.append(index_list[-1]+1)
                    hne_x.append('-')
                    hne_y.append('-')
                    metal_x.append(x_metal_point)
                    metal_y.append(y_metal_point)

                update_df = pd.DataFrame({'index': index_list, 
                               'hne x': hne_x, 'hne y': hne_y, 
                            'metals x': metal_x, 'metals y': metal_y, })
                xy_coords_table = dash_table.DataTable(
                                                id='datatable_coord_co',
                                                columns=[
                                                    {"name": i, "id": i,} for i in update_df.columns
                                                ],
                                                data=update_df.to_dict('records'),
                                                editable=True, row_deletable=True,)
                return xy_coords_table
            else:
                update_df = pd.DataFrame({'index': index_list, 
                               'hne x': hne_x, 'hne y': hne_y, 
                            'metals x': metal_x, 'metals y': metal_y, })
                xy_coords_table = dash_table.DataTable(
                                                id='datatable_coord_co',
                                                columns=[
                                                    {"name": i, "id": i,} for i in update_df.columns
                                                ],
                                                data=update_df.to_dict('records'),
                                                editable=True, row_deletable=True,)
                return xy_coords_table
    else:
        points = clickData_metal.get('points')[0]
        x_metal_point = points.get('x')
        y_metal_point = points.get('y')

        metal_point_exist = 0
        for one_dict in table_data:
            if (one_dict['metals x'] == x_metal_point) and (one_dict['metals y'] == y_metal_point):
                metal_point_exist = 1

        if metal_point_exist == 0:
            if ('-' in metal_x) or ('-' in metal_y):
                for one_index in range(len(index_list)):
                    if metal_x[one_index] == '-' or metal_y[one_index] == '-':
                        metal_x[one_index] = x_metal_point
                        metal_y[one_index] = y_metal_point
                        break
            else:
                if len(index_list) == 0:
                    index_list.append(1)
                else:
                    index_list.append(index_list[-1]+1)
                hne_x.append('-')
                hne_y.append('-')
                metal_x.append(x_metal_point)
                metal_y.append(y_metal_point)

            update_df = pd.DataFrame({'index': index_list, 
                           'hne x': hne_x, 'hne y': hne_y, 
                        'metals x': metal_x, 'metals y': metal_y, })
            xy_coords_table = dash_table.DataTable(
                                            id='datatable_coord_co',
                                            columns=[
                                                {"name": i, "id": i,} for i in update_df.columns
                                            ],
                                            data=update_df.to_dict('records'),
                                            editable=True, row_deletable=True,)
            return xy_coords_table
        else:
            update_df = pd.DataFrame({'index': index_list, 
                           'hne x': hne_x, 'hne y': hne_y, 
                        'metals x': metal_x, 'metals y': metal_y, })
            xy_coords_table = dash_table.DataTable(
                                            id='datatable_coord_co',
                                            columns=[
                                                {"name": i, "id": i,} for i in update_df.columns
                                            ],
                                            data=update_df.to_dict('records'),
                                            editable=True, row_deletable=True,)
            return xy_coords_table

# @pysnooper.snoop()
def show_coregistered_images(config,n_clicks, table_data, old_selected_rows, threshold):
    if n_clicks > 0:
        slide_x, slide_y, metals_x, metals_y = [], [], [], []
        for one_index in range(len(table_data)):
            if table_data[one_index]['hne x'] != '-' and table_data[one_index]['hne y'] != '-' and table_data[one_index]['metals x'] != '-' and table_data[one_index]['metals y'] != '-':
                slide_x.append(table_data[one_index]['hne x'])
                slide_y.append(table_data[one_index]['hne y'])
                metals_x.append(table_data[one_index]['metals x'])
                metals_y.append(table_data[one_index]['metals y'])
        
        df_co = {'image': [], 'element': [],}
        for one_metal in list(config.metal_data['metals'].keys()):
            df_co['image'].append(config.metal_data['metals'][one_metal])
            df_co['element'].append(one_metal)
        dfs_new_df_co = pd.DataFrame(df_co)
        metals_im_gray=dfs_new_df_co['image'].mean(0)
        warped_metals_dict = warp_metals_new(slide_x, slide_y, metals_x, metals_y, dfs_new_df_co, metals_im_gray.shape, config.hne_shape)
        config.warped_metals=warped_metals_dict.pop('warped_metals')
        homo=warped_metals_dict.pop("homo")
        padded_metal_image_co = config.warped_metals['All']
        cmap_jet = plt.cm.get_cmap('jet')
        padded_metal_image_co[padded_metal_image_co <= 0] = 0.000001
        padded_metal_image_co = np.nan_to_num(padded_metal_image_co, nan=0.000001)
        padded_metal_image_co = np.log(padded_metal_image_co)+10
        padded_metal_image_normalized_co = (padded_metal_image_co - padded_metal_image_co.min()) / (padded_metal_image_co.max() - padded_metal_image_co.min())
        padded_metal_image_rgb_co = cmap_jet(padded_metal_image_normalized_co)
        padded_metal_image_rgb_co = padded_metal_image_rgb_co[:, :, :3]
        target_color_to_white_co = padded_metal_image_rgb_co[0, 0, :]
        replacement_color_to_white_co = np.array([1, 1, 1])
        mask_to_white_co = np.all(padded_metal_image_rgb_co == target_color_to_white_co, axis=-1)
        padded_metal_image_rgb_co[mask_to_white_co] = replacement_color_to_white_co
        hne_after_coregister = html.Img(src=array_to_data_url(config.im_small_crop_co), style={'width': '50%'})
        metal_after_coregister = html.Img(src=array_to_data_url((padded_metal_image_rgb_co * 255).astype(np.uint8)), style={'width': '50%'})

        im_small_gray = cv2.cvtColor(config.im_small_crop_co[:dfs_new_df_co['image'][0].shape[0], 
                                                      :dfs_new_df_co['image'][0].shape[1], 
                                                      :], cv2.COLOR_RGB2GRAY)
        save_metal_data = {'metals': config.warped_metals,'homo':homo}

        project_df_list, file_name_df_list, file_type_list = [], [], []
        all_files_list_first = os.listdir('./data/')
        all_files_list = []
        for one_file in all_files_list_first:
            if one_file == '.DS_Store':
                continue
            else:
                all_files_list.append(one_file)
        for one_file in all_files_list:
            data_files_under_list = os.listdir('./data/'+one_file)
            for one_data in data_files_under_list:
                if one_data == '.DS_Store':
                    continue
                if 'small' in one_data:
                    continue
                if os.path.splitext(one_data)[1] in config.file_extensions:
                    file_type_list.append('WSI')
                elif os.path.splitext(one_data)[1]==".pkl":
                    file_type_list.append('Metals')
                elif os.path.splitext(one_data)[1]==".zarr":
                    file_type_list.append("Exported Metals")
                elif os.path.splitext(one_data)[1] in ['.xml',".json",".geojson"]:
                    file_type_list.append('Annotations')
                project_df_list.append(one_file)
                file_name_df_list.append(one_data)

        two_files_name_list = []
        for one_index in old_selected_rows:
            two_files_name_list.append(project_df_list[one_index]+file_name_df_list[one_index])

        with open('./data/'+config.selected_project+'/coregistered_metals.pkl', 'wb') as fp:
            pickle.dump(save_metal_data, fp)

        files_df,files_df_columns, new_selected_rows = generate_files_df_records(config, old_selected_rows, return_selected_rows=True)

        display_files_table = dash_table.DataTable(
                                    id='files_table',
                                    selected_rows = new_selected_rows,
                                    columns=[
                                        {"name": i, "id": i,} for i in files_df_columns
                                    ],
                                    data=files_df,
                                    row_selectable='multi')
        

        return [hne_after_coregister, metal_after_coregister], [display_files_table]#files_df.to_dict('records')
    else:
        raise PreventUpdate

def show_coregistered_images(config, n_clicks, table_data, old_selected_rows, metal_colormap_co, metal_dropdown_co, vmin_vmax_input_co, threshold):
    if n_clicks > 0:
        # Extract coordinates from table data
        slide_x, slide_y, metals_x, metals_y = [], [], [], []
        for one_index in range(len(table_data)):
            if table_data[one_index]['hne x'] != '-' and table_data[one_index]['hne y'] != '-' and table_data[one_index]['metals x'] != '-' and table_data[one_index]['metals y'] != '-':
                slide_x.append(table_data[one_index]['hne x'])
                slide_y.append(table_data[one_index]['hne y'])
                metals_x.append(table_data[one_index]['metals x'])
                metals_y.append(table_data[one_index]['metals y'])
        
        # Process metal data
        df_co = {'image': [], 'element': []}
        for one_metal in list(config.metal_data['metals'].keys()):
            df_co['image'].append(config.metal_data['metals'][one_metal])
            df_co['element'].append(one_metal)
        dfs_new_df_co = pd.DataFrame(df_co)
        metals_im_gray = dfs_new_df_co['image'].mean(0)
        
        # Warp metals
        warped_metals_dict = warp_metals_new(slide_x, slide_y, metals_x, metals_y, dfs_new_df_co, metals_im_gray.shape, config.hne_shape)
        config.warped_metals = warped_metals_dict.pop('warped_metals')
        homo = warped_metals_dict.pop("homo")

        # Process metal image
        if "All" in metal_dropdown_co:
            padded_metal_image_co = config.warped_metals['All']
        elif len(metal_dropdown_co) == 1:
            padded_metal_image_co = config.warped_metals[metal_dropdown_co[0]]
        else:
            padded_metal_image_co = np.sum([config.warped_metals[i] for i in metal_dropdown_co], axis=0)

        nan_mask = np.isnan(padded_metal_image_co)
        # Apply transformations
        padded_metal_image_co[padded_metal_image_co <= 0] = 0.000001
        padded_metal_image_co = np.nan_to_num(padded_metal_image_co, nan=0.000001)
        padded_metal_image_co = np.log1p(padded_metal_image_co)
        
        # Create mask using the provided threshold
        mask = padded_metal_image_co > threshold
        
        # Apply histogram equalization with mask
        padded_metal_image_eq = exposure.equalize_hist(padded_metal_image_co, mask=mask)
        
        vmin, vmax = vmin_vmax_input_co
        if vmin and vmax:
            c_norm = Colors_Normalize(vmin=np.percentile(padded_metal_image_eq, vmin), 
                                    vmax=np.percentile(padded_metal_image_eq, vmax), 
                                    clip=True)
            padded_metal_image_normalized_co = c_norm(padded_metal_image_eq)
        else:
            padded_metal_image_normalized_co = padded_metal_image_eq

        # Create RGB image
        cmap_jet = plt.cm.get_cmap(metal_colormap_co)
        padded_metal_image_rgb_co = cmap_jet(padded_metal_image_normalized_co)
        padded_metal_image_rgb_co = padded_metal_image_rgb_co[:, :, :3]
        padded_metal_image_rgb_co[nan_mask] = np.array([1,1,1])
        hne_after_coregister = html.Img(src=array_to_data_url(config.im_small_crop_co), style={'width': '50%'})
        metal_after_coregister = html.Img(src=array_to_data_url((padded_metal_image_rgb_co * 255).astype(np.uint8)), style={'width': '50%'})

        # # Set background to white
        # target_color_to_white_co = padded_metal_image_rgb_co[0, 0, :]
        # mask_to_white_co = np.all(padded_metal_image_rgb_co == target_color_to_white_co, axis=-1)
        # padded_metal_image_rgb_co[mask_to_white_co] = [1, 1, 1]

        # # Create figures
        # fig_hne_co = setup_base_figure(config.im_small_crop_co, add_clickmode=True)
        # fig_hne_co = add_click_grid(fig_hne_co, config.im_small_crop_co.shape[:2])
        
        # fig_metal_co = setup_base_figure((padded_metal_image_rgb_co * 255).astype(np.uint8), add_clickmode=True)
        # fig_metal_co = add_click_grid(fig_metal_co, padded_metal_image_rgb_co.shape[:2])

        # # Update layouts
        # for fig in [fig_hne_co, fig_metal_co]:
        #     fig.update_layout(
        #         autosize=True,
        #         margin={"l": 0, "r": 0, "t": 0, "b": 0},
        #         paper_bgcolor='rgba(0,0,0,0)',
        #         plot_bgcolor='rgba(0,0,0,0)',
        #         showlegend=False,
        #     )

        # Save warped metals
        save_metal_data = {'metals': config.warped_metals, 'homo': homo}
    
        project_df_list, file_name_df_list, file_type_list = [], [], []
        all_files_list_first = os.listdir('./data/')
        all_files_list = []
        for one_file in all_files_list_first:
            if one_file == '.DS_Store':
                continue
            else:
                all_files_list.append(one_file)
        for one_file in all_files_list:
            data_files_under_list = os.listdir('./data/'+one_file)
            for one_data in data_files_under_list:
                if one_data == '.DS_Store':
                    continue
                if 'small' in one_data:
                    continue
                if os.path.splitext(one_data)[1] in config.file_extensions:
                    file_type_list.append('WSI')
                elif os.path.splitext(one_data)[1]==".pkl":
                    file_type_list.append('Metals')
                elif os.path.splitext(one_data)[1]==".zarr":
                    file_type_list.append("Exported Metals")
                elif os.path.splitext(one_data)[1] in ['.xml',".json",".geojson"]:
                    file_type_list.append('Annotations')
                project_df_list.append(one_file)
                file_name_df_list.append(one_data)

        two_files_name_list = []
        for one_index in old_selected_rows:
            two_files_name_list.append(project_df_list[one_index]+file_name_df_list[one_index])

        with open('./data/'+config.selected_project+'/coregistered_metals.pkl', 'wb') as fp:
            pickle.dump(save_metal_data, fp)

        files_df,files_df_columns,new_selected_rows = generate_files_df_records(config, old_selected_rows)

        display_files_table = dash_table.DataTable(
                                    id='files_table',
                                    selected_rows = new_selected_rows,
                                    columns=[
                                        {"name": i, "id": i,} for i in files_df.columns
                                    ],
                                    data=files_df,
                                    row_selectable='multi')

        return [hne_after_coregister, metal_after_coregister], [display_files_table]
    else:
        raise PreventUpdate 