import os
import copy
import pickle
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, dcc, html, dash_table
from dash.dependencies import Input, Output, State, MATCH
from dash.exceptions import PreventUpdate
from dash_canvas.utils import array_to_data_url
from matplotlib.colors import Normalize as Colors_Normalize
from skimage import measure, morphology
import tifffile as tiff
import plotly.express as px
import dash_bootstrap_components as dbc
from dash import no_update
from dash import callback_context
from dash import dcc

# File Handling Functions
from .file_utils import (
    upload_file,
    upload_file_v2,
    update_file_dir,
    make_new_folder,
    display_selected_file,
    upload_xlsx_file
)

# Image Processing Functions
from .image_processing import (
    generate_tissue_mask_func,
    update_pre_metals,
    calculate_mask_threshold,
    change_two_images_and_clean_point_table,
    update_two_image_and_table,
    update_back_to_image,
    show_coregistered_images
)

# Data Processing Functions
from .data_processing import (
    update_data_table,
    update_table_callback,
    update_boxplot_dropdown
)

# Export Functions
from .export_utils import (
    export_on_click,
    export_raw_data_on_click
)

# Annotation Functions
from .annotation_utils import (
    add_annotation_type,
    update_annotation_callback
)

from .app_layout import all_tabs, config

app = Dash(__name__, 
           external_stylesheets=[
               dbc.themes.BOOTSTRAP,
               {
                   'href': 'https://use.fontawesome.com/releases/v5.8.1/css/all.css',
                   'rel': 'stylesheet',
                   'integrity': 'sha384-50oBUHEmvpQ+1lW4y57PTFmhCaXp0ML5d60M1M7uH2+nqUivzIebhndOJK28anvf',
                   'crossorigin': 'anonymous'
               }
           ],
        #    external_scripts=["https://code.jquery.com/jquery-3.6.0.min.js"]
          )

# Add custom CSS directly to the app
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            .btn {
                background-color: white !important;
                color: black !important;
                border: 1px solid #ccc !important;
            }
            .btn:hover {
                background-color: #f8f9fa !important;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

app.layout = html.Div([dcc.Markdown(children=config.markdown_text_title),
                       all_tabs])



@app.callback(
    Output('metal_pre_mask', 'figure', allow_duplicate=True),
    Output('display_data_table_container', 'children', allow_duplicate=True),
    Input('start_generate_mask_button', 'n_clicks'),
    State('calculated_threshold', 'data'),  # Changed from 'pre_threshold_input' to 'pre_threshold_slider'
    State('files_table', 'selected_rows'),
    State('metal_dropdown_pre', 'value'),
    State('metal_colormap_pre', 'value'),
    prevent_initial_call=True,
)
def generate_tissue_mask_func_(n_clicks, threshold, old_selected_rows, metal_dropdown_pre, metal_colormap_pre):
    global config
    return generate_tissue_mask_func(config, n_clicks, threshold, old_selected_rows, metal_dropdown_pre, metal_colormap_pre)

@app.callback(
    Output('metal_pre', 'figure'),
    Output('metal_pre_mask', 'figure'),
    Input('metal_colormap_pre', 'value'),
    Input('metal_dropdown_pre', 'value'),
    Input('vmin_vmax_input_pre', 'value'),
    Input('pre_threshold_slider', 'value'),  # Add this new input
    State('metal_pre', 'relayoutData'),
)               
def update_pre_metals_(colormap, selected_metal, vmin_vmax, threshold, all_relayout_data):
    global config
    return update_pre_metals(config, colormap, selected_metal, vmin_vmax, threshold, all_relayout_data)

@app.callback(
    Output('calculated_threshold', 'data'),
    Output('threshold_display', 'children'),
    Input('metal_pre', 'relayoutData'),
    State('metal_dropdown_pre', 'value'),
)
def calculate_mask_threshold_(relayout_data, selected_metal):
    global config
    threshold = calculate_mask_threshold(config, selected_metal, relayout_data)
    return threshold,f'Threshold: {threshold}'

@app.callback(
    Output('upload_data_folder_dropdown', 'options'),
    Input('make_new_folder_button', 'n_clicks'),
    State('new_folder_input', 'value'),
    State('upload_data_folder_dropdown', 'options'),
)
def make_new_folder_(n_clicks, new_folder_input, upload_data_folder_dropdown):
    global config
    return make_new_folder(config, n_clicks, new_folder_input, upload_data_folder_dropdown)

@app.callback(
    Output('files_table', 'data', allow_duplicate=True),
    Input('upload_multiple_xlsx_files', 'contents'),
    Input('upload_multiple_xlsx_files', 'filename'),
    Input('upload_multiple_xlsx_files', 'last_modified'),
    State('upload_data_folder_dropdown', 'value'),
    prevent_initial_call=True,
)
def upload_file_(list_of_contents, list_of_names, list_of_dates, upload_data_folder_dropdown):
    global config
    return upload_file(config, list_of_contents, list_of_names, list_of_dates, upload_data_folder_dropdown)

@app.callback(
    Output('files_table', 'data', allow_duplicate=True),
    Input('upload_elemental_image_button', 'n_clicks'),
    State('upload_elemental_image_table', 'selected_rows'),
    State('upload_data_folder_dropdown', 'value'),
    prevent_initial_call=True,
)
def upload_xlsx_file_(n_clicks, selected_row, upload_data_folder_dropdown):
    global config
    print(upload_data_folder_dropdown,selected_row)
    return upload_xlsx_file(config, selected_row[0], upload_data_folder_dropdown)

@app.callback(
    Output('upload_files_table', 'data'),
    Input('update_button', 'n_clicks')
)
def update_file_dir_(_):
    global config
    config.upload_files_df_data=update_file_dir(config)
    return config.upload_files_df_data
    
@app.callback(
    [Output('files_table', 'data'),
     Output('upload_files_table', 'selected_rows')],
    Input('upload_button', 'n_clicks'),
    State('upload_data_folder_dropdown', 'value'),
    State('upload_files_table', 'selected_rows')
)

def upload_file_v2_(n_clicks, upload_data_folder_dropdown, selected_rows):
    global config
    return upload_file_v2(config, n_clicks, upload_data_folder_dropdown, selected_rows)

@app.callback(
    Output('output_selected_file', 'children'),
    Input('load_data_button', 'n_clicks'),
    State('files_table', 'selected_rows'),
)
def display_selected_file_(n_clicks, selected_rows):
    global config
    return display_selected_file(config, n_clicks, selected_rows)

@app.callback(
    Output('two_image_container', 'children', allow_duplicate=True),
    Output('check_border_image', 'figure', allow_duplicate=True),
    Output('table_container_co', 'children', allow_duplicate=True),
    Output('boxplot_dropdown', 'options', allow_duplicate=True), 
    Output('boxplot_dropdown', 'value', allow_duplicate=True), 
    Output('annotate_metal_image', 'figure', allow_duplicate=True),
    Output('blank_hne_image', 'figure', allow_duplicate=True),
    Output('graph_hne_co', 'figure', allow_duplicate=True),  

    Output('metal_dropdown_annotation', 'value', allow_duplicate=True),
    Output('metal_dropdown_annotation', 'options', allow_duplicate=True),
    Output('annotation_colormap', 'value', allow_duplicate=True),
    Output('vmin_vmax_input', 'value', allow_duplicate=True), 

    Output('metal_dropdown_co', 'value', allow_duplicate=True),
    Output('metal_dropdown_co', 'options', allow_duplicate=True),
    Output('metal_colormap_co', 'value', allow_duplicate=True),
    Output('vmin_vmax_input_co', 'value', allow_duplicate=True), 

    Output('metal_dropdown_pre', 'value', allow_duplicate=True),
    Output('metal_dropdown_pre', 'options', allow_duplicate=True),
    Output('metal_colormap_pre', 'value', allow_duplicate=True),
    Output('vmin_vmax_input_pre', 'value', allow_duplicate=True), 

    Output('metal_pre', 'figure', allow_duplicate=True),
    Output('metal_pre_mask', 'figure', allow_duplicate=True),

    Output('selected_elements', 'value', allow_duplicate=False),
    Output('selected_elements', 'options', allow_duplicate=False),

    Output('pre_threshold_slider', 'value', allow_duplicate=True),  # Add this new output
    Output('annotation_threshold_slider', 'value', allow_duplicate=True),  # Add this new output
    Output('co_threshold_slider', 'value', allow_duplicate=True),  # Add this new output

    Input('load_data_button', 'n_clicks'),
    State('files_table', 'selected_rows'),
    State('files_table', 'data'),
    prevent_initial_call=True,
)
def change_two_images_and_clean_point_table_(n_clicks, selected_rows, files_table_data):
    global config
    results = change_two_images_and_clean_point_table(config, n_clicks, selected_rows, files_table_data)
    # Add default threshold values (e.g., 90) to the returned results
    return results 


@app.callback(
    Output('type_dropdown_annotation', 'options'),
    Output('type_color_match_list', 'children'),
    Input('add_type_button', 'n_clicks'),
    Input('upload_annotation', 'contents'), 
    State('upload_data_folder_dropdown', 'value'),
    State('type_input', 'value'), 
    State('type_dropdown_annotation', 'options'),
    State('upload_annotation', 'filename'),
    State('upload_annotation', 'last_modified'),
)
def add_annotation_type_(n_clicks, 
                        contents, project_name,
                        type_input, previous_types, 
                        filename, last_modified):
    global config
    return add_annotation_type(config, n_clicks, contents, project_name, type_input, previous_types, filename, last_modified)


@app.callback(
    Output('annotate_metal_image', 'figure'),
    Output('blank_hne_image', 'figure'),
    Input('metal_dropdown_annotation', 'value'),
    Input('annotate_metal_image', 'relayoutData'),
    Input('type_dropdown_annotation', 'value'),
    Input('annotation_colormap', 'value'),
    Input('vmin_vmax_input', 'value'), 
    Input('start_co_button', 'n_clicks'),
    Input('load_data_button', 'n_clicks'),
    Input('upload_annotation', 'contents'),
    Input('annotation_threshold_slider', 'value'),  # Add this new input
    State('upload_annotation', 'filename'),
    State('upload_annotation', 'last_modified'),
)
def update_annotation_callback_(selected_metal, 
                               relayout_data,
                               annotation_type,
                               colormap,
                               vmin_vmax,
                               n_clicks,
                               n_clicks_load_data,
                               contents,
                               threshold,  # Add this new parameter
                               filename,
                               last_modified
                              ):
    global config
    return update_annotation_callback(config, selected_metal, relayout_data, annotation_type, colormap, vmin_vmax, n_clicks, n_clicks_load_data, contents, filename, last_modified, threshold)

@app.callback(
    Output('boxplot_dropdown', 'value'), 
    Output('boxplot_dropdown', 'options'), 
    Input('boxplot_metal_type', 'checked'),
    State('type_dropdown_annotation', 'options'),
)
def update_boxplot_dropdown_(boxplot_metal_type, type_dropdown_annotation):
    global config
    return update_boxplot_dropdown(config, boxplot_metal_type, type_dropdown_annotation)

@app.callback(
    Output('table_container', 'children'),
    Output("result_boxplot", "figure"), 
    Input('update_table_button', 'n_clicks'),
    Input('boxplot_dropdown', 'value'), 
    Input('boxplot_log_scale', 'checked'),
    Input('boxplot_metal_type', 'checked'),
    Input('selected_elements', 'value'),
    State('annotate_metal_image', 'relayoutData'),
    State('table_container', 'children'),
)
def update_table_callback_(n_clicks, boxplot_dropdown, boxplot_log_scale, boxplot_metal_type, selected_elements, relayout_data, table_children):
    global config
    return update_table_callback(config, n_clicks, boxplot_dropdown, boxplot_log_scale, boxplot_metal_type, selected_elements, relayout_data, table_children)

@app.callback(
    Output("download_dataframe_csv", "data"),
    Input("export_data", "n_clicks"),
    State('table_container', 'children'),
    prevent_initial_call=True,
)
def export_on_click_(n_clicks, table_children):
    global config
    return export_on_click(config, n_clicks, table_children)

@app.callback(
        Output('files_table', 'data', allow_duplicate=True),
        Input("export_data_additional", "n_clicks"),
        State('upload_data_folder_dropdown', 'value'),
        prevent_initial_call=True
)
def export_raw_data_on_click_(_, project_name):
    global config
    return export_raw_data_on_click(config, _,project_name)


@app.callback(
    Output('table_container_co', 'children'),
    Input('graph_hne_co', 'clickData'),
    Input('graph_metal_co', 'clickData'),
    State('datatable_coord_co', 'data'),
)
def update_two_image_and_table_(clickData_hne, clickData_metal, table_data):
    global config
    print(clickData_hne, clickData_metal)
    return update_two_image_and_table(config, clickData_hne, clickData_metal, table_data)

@app.callback(
    Output('datatable_coord_co', 'data'),
    Input('datatable_coord_co', 'data_previous'),
    State('datatable_coord_co', 'data')
)
def update_data_table_(previous_data, current_data):
    global config
    return update_data_table(config, previous_data, current_data)


@app.callback(
    Output('graph_hne_co', 'figure'),
    Output('graph_metal_co', 'figure'),
    Input('datatable_coord_co', 'data'),
    Input('metal_colormap_co', 'value'),
    Input('metal_dropdown_co', 'value'),
    Input('vmin_vmax_input_co', 'value'),
    Input('co_threshold_slider', 'value'),  # Add this new input
)
def update_back_to_image_(table_data, metal_colormap_co, metal_dropdown_co, vmin_vmax_input_co, threshold): 
    global config  
    return update_back_to_image(config, table_data, metal_colormap_co, metal_dropdown_co, vmin_vmax_input_co, threshold)

@app.callback(
    Output('two_image_container', 'children'),
    Output('display_data_table_container', 'children', allow_duplicate=True),
    Input('start_co_button', 'n_clicks'),
    State('datatable_coord_co', 'data'),
    State('files_table', 'selected_rows'),
    State('metal_colormap_co', 'value'),
    State('metal_dropdown_co', 'value'),
    State('vmin_vmax_input_co', 'value'),
    State('co_threshold_slider', 'value'),  # Add this new input
    prevent_initial_call=True,
)
def show_coregistered_images_(n_clicks, table_data, old_selected_rows, metal_colormap_co, metal_dropdown_co, vmin_vmax_input_co, threshold_co):
    global config
    return show_coregistered_images(config, n_clicks, table_data, old_selected_rows, metal_colormap_co, metal_dropdown_co, vmin_vmax_input_co, threshold_co)
    # return show_coregistered_images(config, n_clicks, table_data, old_selected_rows, threshold)

# Add a new callback for the threshold slider
@app.callback(
    Output({'type': 'threshold-value', 'index': MATCH}, 'children'),
    Input({'type': 'threshold-slider', 'index': MATCH}, 'value')
)
def update_threshold_value(value):
    return f'Threshold: {value}'
