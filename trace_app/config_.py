import os, numpy as np
import plotly.express as px
import pandas as pd
import plotly.graph_objects as go
from dash import dash_table
import matplotlib.pyplot as plt
from dash_canvas.utils import array_to_data_url
from .image_processing import output_blank_hne
from .file_utils import return_upload_files, generate_files_df_records

class Config:
    DEBUG = bool(int(os.environ.get('DEBUG', '0')))

LOW_MEMORY=bool(int(os.environ.get('LOW_MEMORY', '0')))

file_extensions = [".tif", ".tiff", ".tif", ".tiff", ".ome.tif", ".ome.tiff", ".tif", ".tiff", ".dng", ".zif", ".stk", ".lsm", ".tif", ".tiff", ".tif", ".tiff", ".tif", ".tiff", ".sgi", ".rgb", ".rgba", ".bw", ".img", ".oif", ".oib", ".sis", ".tif", ".tiff", ".gel", ".svs", ".scn", ".bif", ".qptiff", ".qpi", ".pki", ".ndpi", ".avs", ".tif", ".tiff"]
file_extensions = list(set(file_extensions))

im_small_crop_annotation_tab = np.full((871, 1499, 3), 255, dtype=np.uint8)
im_small_crop_co = np.full((871, 1499, 3), 255, dtype=np.uint8)
metal_data = {'metals': {'All': np.full((871, 1499), 0.01), 
                         }}

history_co_click = 0
file_name_history_1 = ''
file_name_history_2 = ''

history_selected_metal = ''
history_colormap = ''
history_vmin = ''
history_vmax = ''
history_selected_rows = []
history_co_transport_n_clicks = 99999
compression_annotation_tab = 1
scale_factor_annotation = 0.3

compression_value_again = 0

last_clickData_hne = {}

last_select_tiff_file = ''
last_select_pkl_file = ''
selected_project = ''

files_to_upload_path = '/upload_dir/'
num_workers=os.cpu_count()-1

project_df_list, file_name_df_list, file_type_list = [], [], []

if not os.path.exists('./data/'):
    os.makedirs('./data/',exist_ok=True)
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
        if os.path.splitext(one_data)[1] in file_extensions:
            file_type_list.append('WSI')
        elif os.path.splitext(one_data)[1]==".pkl":
            file_type_list.append('Metals')
        elif os.path.splitext(one_data)[1]==".zarr":
            file_type_list.append("Exported Metals")
        elif os.path.splitext(one_data)[1] in ['.xml',".json",".geojson"]:
            file_type_list.append('Annotations')
        else:
            file_type_list.append('Unknown')
        project_df_list.append(one_file)
        file_name_df_list.append(one_data)


xy_index = []

co_hne_x_list = []
co_hne_y_list = []

fig_hne_co = px.imshow(im_small_crop_annotation_tab)
fig_hne_co.add_scatter(
    x=[],
    y=[],
    mode='markers',
    marker_color='black',
    marker_size=5,
)
fig_hne_co.update_layout(
    template='plotly_dark',
    plot_bgcolor='rgba(0, 0, 0, 0)',
    paper_bgcolor='rgba(0, 0, 0, 0)',
    width=100,#700,
    height=100,#500,
    margin={
        'l': 0,
        'r': 0,
        't': 0,
        'b': 0,
    }
)
fig_hne_co.update_xaxes(visible=False,)
fig_hne_co.update_yaxes(visible=False,)
fig_hne_co.update_coloraxes(showscale=False)

co_metal_x_list = []
co_metal_y_list = []

padded_rows_co = 1000
padded_columns_co = 1600

original_metal_image_co = metal_data['metals']['All']
padded_metal_image_co = original_metal_image_co

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

fig_metal_co = px.imshow(padded_metal_image_rgb_co)
fig_metal_co.add_scatter(
    x=[],
    y=[],
    mode='markers+text',
    marker_color='black',
    marker_size=5,
    textposition='top center',
    textfont=dict(color='black'),
)
fig_metal_co.update_layout(
    template='plotly_dark',
    plot_bgcolor='rgba(0, 0, 0, 0)',
    paper_bgcolor='rgba(0, 0, 0, 0)',
    width=700,#700,
    height=500,#500,
    margin={
        'l': 0,
        'r': 0,
        't': 0,
        'b': 0,
    }
)
fig_metal_co.update_xaxes(visible=False,)
fig_metal_co.update_yaxes(visible=False,)
fig_metal_co.update_coloraxes(showscale=False)


original_df_co = pd.DataFrame({'index': xy_index, 
                               'hne x': co_hne_x_list, 'hne y': co_hne_y_list, 
                            'metals x': co_metal_x_list, 'metals y': co_metal_y_list, })

files_df = pd.DataFrame({'Project': project_df_list, 
                         'File Name': file_name_df_list, 
                         'File Type': file_type_list,})
display_files_table = dash_table.DataTable(
                                    id='files_table',
                                    columns=[
                                        {"name": i, "id": i,} for i in files_df.columns
                                    ],
                                    data=files_df.to_dict('records'),
                                    row_selectable='multi')

upload_files_df_data,cols=return_upload_files(return_cols=True,files_to_upload_path=files_to_upload_path,exts=file_extensions+[".pkl"])

upload_files_df_data_xlsx,cols_xlsx=return_upload_files(return_cols=True,colname="Elemental Image Excel File Name",files_to_upload_path=files_to_upload_path,exts=[".xlsx"])

df_co = {'image': [], 'element': [],}
for one_metal in list(metal_data['metals'].keys()):
    df_co['image'].append(metal_data['metals'][one_metal])
    df_co['element'].append(one_metal)
dfs_new_df_co = pd.DataFrame(df_co)

white_image_co = np.ones((100, 100, 3), dtype=np.uint8) * 255
white_fig_co = px.imshow(white_image_co, binary_string=True)
white_fig_co.update_xaxes(visible=False,)
white_fig_co.update_yaxes(visible=False,)
white_fig_co.update_coloraxes(showscale=False)
white_fig_co.update_traces(
   hovertemplate=None,
   hoverinfo='skip'
)

xy_coords_table = dash_table.DataTable(
        id='datatable_coord_co',
        columns=[
            {"name": i, "id": i,} for i in original_df_co.columns
        ],
        data=original_df_co.to_dict('records'),
        editable=True, row_deletable=True,)

markdown_text_title = '''
'''

all_color_list = px.colors.qualitative.Dark24
result_df = pd.DataFrame({'type': [], 'metal': [], 
                              'mean': [], 'std': []})
newest_result_df = pd.DataFrame({'type': [], 'metal': [], 
                              'mean': [], 'std': []})
box_df = pd.DataFrame({'type': [], 'metal': [], 
                              'value': []})

all_relayout_data = {'shapes': [], 'dragmode': ''}

type_to_color_dict = {'immune': all_color_list[0], 'tumor': all_color_list[1]}
color_to_type_dict = {all_color_list[0]: 'immune', all_color_list[1]: 'tumor'}


image_no_axis_layout = go.Layout(
    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
)

temp_folder_list = os.listdir('./data/')
try:
    temp_folder_list.remove('.DS_Store')
except:
    pass


blank_figure_hne = output_blank_hne(im_small_crop_annotation_tab)

padded_rows = 1000
padded_columns = 1600

original_metal_image = metal_data['metals']['All']
rows_padding = max(0, padded_rows - original_metal_image.shape[0])
columns_padding = max(0, padded_columns - original_metal_image.shape[1])
top_pad = rows_padding // 2
bottom_pad = rows_padding - top_pad
left_pad = columns_padding // 2
right_pad = columns_padding - left_pad
padded_metal_image = original_metal_image

cmap_jet = plt.cm.get_cmap('jet')
padded_metal_image[padded_metal_image <= 0] = 0.000001
padded_metal_image = np.nan_to_num(padded_metal_image, nan=0.000001)
padded_metal_image = np.log(padded_metal_image)+10
padded_metal_image_normalized = (padded_metal_image - padded_metal_image.min()) / (padded_metal_image.max() - padded_metal_image.min())
padded_metal_image_rgb = cmap_jet(padded_metal_image_normalized)
padded_metal_image_rgb = padded_metal_image_rgb[:, :, :3]


blank_figure = go.Figure()

blank_img_width = padded_metal_image.shape[1]
blank_img_height = padded_metal_image.shape[0]
scale_factor = 0.3

blank_figure.add_trace(
    go.Scatter(
        x=[0, blank_img_width * scale_factor],
        y=[0, blank_img_height * scale_factor],
        mode="markers",
        marker_opacity=0
    )
)

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

blank_figure.update_layout(
    width=blank_img_width * scale_factor,
    height=blank_img_height * scale_factor,
    margin={"l": 0, "r": 0, "t": 0, "b": 0},
)


all_annot_data = dict()

config = Config()

# File and project related
config.file_extensions = file_extensions
config.file_name_history_1 = file_name_history_1
config.file_name_history_2 = file_name_history_2
config.last_select_tiff_file = last_select_tiff_file
config.last_select_pkl_file = last_select_pkl_file
config.select_pkl_file = None
config.selected_project = selected_project
config.temp_folder_list = temp_folder_list
config.files_to_upload_path = files_to_upload_path

# Data structures
config.metal_data = metal_data
config.original_metal_data = None
config.metal_data_preprocess = None
config.warped_metals = None
config.all_annot_data = all_annot_data
config.box_df = box_df
config.newest_result_df = newest_result_df
config.result_df = result_df

# UI elements
config.markdown_text_title = markdown_text_title
config.type_to_color_dict = type_to_color_dict
config.color_to_type_dict = color_to_type_dict
config.all_color_list = all_color_list

# Layout and display
config.all_relayout_data = all_relayout_data
config.im_small_crop_annotation_tab = im_small_crop_annotation_tab
config.im_small_crop_co = im_small_crop_co
config.hne_shape = None
config.blank_figure = blank_figure
config.blank_figure_hne = blank_figure_hne
config.fig_hne_co = fig_hne_co
config.fig_metal_co = fig_metal_co
config.white_fig_co = white_fig_co

# Tables
config.xy_coords_table = xy_coords_table
config.display_files_table = display_files_table
config.upload_files_df_data = upload_files_df_data
config.upload_files_df_data_xlsx = upload_files_df_data_xlsx
config.cols = cols
config.cols_xlsx = cols_xlsx

# Image processing
config.compression_annotation_tab = compression_annotation_tab
config.compression_value_again = compression_value_again
config.scale_factor_annotation = scale_factor_annotation
config.scale_factor = scale_factor
config.padded_rows = padded_rows
config.padded_columns = padded_columns
config.original_metal_image = original_metal_image
config.blank_img_width = blank_img_width
config.blank_img_height = blank_img_height

# State variables
config.history_selected_metal = history_selected_metal
config.history_colormap = history_colormap
config.history_vmin = history_vmin
config.history_vmax = history_vmax
config.history_co_transport_n_clicks = history_co_transport_n_clicks
config.last_clickData_hne = last_clickData_hne

# Settings
config.LOW_MEMORY = LOW_MEMORY
config.num_workers = num_workers

# elements
config.elements = "H, He, Li, Be, B, C, N, O, F, Ne, Na, Mg, Al, Si, P, S, Cl, Ar, K, Ca, Sc, Ti, V, Cr, Mn, Fe, Co, Ni, Cu, Zn, Ga, Ge, As, Se, Br, Kr, Rb, Sr, Y, Zr, Nb, Mo, Tc, Ru, Rh, Pd, Ag, Cd, In, Sn, Sb, Te, I, Xe, Cs, Ba, La, Ce, Pr, Nd, Pm, Sm, Eu, Gd, Tb, Dy, Ho, Er, Tm, Yb, Lu, Hf, Ta, W, Re, Os, Ir, Pt, Au, Hg, Tl, Pb, Bi, Po, At, Rn, Fr, Ra, Ac, Th, Pa, U, Np, Pu, Am, Cm, Bk, Cf, Es, Fm, Md, No, Lr, Rf, Db, Sg, Bh, Hs, Mt, Ds, Rg, Cn, Nh, Fl, Mc, Lv, Ts, Og"
config.elements = config.elements.split(", ")
