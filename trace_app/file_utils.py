import os
import glob
import json
import pickle
import shutil
import base64
import io
from functools import reduce
from collections import OrderedDict

import numpy as np
import pandas as pd
import geopandas
import cv2
import pyvips
import tifffile as tiff
import openslide
from dash import html
from dash.dcc import Upload
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
from dash_canvas.utils import array_to_data_url
from dask.diagnostics import ProgressBar
from xlsx2csv import Xlsx2csv
import dask
import re
import pysnooper

def generate_files_df_records(config, old_selected_rows=[]):
    project_df_list, file_name_df_list, file_type_list = [], [], []
    all_files_list_first = os.listdir(config.workdir_data_path)
    all_files_list = [f for f in all_files_list_first if f != '.DS_Store']
    
    for one_file in all_files_list:
        data_files_under_list = os.listdir(os.path.join(config.workdir_data_path, one_file))
        for one_data in data_files_under_list:
            if one_data == '.DS_Store' or 'small' in one_data:
                continue
            
            file_extension = os.path.splitext(one_data)[1]
            file_basename = os.path.basename(one_data)
            
            if file_extension in config.file_extensions:
                file_type = 'WSI'
            elif file_basename == "preprocess_metals.pkl":
                file_type = "Preprocessed Metals"
            elif file_basename == "coregistered_metals.pkl":
                file_type = "Coregistered Metals"
            elif file_basename == "exported_metals_annots.pkl":
                file_type = "Exported Metals"
            elif file_extension == ".pkl":
                file_type = 'Metals'
            elif file_extension == ".zarr":
                file_type = "Exported Metals"
            elif file_extension in ['.xml', ".json", ".geojson"]:
                file_type = 'Annotations'
            else:
                continue
            
            file_type_list.append(file_type)
            project_df_list.append(one_file)
            file_name_df_list.append(one_data)

    files_df = pd.DataFrame({
        'Project': project_df_list, 
        'File Name': file_name_df_list, 
        'File Type': file_type_list,
    })
    if old_selected_rows:
        two_files_name_list = [project_df_list[i] + file_name_df_list[i] for i in old_selected_rows]
        new_selected_rows = [i for i, (project, file) in enumerate(zip(project_df_list, file_name_df_list))
                             if project + file in two_files_name_list]
        return files_df.to_dict('records'), files_df.columns, new_selected_rows
    return files_df.to_dict('records'), files_df.columns

def return_upload_files(config=None, return_cols=False,colname="Image File Name",files_to_upload_path=None,exts=None):
    assert (config is not None) or (exts is not None), "Must provide config or exts"
    # print(files_to_upload_path,config.files_to_upload_path if config is not None else files_to_upload_path,exts,glob.glob(os.path.join(files_to_upload_path if files_to_upload_path is not None else config.files_to_upload_path,"*")))
    if exts is None:
        exts = config.file_extensions + [".pkl"]
    files_by_ext = {}
    for ext in exts:
        pattern = os.path.join(
            files_to_upload_path if files_to_upload_path is not None else config.files_to_upload_path,
            f"*{ext}"
        )
        files = glob.glob(pattern)
        files_by_ext[ext] = files
    all_files = list(reduce(lambda x, y: x + y, files_by_ext.values()))
    if len(all_files) > 0:
        col_values = np.vectorize(os.path.basename)(all_files)
    else:
        col_values = []
    upload_files_df = pd.DataFrame({colname: col_values})
    if not return_cols:
        return upload_files_df.to_dict("records")
    return upload_files_df.to_dict("records"),upload_files_df.columns

def parse_contents(config,filename):

    read_path = config.files_to_upload_path+str(filename)

    try:
        df = geopandas.read_file(read_path).explode()
    except:
        try:
            with open(read_path) as f:
                auto_annot=json.load(f)
                df = geopandas.GeoDataFrame.from_features(auto_annot).explode()
        except Exception as e:
            print(e)
            return html.Div([
                'There was an error processing this file.'
            ])
    return df

def read_thumbnail_openslide(path, scale=0.25):
    import openslide
    import numpy as np

    slide = openslide.OpenSlide(path)
    w, h = slide.dimensions

    tw = max(1, int(w * scale))
    th = max(1, int(h * scale))

    thumb = slide.get_thumbnail((tw, th))
    return (w, h), np.array(thumb)


def read_thumbnail_pyvips(path, scale=0.25):
    import pyvips
    import numpy as np
    import os

    # Memory safety settings
    os.environ.setdefault("VIPS_CONCURRENCY", "1")
    os.environ.setdefault("VIPS_DISC_THRESHOLD", "10m")
    pyvips.cache_set_max_mem(200 * 1024 * 1024)

    hdr = pyvips.Image.new_from_file(path, access="sequential")
    w, h = hdr.width, hdr.height

    tw = max(1, int(w * scale))
    im = pyvips.Image.thumbnail(path, tw)
    return (w, h), im.numpy()


def read_thumbnail_lowmem(path, filename):
    ext = os.path.splitext(filename)[1].lower()

    # Formats pyvips can handle well
    pyvips_exts = {'.svs', '.tif', '.tiff'}

    if ext in pyvips_exts:
        # Try pyvips first (faster & lower memory)
        try:
            return read_thumbnail_pyvips(path)
        except Exception:
            # fallback to OpenSlide
            return read_thumbnail_openslide(path)

    # All other formats → OpenSlide
    return read_thumbnail_openslide(path)

@pysnooper.snoop()
def upload_contents(config,filename, project_name):
    print('start function')
    path = os.path.join(config.files_to_upload_path, filename)
    ext = os.path.splitext(filename)[1].lower()
    os_path = path#("" if ext in ['.svs','.tif','.tiff'] else "openslide:")+

    if ext in config.file_extensions:
        if not config.LOW_MEMORY:
            image = tiff.imread(path)
            im_width,im_height = image.shape[1], image.shape[0]
        else:
            ext_lower = os.path.splitext(filename)[1].lower()
            if ext_lower in ['.svs', '.tif', '.tiff']:
                (im_width, im_height), image = read_thumbnail_lowmem(os_path, filename)
        com_1 = image.shape[0]/1500
        com_2 = image.shape[1]/1500
        compressed_image = cv2.resize(image,None,fx=1/max(com_1, com_2),fy=1/max(com_1, com_2))
        if config.LOW_MEMORY:
            com_1,com_2 = im_height/1500,im_width/1500

        if not os.path.exists('compression_value_dict.json'):
            if os.path.splitext(filename)[1] in config.file_extensions:
                compression_value_dict = {project_name+filename: 1/max(com_1, com_2),
                                          project_name+os.path.splitext(filename)[0]+'.tiff': 1/max(com_1, com_2),
                                          project_name+os.path.splitext(filename)[0]+'.svs': 1/max(com_1, com_2)}
            with open('compression_value_dict.json', 'w') as json_file:
                json.dump(compression_value_dict, json_file)
        else:
            with open('compression_value_dict.json', 'r') as json_file:
                compression_value_dict = json.load(json_file)
            if os.path.splitext(filename)[1] in config.file_extensions:
                compression_value_dict.update({project_name+filename: 1/max(com_1, com_2),
                                          project_name+os.path.splitext(filename)[0]+'.tiff': 1/max(com_1, com_2),
                                          project_name+os.path.splitext(filename)[0]+'.svs': 1/max(com_1, com_2)})
            with open('compression_value_dict.json', 'w') as json_file:
                json.dump(compression_value_dict, json_file)
        # os.system("touch "+os.path.join(config.workdir_data_path,f"{project_name}/"+os.path.splitext(filename)[0]+'.tiff'))
        with open(os.path.join(config.workdir_data_path,f"{project_name}/"+os.path.splitext(filename)[0]+'.tiff'), "w") as f:
            pass
        tiff.imwrite(os.path.join(config.workdir_data_path,f"{project_name}/"+os.path.splitext(filename)[0]+'_small.tiff'), compressed_image, photometric='rgb')
    elif '.pkl' in filename:
        with open(config.files_to_upload_path+filename, "rb") as file:
            metal_dict = pickle.load(file)
        with open(os.path.join(config.workdir_data_path,f"{project_name}/"+os.path.splitext(filename)[0]+'.pkl'), 'wb') as handle:
            pickle.dump(metal_dict, handle)

# @pysnooper.snoop()
def upload_file(config, list_of_contents, list_of_names, list_of_dates, upload_data_folder_dropdown):
    tmp_metal_path = os.path.join(config.tmp_folder_path, "tmp_metals.pkl")
    temp_metal_dict = {}
    print(list_of_names)
    if list_of_names is not None:
        
        with ProgressBar():
            d=dask.compute({f:dask.delayed(read_excel_fast_v2)(f,content) for f,content in zip(list_of_names,list_of_contents)},scheduler="single-threaded",num_workers=min(config.num_workers,len(list_of_names)))[0]
        d_=dict()#defaultdict(lambda : dict())
        for k in d:
            if "_ppm" in k:
                if np.prod(d[k].shape)>0:
                    d_[k.replace(" ","_").split("_")[-3]]=np.array(d[k])#[k.split("/")[-3 if k.split("/")[-3]!="LAICPMS" else -2]]#.replace(" ","_").split("_")[-4]
        elements=list(d_.keys())
        atomic_numbers=[int(re.sub(r"\D","",elem)) for elem in elements]
        
        d_=OrderedDict([(elem,d_[elem]) for elem in np.array(elements)[np.argsort(atomic_numbers)]])
        d_["All"]=np.sum(list(d_.values()), axis=0)
        d_=dict(metals=d_)
        
        pd.to_pickle(d_,tmp_metal_path)
        shutil.copy(tmp_metal_path,os.path.join(os.path.abspath(config.workdir_data_path),f"{upload_data_folder_dropdown}/"+str(upload_data_folder_dropdown)+'_metals.pkl'))

    return generate_files_df_records(config)[0]

# @pysnooper.snoop()
def upload_xlsx_file(config, selected_row, upload_data_folder_dropdown):
    tmp_metal_path = os.path.join(config.tmp_folder_path, "tmp_metals.pkl")
    sample=pd.DataFrame.from_records(config.upload_files_df_data_xlsx).iloc[np.array(selected_row)].values.flatten()[0]
    excel_files=glob.glob(os.path.join(config.files_to_upload_path,f"{sample} *_ppm matrix.xlsx"))
    if excel_files is not None:
        with ProgressBar():
            d=dask.compute({f:dask.delayed(read_excel_fast_v3)(f) for f in excel_files},scheduler="single-threaded",num_workers=min(config.num_workers,len(excel_files)))[0]
        d_=dict()#defaultdict(lambda : dict())
        for k in d:
            if "_ppm" in k:
                if np.prod(d[k].shape)>0:
                    d_[k.replace(" ","_").split("_")[-3]]=np.array(d[k])#[k.split("/")[-3 if k.split("/")[-3]!="LAICPMS" else -2]]#.replace(" ","_").split("_")[-4]
        elements=list(d_.keys())
        atomic_numbers_str=[re.sub(r"\D","",elem) for elem in elements]
        atomic_numbers=[(int(atomic_number) if atomic_number else -1000) for atomic_number in atomic_numbers_str]
        for elem,atomic_number in zip(elements,atomic_numbers):
            if atomic_number==-1000:
                elements.remove(elem)
                atomic_numbers.remove(atomic_number)    
        
        d_=OrderedDict([(elem,d_[elem]) for elem in np.array(elements)[np.argsort(atomic_numbers)]])
        d_["All"]=np.sum(list(d_.values()), axis=0)
        d_=dict(metals=d_)
        
        pd.to_pickle(d_,tmp_metal_path)
        shutil.copy(tmp_metal_path,os.path.join(os.path.abspath(config.workdir_data_path),f"{upload_data_folder_dropdown}/"+str(upload_data_folder_dropdown)+'_metals.pkl'))

    return generate_files_df_records(config)[0]

def upload_file(config, list_of_contents, list_of_names, list_of_dates, upload_data_folder_dropdown):
    tmp_metal_path = os.path.join(config.tmp_folder_path, "tmp_metals.pkl")
    temp_metal_dict = {}
    elements_search = '|'.join(f'(?:{element})' for element in config.elements)
    elements_pattern = re.compile(rf'.*\s+({elements_search})\d+_ppm.*\.xlsx$')
    print(list_of_names)
    if list_of_names is not None:
        
        with ProgressBar():
            d=dask.compute({f:dask.delayed(read_excel_fast_v2)(f,content) for f,content in zip(list_of_names,list_of_contents)},scheduler="single-threaded",num_workers=min(config.num_workers,len(list_of_names)))[0]
        d_=dict()#defaultdict(lambda : dict())
        for k in d:
            k=os.path.basename(k)
            if "_ppm" in k:
                if np.prod(d[k].shape)>0:
                    match = elements_pattern.match(k)
                    if match:
                        element = match.group(1)
                    else:
                        element = re.sub(r'\d+', '', k.replace(" ","_").split("_")[-3])
                    d_[element]=np.array(d[k])#[k.split("/")[-3 if k.split("/")[-3]!="LAICPMS" else -2]]#.replace(" ","_").split("_")[-4]
        elements=list(d_.keys())
        sorted_elements = sorted(elements, key=lambda x: config.elements.index(x))
        
        d_=OrderedDict([(elem,d_[elem]) for elem in sorted_elements])
        d_["All"]=np.sum(list(d_.values()), axis=0)
        d_=dict(metals=d_)
        
        pd.to_pickle(d_,tmp_metal_path)
        shutil.copy(tmp_metal_path,os.path.join(os.path.abspath(config.workdir_data_path),f"{upload_data_folder_dropdown}/"+str(upload_data_folder_dropdown)+'_metals.pkl'))

    return generate_files_df_records(config)[0]

# @pysnooper.snoop()
def upload_file_v2(config, _, upload_data_folder_dropdown, selected_rows):
    print(selected_rows)
    print('upload')
    if selected_rows is not None:
        for filename in pd.DataFrame.from_records(config.upload_files_df_data).iloc[np.array(selected_rows)].values.flatten(): 
            upload_contents(config, filename, upload_data_folder_dropdown)
    print('upload done')

    files_df = generate_files_df_records(config)[0]
    # config.upload_files_df_data = files_df
    
    return files_df,[]

    
def read_excel_fast(xlsx_file):
    csv_file=os.path.join("/tmpdir",os.path.basename(xlsx_file.replace(".xlsx",".csv")))
    if not os.path.exists(csv_file):
        Xlsx2csv(xlsx_file, outputencoding="utf-8").convert(csv_file)
    return pd.read_csv(csv_file)

def read_excel_fast_v2(xlsx_file,contents):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    return pd.read_excel(io.BytesIO(decoded),engine="calamine")

def read_excel_fast_v3(xlsx_file):
    return pd.read_excel(xlsx_file,engine="calamine")

def update_file_dir(config):
    upload_files_df_data=return_upload_files(config,exts=config.file_extensions+[".pkl"])
    return upload_files_df_data


def make_new_folder(config, n_clicks, new_folder_input, upload_data_folder_dropdown):
    if n_clicks > 0:
        if new_folder_input != None:
            if new_folder_input not in upload_data_folder_dropdown:
                os.mkdir(config.workdir_data_path+'/'+new_folder_input)
                upload_data_folder_dropdown.append(new_folder_input)
    return upload_data_folder_dropdown

def display_selected_file(config, n_clicks, selected_rows):
    
    all_lines = []
    if n_clicks > 0:
        print('1')
        if selected_rows:

            print('11')
            print('display selected_rows', selected_rows)

            project_df_list, file_name_df_list, file_type_list = [], [], []
            all_files_list_first = os.listdir(config.workdir_data_path)
            all_files_list = []
            for one_file in all_files_list_first:
                if one_file == '.DS_Store':
                    continue
                else:
                    all_files_list.append(one_file)
            for one_file in all_files_list:
                data_files_under_list = os.listdir(os.path.join(config.workdir_data_path, one_file))
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
            for one_file_name_index in range(len(file_name_df_list)):
                if one_file_name_index in selected_rows:
                    one_file_name = file_name_df_list[one_file_name_index]
                    one_project_name = project_df_list[one_file_name_index]
                    print('display', str(one_file_name))
                    
                    file_path = os.path.join(config.workdir_data_path,f"{one_project_name}/{one_file_name}")
                    if os.path.splitext(one_file_name)[1] in config.file_extensions:
                        with open('compression_value_dict.json', 'r') as json_file:
                            compression_value_dict = json.load(json_file)
                        config.compression_annotation_tab = compression_value_dict[one_project_name+one_file_name]

                        if not one_file_name.endswith('_small.tiff'):
                            file_path = os.path.splitext(file_path)[0]+'_small.tiff'

                        # im_small_crop_annotation_tab = tiff.imread(file_path)
                        # im_small_crop_co = tiff.imread(file_path)
                    else:
                        with open(file_path, "rb") as input_file:
                            config.metal_data = pickle.load(input_file)
                        with open(file_path, "rb") as input_file:
                            config.original_metal_data = pickle.load(input_file)
                        with open(file_path, "rb") as input_file:
                            config.metal_data_preprocess = pickle.load(input_file)
                            
                    new_line = html.P(
                            [str(one_file_name), ' is loaded'],
                            style={"display": "flex", "align-items": "center"}
                        )
                    all_lines.append(new_line)
            return all_lines
        return ''
