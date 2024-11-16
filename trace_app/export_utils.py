import copy
import os
import numpy as np
import pandas as pd
import xarray as xr
from spatialdata import SpatialData
from spatialdata.models import Image2DModel, Labels2DModel
import dask
from dask.diagnostics import ProgressBar
from dash import dcc
from dash.exceptions import PreventUpdate

from .data_processing import path_to_mask
from .file_utils import generate_files_df_records
def export_on_click(config, n_clicks, table_children):
    return dcc.send_data_frame(config.newest_result_df.to_csv, "export_result.csv")

def export_raw_data_on_click(config,_, project_name):
    metal_data_export=copy.deepcopy(config.metal_data)
    metal_data_export["annotations"]=copy.deepcopy(config.all_annot_data)

    all_type_list = []
    for one_shape in metal_data_export["annotations"]['shapes']:
        all_type_list.append(config.color_to_type_dict[one_shape['line']['color']])
    all_type_list = list(set(all_type_list))
    
    all_type_area_list_dict = {}
    for one_type in all_type_list:
        all_type_area_list_dict[one_type] = []
    
    for one_shape in metal_data_export["annotations"]['shapes']:
        all_type_area_list_dict[config.color_to_type_dict[one_shape['line']['color']]].append(one_shape['path'])
        
    all_type_area_dict = {}
        
    for one_type in list(all_type_area_list_dict.keys()):
        for one_path in all_type_area_list_dict[one_type]:
            if one_type not in list(all_type_area_dict.keys()):
                all_type_area_dict[one_type] = path_to_mask(one_path, 
                                                            metal_data_export['metals']['All'].shape,
                                                            flip_y=False,
                                                            # config.scale_factor, 
                                                           )
            else:
                all_type_area_dict[one_type] += path_to_mask(one_path, 
                                                             metal_data_export['metals']['All'].shape,
                                                             flip_y=False,
                                                            #  config.scale_factor, 
                                                            )
    metal_data_export["annotations"]=all_type_area_dict
    metal_data_export["annotations_polygons"]=all_type_area_list_dict

    pd.to_pickle(metal_data_export,f"./data/{project_name}/exported_metals_annots.pkl")
    images_dataarray = xr.DataArray(
        np.stack([metal_data_export["metals"][key] for key in metal_data_export["metals"]], axis=0),
        dims=["element", "r", "c"],
        coords={"element": list(metal_data_export["metals"].keys())}
    )
    annotations_dataarray = xr.DataArray(
        np.stack([metal_data_export["annotations"][key] for key in metal_data_export["annotations"]], axis=0),
        dims=["annotation", "r", "c"],
        coords={"annotation": list(metal_data_export["annotations"].keys())}
    )
    combined_dataset = xr.Dataset({
        "images": images_dataarray,
        "annotations": annotations_dataarray
    })

    combined_dataset.to_zarr(f"./data/{project_name}/exported_metals_annots.zarr",mode="w")

    images_dataarray=combined_dataset['images']
    images_dataarray.attrs["element_names"] = images_dataarray.coords['element']  # Save unique names as metadata

    annotations_dataarray=combined_dataset['annotations']

    background_mask = (annotations_dataarray.sum(dim="annotation") == 0).astype(int)
    annotations_with_background = xr.concat(
        [background_mask.rename("annotation").expand_dims(annotation=[0]), annotations_dataarray],
        dim="annotation"
        )
    annotations_with_background = annotations_with_background.assign_coords(
        annotation=["background"] + list(annotations_dataarray.annotation.values)
        ).argmax(dim="annotation")
    annotations_with_background.attrs["annotation_names"] = ["background"] + list(annotations_dataarray.annotation.values)  # Save unique names as metadata

    images_obj = Image2DModel.parse(images_dataarray.rename({"element": "c", "r": "y", "c": "x"}))#,dims=("c","x","y"))
    annotations_obj = Labels2DModel.parse(annotations_with_background.rename({"r": "y", "c": "x"}))#,dims=("c","x","y"))
    spatial_data = SpatialData(images={"images": images_obj}, labels={"annotations": annotations_obj})

    spatial_data.write(f"./data/{project_name}/exported_metals_annots_spatialdata.zarr")
    
    files_df_records = generate_files_df_records(config)[0]
    return files_df_records

