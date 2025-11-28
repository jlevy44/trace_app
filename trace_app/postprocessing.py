import fire

class Postprocessing:
    def __init__(self):
        pass

    def transfer_back_coordinates(self, exported_metals_annots_pkl_path="", preprocessed_metals_pkl_path="", export_unwarped_metals_annots_path=""):
        import pandas as pd
        from trace_app.data_processing import path_to_mask
        import numpy as np
        from collections import defaultdict

        exported_metals_annots = pd.read_pickle(exported_metals_annots_pkl_path)
        preprocessed_metals = pd.read_pickle(preprocessed_metals_pkl_path)

        unwarped_exported_metals_annots = exported_metals_annots.copy()
        unwarped_exported_metals_annots['original_warped_shape']=exported_metals_annots['metals']['All'].shape
        H = exported_metals_annots['homo']
        H/=H[2,2]
        H_inv = np.linalg.inv(H)
        H_inv/=H_inv[2,2]
        annotations_polygons = exported_metals_annots['annotations_polygons'].copy()
        annotation_masks=defaultdict(lambda : np.zeros(preprocessed_metals['metals']['All'].shape,dtype=np.bool))
        for k in exported_metals_annots['annotations_polygons'].keys():
            for i,path in enumerate(exported_metals_annots['annotations_polygons'][k]):
                path=np.array([np.array(x.split(",")) for x in path[1:-1].split("L")]).astype(float)#[:,::-1]
                pts_h = np.column_stack([path, np.ones(len(path))])   # Nx3
                warped_pts_hom = H_inv @ pts_h.T
                warped_pts_hom /= warped_pts_hom[2,:] 
                warped_pts = warped_pts_hom[:2,:].T  # shape (N,2)
                path_points = []
                for x,y in warped_pts:
                    path_points.append(f"{x},{y}")
                path_string = "M" + "L".join(path_points) + "Z"
                annotations_polygons[k][i] = path_string
                mask=path_to_mask(path_string, preprocessed_metals['metals']['All'].shape,flip_y=False)
                annotation_masks[k] |= mask
        annotation_masks=dict(annotation_masks)
        unwarped_exported_metals_annots['annotations_polygons'] = annotations_polygons
        unwarped_exported_metals_annots['metals'] = preprocessed_metals['metals'].copy()
        unwarped_exported_metals_annots['annotations']=annotation_masks
        unwarped_exported_metals_annots['unwarped']=True
        
        if not export_unwarped_metals_annots_path:
            export_unwarped_metals_annots_path = os.path.join(os.path.dirname(exported_metals_annots_pkl_path), "unwarped_exported_metals_annots.pkl")
        pd.to_pickle(unwarped_exported_metals_annots, export_unwarped_metals_annots_path)

    def rescale_generate_pointcloud(self, exported_metals_annots_path, wsi_path=None, compression_dict_json_path=None, wsi_basename=None, export_pointcloud_df_path=None):
        import pyvips
        import openslide
        import json, os
        import numpy as np
        import pandas as pd

        exported_metals_annots_ = pd.read_pickle(exported_metals_annots_path)
        is_unwarped = exported_metals_annots_.get('unwarped', False)
        if not is_unwarped:
            assert os.path.basename(exported_metals_annots_path).endswith("exported_metals_annots.pkl") or os.path.basename(exported_metals_annots_path).endswith("coregistered_metals.pkl"), "Exported metals annots path must be a exported_metals_annots.pkl or coregistered_metals.pkl file"
        tissue_mask=~np.isnan(exported_metals_annots_['metals']['All'])
        elements=list(exported_metals_annots_['metals'].keys())
        elemental_dict={element:exported_metals_annots_['metals'][element][tissue_mask] for element in elements}
        warped_shape=exported_metals_annots_.get('original_warped_shape',None) if is_unwarped else exported_metals_annots_['metals']['All'].shape
        pts=np.column_stack(np.where(tissue_mask))
        if is_unwarped:
            pts_h = np.column_stack([pts[:,::-1], np.ones(len(pts))])   # Nx3
            warped_pts_hom = H @ pts_h.T
            warped_pts_hom /= warped_pts_hom[2,:] 
            warped_pts = warped_pts_hom[:2,:].T  # shape (N,2)
            pts=warped_pts[:,::-1]
        elemental_dict['X']=pts[:,1]
        elemental_dict['Y']=pts[:,0]
        pointcloud_df=pd.DataFrame(elemental_dict)

        if compression_dict_json_path and wsi_basename:
            with open(compression_dict_json_path, 'r') as json_file:
                compression_dict = json.load(json_file)
                w,h=compression_dict[f"{wsi_basename}_im_width"],compression_dict[f"{wsi_basename}_im_height"]
        elif wsi_path:
            ext = os.path.splitext(wsi_path)[1].lower()

            pyvips_exts = {'.svs', '.tif', '.tiff', '.ome.tif', '.ome.tiff'}

            if ext in pyvips_exts:
                try:
                    im=pyvips.Image.new_from_file(wsi_path)
                    w,h=im.width,im.height
                except Exception:
                    im=openslide.OpenSlide(wsi_path)
                    w,h=im.get_level_dimensions(0)
            else:
                try:
                    im=openslide.OpenSlide(wsi_path)
                    w,h=im.get_level_dimensions(0)
                except:
                    import tifffile as tiff
                    im=tiff.imread(wsi_path)
                    if im.shape[0]==3: w,h=im.shape[2],im.shape[1]
                    else: w,h=im.shape[1],im.shape[0]
        else:
            raise ValueError("Either wsi_path or compression_dict_json_path and wsi_basename must be provided")

        upscale_factor=np.mean(np.array([h,w])/np.array(warped_shape))
        pointcloud_df['X']*=upscale_factor
        pointcloud_df['Y']*=upscale_factor
   
        if not export_pointcloud_df_path:
            export_pointcloud_df_path = os.path.join(os.path.dirname(exported_metals_annots_path), "pointcloud_df_wsi_coords.pkl")
        pointcloud_df.to_pickle(export_pointcloud_df_path)

def main(): fire.Fire(Postprocessing)

if __name__ == "__main__":
    main()