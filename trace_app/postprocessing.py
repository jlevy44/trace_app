import fire
import os
# import pysnooper

class Postprocessing:
    def __init__(self):
        pass

    def transfer_back_coordinates(self, exported_metals_annots_pkl_path="", preprocessed_metals_pkl_path="", export_unwarped_metals_annots_path=""):
        import pandas as pd
        from trace_app.data_processing import path_to_mask
        import numpy as np
        from collections import defaultdict
        from functools import reduce
        
        exported_metals_annots = pd.read_pickle(exported_metals_annots_pkl_path)
        preprocessed_metals = pd.read_pickle(preprocessed_metals_pkl_path)

        unwarped_exported_metals_annots = exported_metals_annots.copy()
        unwarped_exported_metals_annots['original_warped_shape']=exported_metals_annots['metals']['All'].shape
        H = exported_metals_annots.get('previous_homo', [exported_metals_annots['homo']])
        if len(H) == 1:
            H = H[0]
        else:
            H = reduce(np.matmul, H[::-1])
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

    # @pysnooper.snoop()
    def rescale_generate_pointcloud(self, exported_metals_annots_path, wsi_path=None, compression_dict_json_path=None, wsi_basename=None, upscale_factor=1.0, export_pointcloud_df_path="pointcloud_df_wsi_coords.pkl"):
        import pyvips
        import openslide
        import json, os
        import numpy as np
        import pandas as pd
        from functools import reduce
        VALID_EXTENSIONS = [".xlsx" , ".csv", ".pkl"]
        assert any(export_pointcloud_df_path.endswith(ext) for ext in VALID_EXTENSIONS), "export_pointcloud_df_path must be a .xlsx or .pkl file"
        excel = export_pointcloud_df_path.endswith(VALID_EXTENSIONS[0]) 

        transform_to_wsi_coords = (compression_dict_json_path and wsi_basename) or wsi_path

        exported_metals_annots_ = pd.read_pickle(exported_metals_annots_path)
        is_unwarped = exported_metals_annots_.get('unwarped', False)
        if not is_unwarped:
            assert os.path.basename(exported_metals_annots_path).endswith("exported_metals_annots.pkl") or os.path.basename(exported_metals_annots_path).endswith("coregistered_metals.pkl"), "Exported metals annots path must be a exported_metals_annots.pkl or coregistered_metals.pkl file"
        tissue_mask=~np.isnan(exported_metals_annots_['metals']['All'])
        
        elements=list(exported_metals_annots_['metals'].keys())
        elemental_dict={element:exported_metals_annots_['metals'][element][tissue_mask] for element in elements}
        pts=np.column_stack(np.where(tissue_mask))

        if transform_to_wsi_coords:
            H = exported_metals_annots_.get('previous_homo', [exported_metals_annots_['homo']])
            if len(H) == 1:
                H = H[0]
            else:
                H = reduce(np.matmul, H[::-1])
            H/=H[2,2]
            if is_unwarped:
                pts_h = np.column_stack([pts[:,::-1], np.ones(len(pts))])   # Nx3
                warped_pts_hom = H @ pts_h.T
                warped_pts_hom /= warped_pts_hom[2,:] 
                warped_pts = warped_pts_hom[:2,:].T  # shape (N,2)
                pts=warped_pts[:,::-1]
            warped_shape=exported_metals_annots_.get('original_warped_shape',None) if is_unwarped else exported_metals_annots_['metals']['All'].shape
        elemental_dict['X']=pts[:,1]
        elemental_dict['Y']=pts[:,0]
        pointcloud_df=pd.DataFrame(elemental_dict)

        if transform_to_wsi_coords:
            if compression_dict_json_path and wsi_basename:
                with open(compression_dict_json_path, 'r') as json_file:
                    compression_dict = json.load(json_file)
                    w,h=compression_dict[f"{wsi_basename}_im_width"],compression_dict[f"{wsi_basename}_im_height"]
                upscale_factor=np.mean(np.array([h,w])/np.array(warped_shape))
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
                upscale_factor=np.mean(np.array([h,w])/np.array(warped_shape))
                # else:
                #     raise ValueError("Either wsi_path or compression_dict_json_path and wsi_basename must be provided")
            
        pointcloud_df['X']*=upscale_factor
        pointcloud_df['Y']*=upscale_factor

        if "annotations" in list(exported_metals_annots_.keys()):
            annotations = exported_metals_annots_.get('annotations',{})
            annotations_list = list(annotations.keys())
            for annotation in annotations_list:
                pointcloud_df[annotation] = annotations[annotation][tissue_mask]
            pointcloud_df['label'] = pointcloud_df[annotations_list].idxmax(axis=1)
            pointcloud_df.loc[pointcloud_df[annotations_list].sum(axis=1) == 0, 'label'] = ''

        # if not export_pointcloud_df_path: export_pointcloud_df_path = os.path.join(os.path.dirname(exported_metals_annots_path), f"pointcloud_df_wsi_coords.{'xlsx' if excel else 'pkl'}")
        if excel: 
            pointcloud_df.to_excel(export_pointcloud_df_path)
        elif export_pointcloud_df_path.endswith(VALID_EXTENSIONS[1]):
            pointcloud_df.to_csv(export_pointcloud_df_path)
        else: pointcloud_df.to_pickle(export_pointcloud_df_path)

    def rescale_ei_to_wsi_dimensions(
    self,
    exported_metals_annots_path,
    wsi_path=None,
    compression_dict_json_path=None,
    wsi_basename=None,
    upscale_factor=1.0,
):
    import os
    import json
    import cv2
    import pyvips
    import openslide
    import tifffile as tiff
    import numpy as np
    import pandas as pd
    from functools import reduce

    exported_metals_annots_ = pd.read_pickle(exported_metals_annots_path)

    metals = exported_metals_annots_["metals"]
    is_unwarped = exported_metals_annots_.get("unwarped", False)

    # ---------------------------------------------------------
    # Determine registered / warped EI coordinate-space shape
    # ---------------------------------------------------------

    if is_unwarped:
        warped_shape = exported_metals_annots_.get(
            "original_warped_shape",
            None,
        )
        if warped_shape is None:
            raise ValueError(
                "original_warped_shape required when unwarped=True"
            )
    else:
        warped_shape = metals["All"].shape

    warped_h, warped_w = warped_shape

    # ---------------------------------------------------------
    # Homography: original EI -> registered EI coordinates
    # ---------------------------------------------------------

    H = None

    if is_unwarped:
        H = exported_metals_annots_.get(
            "previous_homo",
            [exported_metals_annots_["homo"]],
        )

        if len(H) == 1:
            H = H[0]
        else:
            H = reduce(np.matmul, H[::-1])

        H = np.asarray(H, dtype=np.float64)
        H /= H[2, 2]

    # ---------------------------------------------------------
    # Determine WSI dimensions
    # ---------------------------------------------------------

    if compression_dict_json_path and wsi_basename:

        with open(compression_dict_json_path, "r") as json_file:
            compression_dict = json.load(json_file)

        w = compression_dict[f"{wsi_basename}_im_width"]
        h = compression_dict[f"{wsi_basename}_im_height"]

    elif wsi_path:

        ext = os.path.splitext(wsi_path)[1].lower()

        pyvips_exts = {
            ".svs",
            ".tif",
            ".tiff",
            ".ome.tif",
            ".ome.tiff",
        }

        if ext in pyvips_exts:
            try:
                im = pyvips.Image.new_from_file(wsi_path)
                w, h = im.width, im.height
            except Exception:
                im = openslide.OpenSlide(wsi_path)
                w, h = im.get_level_dimensions(0)

        else:
            try:
                im = openslide.OpenSlide(wsi_path)
                w, h = im.get_level_dimensions(0)

            except Exception:
                im = tiff.imread(wsi_path)

                if im.shape[0] == 3:
                    w, h = im.shape[2], im.shape[1]
                else:
                    w, h = im.shape[1], im.shape[0]

    else:
        h = round(warped_h * upscale_factor)
        w = round(warped_w * upscale_factor)

    # ---------------------------------------------------------
    # Process one float32 EI channel
    # ---------------------------------------------------------

    def process_channel(channel):

        if channel.dtype != np.float32:
            raise TypeError("EI channels must be float32")

        valid_mask = ~np.isnan(channel)

        channel_filled = channel.copy()
        channel_filled[~valid_mask] = 0.0

        # Original EI -> registered EI coordinates
        if is_unwarped:
            channel_filled = cv2.warpPerspective(
                channel_filled,
                H,
                dsize=(warped_w, warped_h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0.0,
            )

            valid_mask = cv2.warpPerspective(
                valid_mask.astype(np.uint8),
                H,
                dsize=(warped_w, warped_h),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            ).astype(bool)

        # Registered EI coordinates -> WSI dimensions
        channel_wsi = cv2.resize(
            channel_filled,
            (w, h),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.float32, copy=False)

        valid_mask_wsi = cv2.resize(
            valid_mask.astype(np.uint8),
            (w, h),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)

        channel_wsi[~valid_mask_wsi] = np.nan

        return channel_wsi

    return {
        element: process_channel(channel)
        for element, channel in metals.items()
    }

def rescale_wsi_to_ei_dimensions(
    self,
    wsi_data,
    exported_metals_annots_path,
    wsi_path=None,
    compression_dict_json_path=None,
    wsi_basename=None,
    upscale_factor=1.0,
):
    import cv2
    import numpy as np
    import pandas as pd
    from functools import reduce

    exported_metals_annots_ = pd.read_pickle(exported_metals_annots_path)

    metals = exported_metals_annots_["metals"]
    is_unwarped = exported_metals_annots_.get("unwarped", False)

    original_ei_shape = metals["All"].shape
    original_h, original_w = original_ei_shape

    if is_unwarped:
        warped_shape = exported_metals_annots_.get(
            "original_warped_shape",
            None,
        )
        if warped_shape is None:
            raise ValueError(
                "original_warped_shape required when unwarped=True"
            )
    else:
        warped_shape = original_ei_shape

    warped_h, warped_w = warped_shape

    # ---------------------------------------------------------
    # Inverse homography:
    # registered EI coordinates -> original EI coordinates
    # ---------------------------------------------------------

    H_inv = None

    if is_unwarped:
        H = exported_metals_annots_.get(
            "previous_homo",
            [exported_metals_annots_["homo"]],
        )

        if len(H) == 1:
            H = H[0]
        else:
            H = reduce(np.matmul, H[::-1])

        H = np.asarray(H, dtype=np.float64)
        H /= H[2, 2]

        H_inv = np.linalg.inv(H)
        H_inv /= H_inv[2, 2]

    # ---------------------------------------------------------
    # Process one WSI-space float32 channel
    # ---------------------------------------------------------

    def process_channel(channel):

        if channel.dtype != np.float32:
            raise TypeError("Input channels must be float32")

        valid_mask = ~np.isnan(channel)

        channel_filled = channel.copy()
        channel_filled[~valid_mask] = 0.0

        # WSI dimensions -> registered EI dimensions
        channel_ei = cv2.resize(
            channel_filled,
            (warped_w, warped_h),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.float32, copy=False)

        valid_mask_ei = cv2.resize(
            valid_mask.astype(np.uint8),
            (warped_w, warped_h),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)

        # Registered EI -> original/unwarped EI coordinates
        if is_unwarped:

            channel_ei = cv2.warpPerspective(
                channel_ei,
                H_inv,
                dsize=(original_w, original_h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0.0,
            )

            valid_mask_ei = cv2.warpPerspective(
                valid_mask_ei.astype(np.uint8),
                H_inv,
                dsize=(original_w, original_h),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            ).astype(bool)

        channel_ei[~valid_mask_ei] = np.nan

        return channel_ei.astype(np.float32, copy=False)

    if isinstance(wsi_data, dict):
        return {
            name: process_channel(channel)
            for name, channel in wsi_data.items()
        }

    return process_channel(wsi_data)

def main():
    fire.Fire(Postprocessing)

if __name__ == "__main__":
    main()