import fire

class Postprocessing:
    def __init__(self):
        pass

    def rescale_generate_pointcloud(self, exported_metals_annots_pkl_path, wsi_path=None, compression_dict=None):
        raise NotImplementedError("Not implemented yet")
        # either wsi_path or compression_dict must be provided
        if wsi_path is None and compression_dict is None:
            raise ValueError("Either wsi_path or compression_dict must be provided")
        if wsi_path is not None:
            # read wsi
            pass
        if compression_dict is not None:
            # read compression_dict
            pass
        pass

    def transfer_back_coordinates(self, exported_metals_annots_pkl_path, preprocessed_metals_pkl_path):
        raise NotImplementedError("Not implemented yet")
        # read exported_metals_annots_pkl
        with open(exported_metals_annots_pkl_path, "rb") as f:
            exported_metals_annots = pickle.load(f)
        # read preprocessed_metals_pkl
        with open(preprocessed_metals_pkl_path, "rb") as f:
            preprocessed_metals = pickle.load(f)
        pass

if __name__ == "__main__":
    fire.Fire(Postprocessing)