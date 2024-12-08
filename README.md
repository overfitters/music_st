# Music Style Transfer
Style Transfer for Music

- `fma/` directory contains the free music archive dataset, with a metadata folder inside of it that contains the metadata.
  - Is not uploaded with the repo and should be taken from: [fma](https://github.com/mdeff/fma)
  - There is some additional preprocessing done on it afterwards.
    - We use `prepare_data.ipynb` to find invalid files in the dataset and remove them.
- `replication/` contains notebooks trying to replicate some of the earlier work done in this field.
  - `vq_vae` is not complete and does not work as of yet.
- `src/` contains the main code for the project.
- `train_pairs` & `val_pairs` contain the pairs for the main model that is based on the `vq_vae` paper.
  - these are generated using the `prepare_data.ipynb` notebook.
  - requires `fma` to be present in the same directory. (or change `DATASET` in `prepare_data.ipynb`)
