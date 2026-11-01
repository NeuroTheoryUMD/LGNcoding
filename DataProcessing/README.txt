# LGN recordings (DANDI:002028): reader and demo

lgndata_nwb.py: python tools for reading the Dandiset and formatting the data for various applications. Requires: numpy, pandas, h5py

LGN_NWB_demo.ipynb: a notebook that demonstrates various organization and applications with the dataset. Requires additionally: matplotlib.

Quick start for using the LGN DANDIset:

# dandi download DANDI:002028          # creates the folder 002028/
import lgndata_nwb as lgn
lgn.set_data_root("002028")
cells = lgn.find_cells(modality="NAT", regime="unique")
c = lgn.load_cell(int(cells.experiment_number.iloc[0]), int(cells.penetration_number.iloc[0]), int(cells.cell_number.iloc[0]), cells.trial_key.iloc[0])
