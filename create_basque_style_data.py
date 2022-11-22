#!/usr/bin/env python
# coding: utf-8

# In[32]:


import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm


# In[33]:


import neuralhydrology.datasetzoo.camelsus as camelsus
import neuralhydrology.datautils.pet as pet


# In[38]:


# Path to the CAMELS dataset.
camels_data_path = Path('camels/basin_dataset_public_v1p2')

def load_basin_data(basin: str) -> pd.DataFrame:
    # Load forcing data.
    forcings, area = camelsus.load_camels_us_forcings(
        data_dir=camels_data_path,
        basin=basin,
        forcings='daymet'
    )

    # Load catchment attributes to get latitude.
    attributes = camelsus.load_camels_us_attributes(
        data_dir=camels_data_path,
        basins=[basin]
    )

    # Calculate PET from forcing data.
    forcings['PET'] = pet.get_priestley_taylor_pet(
        t_min=forcings['tmin(C)'].values,
        t_max=forcings['tmax(C)'].values,
        s_rad=forcings['srad(W/m2)'].values,
        lat=attributes.loc[basin, 'gauge_lat'],
        elev=attributes.loc[basin, 'elev_mean'],
        doy=np.array([ts.day_of_year for ts in forcings.index])
    )

    # Load area-normalized discharge.
    forcings['streamflow'] = camelsus.load_camels_us_discharge(
        data_dir=camels_data_path,
        basin=basin, 
        area=area
    )

    # Ensure that all values are >= 0.
    # This only works for the columns we are using here.
    # forcings[forcings < 0] = 0

    forcings['temperature'] = (forcings['tmin(C)'] + forcings['tmax(C)']) / 2

    forcings.rename(columns={'prcp(mm/day)': 'precipitation'}, inplace=True)
    return forcings[['precipitation', 'PET', 'streamflow', 'temperature']]


# In[39]:


basin_file = '531_basin_list.txt'
with open(basin_file, 'rt') as f:
    lines = f.readlines()
basins = [basin.strip('\n') for basin in lines]
print(f'There are {len(basins)} basins.')


# In[40]:


output_file_directory = Path('basque-data')
def save_forcing_file(basin, df):
    with open(output_file_directory / f'{basin}.txt', 'wt') as f:
        df.to_csv(f)


# In[41]:


for basin in tqdm(basins):
    df = load_basin_data(basin)
    save_forcing_file(basin, df)


# In[ ]:





# In[ ]:




