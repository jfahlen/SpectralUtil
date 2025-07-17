import numpy as np
import os
import json
import glob
from osgeo import gdal
import pdb
import click
import skimage.exposure

import earthaccess
import spec_io

@click.command()
@click.argument('output_folder', type=click.Path(dir_okay=True, file_okay=False), required=True)
@click.option('--temporal', type=(click.DateTime(), click.DateTime()), help='Start and end time: %Y-%m-%dT%H:%M:%S %Y-%m-%dT%H:%M:%S')
@click.option('--count', type=int, default=2000, help='Max number of granules to search earthaccess for')
@click.option('--bounding_box', type=(float, float, float, float), help='lower_left_lon lower_left_lat upper_right_lon upper_right_lat')
@click.option('--overwrite', is_flag=True, default=False, help='Set to true to overwrite granules (download them again)')
@click.option('--symlinks_folder', type=click.Path(exists=True, dir_okay=True, file_okay=False), help = 'Location to put symlinks')
@click.option('--search_only', is_flag=True, default=False, help='Location to put symlinks')
def find_download_and_combine_EMIT(output_folder, 
                              temporal = None, 
                              count = 2000,
                              bounding_box = None,
                              overwrite = False,
                              symlinks_folder = None,
                              search_only = False):
    '''Find, download, and combine into VRTs all matching granules and store in OUTPUT_FOLDER

    Recommended usage: start with --search_only to review FIDs before downloading
    
    Search a DAAC using Earthaccess to find granules matching temporal and bounding_box. Download all the L1B and L2B products for
    each granule (except the full radiance as it takes too long and we don't need it generally for GHG analysis). Then, for each line,
    combine the granules into a single vrt file so they can be easily handled.

    The file structure is as follows:
    output_folder/granules/*: all files for each granule are stored here
    output_folder/EMIT*: the vrts are stored here

    For example:
    output_folder/granules/EMIT_L2B_CH4ENH_002_20250603T173453_2515412_004 # Note that the folder name says L2B but it has L1B too
    output_folder/granules/EMIT_L2B_CH4ENH_002_20250603T173505_2515412_005
    output_folder/EMIT20250603T173453_004_005 # Contains the vrts for the two scenes in the granules folder

    If symlinks_folder is provided, then symlinks to each of the vrt folders are created in this folder. The purpose is to have a single
    repository with links to all the data in it in case you don't remember where you put a case.

    Example call:

    python earthaccess_helpers_EMIT.py /store/jfahlen/test/EMIT_data --temporal 2024-10-04T16:00:00 2024-10-04T17:00:00 --bounding_box -103.74460188 32.22680624 -103.74481188 32.22700624 --search_only
    '''
    short_name_ghg, short_name_rdn, short_name_rfl = 'EMITL2BCH4ENH', 'EMITL1BRAD', 'EMITL2ARFL'

    r_ghg = earthaccess.search_data(short_name = short_name_ghg,
                                    temporal = temporal, count = count, bounding_box = bounding_box)
    r_rdn = earthaccess.search_data(short_name = short_name_rdn,
                                    temporal = temporal, count = count, bounding_box = bounding_box)
    r_rfl = earthaccess.search_data(short_name = short_name_rfl,
                                    temporal = temporal, count = count, bounding_box = bounding_box)

    earthaccess_fids = [g['meta']['native-id'] for g in r_ghg] # Ex: EMIT_L2B_CH4ENH_002_20250603T173453_2515412_004
    
    # Print names
    if search_only:
        print(f'Found {len(r_ghg)} GHG and {len(r_rdn)} RDN files.')
        for earthaccess_fid in earthaccess_fids:
            print(earthaccess_fid)
        return

    if os.path.exists(output_folder):
        if overwrite:
            raise ValueError(f'The output_folder {output_folder} already exists.')
    else:
        os.mkdir(output_folder)
    
    granule_path = os.path.join(output_folder, 'granules')
    if not os.path.exists(granule_path):
        os.mkdir(granule_path)

    for i, (efid, rd, gh, rf) in enumerate(zip(earthaccess_fids, r_rdn, r_ghg, r_rfl)):
        print(f'Downloading {efid}, #{i+1} of {len(earthaccess_fids)}')
        download_an_EMIT_granule(rd, gh, rf, granule_path, overwrite = False)
    
    # Get all orbit IDs
    orbit_ids = sorted(list(set([x.split('_')[-2] for x in earthaccess_fids]))) 

    # Make vrt files that combine each scene in a pass into one vrt file
    fids_with_scene_numbers = [] # Ex: EMIT20250603T173453_004_005
    for orbit_id in orbit_ids:
        fid_with_scene_numbers = join_EMIT_scenes_as_VRT(orbit_id, granule_path, output_folder, 
                                                        tags_to_join = ['L2A_MASK', 'CH4ENH', 'CH4SENS', 'CH4UNCERT'],
                                                        rgb_channel_idx = [35, 23, 11]) # 641, 552, 462 nm  
        fids_with_scene_numbers.append(fid_with_scene_numbers)
        join_EMIT_scenes_as_VRT_pixel_time_only(orbit_id, granule_path, output_folder)
                                              
    # Make symlinks to the granules folder in the symlinks_folder
    if symlinks_folder is not None:
        folders = glob.glob(granule_path + '/*')
        for folder in folders:
            dst = os.path.join(symlinks_folder, os.path.split(folder)[-1])
            if os.path.islink(dst): # link exists
                if os.path.exists(dst): # target exists (link is not broken)
                    print(f'Valid link {dst} already exists')
                    pass
                else: # link is broken, remove existing link and update it
                    os.remove(dst.rstrip('/').rstrip('\\'))
                    os.symlink(folder, dst)
            else: # link did not exist, so create it
                os.symlink(folder, dst)

def join_EMIT_scenes_as_VRT_pixel_time_only(orbit_id, granule_storage_location, output_location):

    glob_str = os.path.join(granule_storage_location, f'*_{orbit_id}_*')
    folders = glob.glob(glob_str)
    if len(folders) == 0:
        raise ValueError(f'There are no folders matching {glob_str}')

    files = []
    for folder in folders:
        j = json.load(open(os.path.join(folder, 'data_files.json'),'r'))
        obs_filename = j[f'OBS']
        out_tif = os.path.join(folder, os.path.split(obs_filename)[-1].split('.')[0] + '_times_only.tif')

        m_obs, d_obs = spec_io.load_data(obs_filename, load_glt = True, lazy = False)
        d_obs_ort = spec_io.ortho_data(d_obs[:,:,-2], m_obs.glt)

        # Save the orthoed times to tif so we can make a vrt below
        spec_io.write_geotiff(d_obs_ort, m_obs, out_tif)

        j['OBS_ORT_times_only'] = out_tif
        json.dump(j, open(os.path.join(folder, 'data_files.json'),'w'), indent = 4)

        files.append(out_tif)

    # Create folder and vrt name like: EMIT20250603T173453_004_005
    scene_numbers = [os.path.splitext(x)[0].split('_')[-3] for x in files]
    scene_numbers = sorted(scene_numbers)

    fids = [os.path.splitext(x)[0].split('_')[-5] for x in files]
    fids = sorted(fids)

    fid_with_scene_numbers = f'EMIT{fids[0]}_{scene_numbers[0]}_{scene_numbers[-1]}'
    output_folder = os.path.join(output_location, fid_with_scene_numbers)
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    vrt_filename = os.path.join(output_folder, f'{fid_with_scene_numbers}_obs_times.vrt')
    my_vrt = gdal.BuildVRT(vrt_filename, files)
    my_vrt = None

def join_EMIT_scenes_as_VRT(orbit_id, granule_storage_location, output_location, 
                           tags_to_join = ['CH4_UNC_ORT', 'CH4_SNS_ORT', 'CH4_ORT', \
                                           'CO2_UNC_ORT', 'CO2_SNS_ORT', 'CO2_ORT'], 
                           rgb_channel_idx = [0,1,2]):
    '''Combine all the granule files that match {tags_to_join} with {fid} in {granule_storage_location} into 
    vrt files in output_location.

    Create an RGB from the radiance called RDN_QL
    '''
    glob_str = os.path.join(granule_storage_location, f'*_{orbit_id}_*')
    folders = glob.glob(glob_str)
    if len(folders) == 0:
        raise ValueError(f'There are no folders matching {glob_str}')
    
    for tag in tags_to_join:
        files = []
        for folder in folders:
            j =json.load(open(os.path.join(folder, 'data_files.json'),'r'))
            fs = j[f'{tag}']
            
            # The L2A mask file is not orthorectified, so do it here and save at tif.
            # Append the tif filename instead of the .nc filename
            if tag == 'L2A_MASK': 
                m, d = spec_io.load_data(fs, load_glt = True, load_loc = True)
                d_ort = spec_io.ortho_data(d, m.glt)
                fs = os.path.splitext(fs)[0] +  '.tif'
                spec_io.write_geotiff(d_ort, m, fs)
                j['L2A_MASK_ORT'] = fs
                json.dump(j, open(os.path.join(folder, 'data_files.json'),'w'), indent = 4)

            files.append(fs)
        
        scene_numbers = [os.path.splitext(x)[0].split('_')[-1] for x in files]
        scene_numbers = sorted(scene_numbers)

        fids = [os.path.splitext(x)[0].split('_')[-3] for x in files]
        fids = sorted(fids)

        output_folder = os.path.join(output_location, f'EMIT{fids[0]}_{scene_numbers[0]}_{scene_numbers[-1]}')
        if not os.path.exists(output_folder):
            os.mkdir(output_folder)

        # Make output name be the fid plume the first and last scene included in the vrt plus the tag
        fid_with_scene_numbers = f'EMIT{fids[0]}_{scene_numbers[0]}_{scene_numbers[-1]}'
        vrt_filename = os.path.join(output_folder, f'{fid_with_scene_numbers}_{tag.split(".")[0]}.vrt')
        my_vrt = gdal.BuildVRT(vrt_filename, files)
        my_vrt = None

    # Now convert RDN_QL from unorthoed netCDF4 to orthoed geotiff, then make VRT
    if rgb_channel_idx is not None:
        rdn_ort_tif_filenames = []
        for folder in folders:

            j = json.load(open(os.path.join(folder, 'data_files.json'),'r'))
            rdn_filename = j['RAD']
            obs_filename = j['OBS']

            _, d_rad = spec_io.load_data(rdn_filename, load_glt = True, lazy = False)
            m_obs, _ = spec_io.load_data(obs_filename, load_glt = True, lazy = False)
            rgb_data = spec_io.ortho_data(d_rad[:,:,rgb_channel_idx], m_obs.glt)
            rgb_data = scale_for_rgb(rgb_data)

            # Make orthoed tif so we can create a vrt below
            rdn_ort_tif_filename = rdn_filename.replace('L1B_RAD', 'RDN_QL').replace('.nc', '.tif')
            spec_io.write_geotiff(rgb_data, m_obs, rdn_ort_tif_filename)

            j['RDN_QL_ORT'] = rdn_ort_tif_filename
            json.dump(j, open(os.path.join(folder, 'data_files.json'),'w'), indent = 4)

            rdn_ort_tif_filenames.append(rdn_ort_tif_filename)
        
        fid_with_scene_numbers = f'EMIT{fids[0]}_{scene_numbers[0]}_{scene_numbers[-1]}'
        vrt_filename = os.path.join(output_folder, f'{fid_with_scene_numbers}_RDN_QL.vrt')
        my_vrt = gdal.BuildVRT(vrt_filename, rdn_ort_tif_filenames)
        my_vrt = None
    
    return fid_with_scene_numbers

def download_an_EMIT_granule(rdn_granule, ghg_granule, rfl_granule, storage_location, overwrite = False):
    name = ghg_granule['meta']['native-id']
    output_folder = os.path.join(storage_location, name)
    download = False
    if not os.path.exists(output_folder):
        download = True
        os.mkdir(output_folder)
    if overwrite:
        download = True
    
    if download:
        earthaccess.login(persist=True)
        rdn_files_without_full_RDN = [x for x in rdn_granule.data_links() if 'RDN.nc' not in x]
        download_from_urls(rdn_files_without_full_RDN, output_folder)
        download_from_urls(ghg_granule.data_links(), output_folder)
        l2a_mask_files = [x for x in rfl_granule.data_links() if 'L2A_MASK' in x]
        download_from_urls(l2a_mask_files, output_folder)

        tags = ['OBS', 'RAD', 'L2A_MASK', 'CH4ENH', 'CH4SENS', 'CH4UNCERT']
        make_files_list_from_urls_or_glob(rdn_files_without_full_RDN + 
                                          ghg_granule.data_links() +
                                          rfl_granule.data_links(), 
                                          tags, output_folder)

    return

def download_from_urls(urls, outpath):
    # Get requests https Session using Earthdata Login Info
    fs = earthaccess.get_requests_https_session()
    # Retrieve granule asset ID from URL (to maintain existing naming convention)
    for url in urls:
        granule_asset_id = url.split('/')[-1]
        # Define Local Filepath
        fp = os.path.join(outpath, granule_asset_id)
        # Download the Granule Asset if it doesn't exist
        if not os.path.isfile(fp):
            with fs.get(url,stream=True) as src:
                with open(fp,'wb') as dst:
                    for chunk in src.iter_content(chunk_size=64*1024*1024):
                        dst.write(chunk)

def make_files_list_from_urls_or_glob(urls, tags, outpath):
    '''Using a list of URLs from granule.data_links() or the file paths from glob.glob(granules_folder), create and
    store a json file containing the full path to the granule component files downloaded with download_from_urls. This
    is helpful since the downloaded files have an unpredictable (or at least hard to find) hash in them.
    
    Parameters:
        urls : list of strings
            List of URLs or full paths to the granules folder
        tags : list of strings
            The file name tags to include in the json file, ex: 'OBS', 'CH4_SNS_ORT', etc.
        output : string
            The path to the granules location where the output json will go
    '''
    # URL example: https://data.ornldaac.earthdata.nasa.gov/protected/aviris/AV3_L1B_RDN/data/AV320241002t160425_014_L1B_ORT_55901fd4_OBS.nc
    urls_cut = [url.split('/')[-1] for url in urls] # Ex: AV320241002t160425_014_L1B_ORT_55901fd4_OBS 
    urls_cut_noext = [x.split('.')[0] for x in urls_cut]

    files_list = {}
    for tag in tags:
        try:
            #ind = [i for i, x in enumerate(urls_cut_noext) if x.endswith(tag)][0]
            ind = [i for i, x in enumerate(urls_cut_noext) if tag in x][0]
        except:
            pdb.set_trace()
        files_list[tag] = os.path.join(outpath, f'{urls_cut[ind]}')
    
    json.dump(files_list, open(os.path.join(outpath, 'data_files.json'), 'w'), indent = 4)
    return

def scale_for_rgb(d_in):
    d = np.where(d_in < -9990, np.nan, d_in)
    d_out = np.zeros_like(d) #.astype(int)
    for i in np.arange(d.shape[-1]):
        mi, ma = np.nanpercentile(d[:,:,i], [.5,99.5])
        di = skimage.exposure.rescale_intensity(d_in[:,:,i], in_range = (mi,ma))
        di = skimage.exposure.equalize_hist(di)
        d_out[:,:,i] = skimage.exposure.rescale_intensity(di, in_range = (di.min(), di.max()))

    return np.where(d_in > -9990, d_out, -9999)

if __name__ == '__main__':
    find_download_and_combine_EMIT()