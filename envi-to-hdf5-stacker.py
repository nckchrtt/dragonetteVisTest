import spectral
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.transform import Affine
from rasterio.crs import CRS
import numpy as np
import os
import h5py
from pyproj import CRS as PyProjCRS # To handle ENVI's projection names

def parse_envi_map_info(map_info_str, width, height):
    """
    Parses the 'map info' string from an ENVI header to create a CRS and transform.
    This is a simplified parser and may need adjustment for specific projections.

    Args:
        map_info_str (str): The 'map info' string.
        width (int): The width of the raster image.
        height (int): The height of the raster image.

    Returns:
        tuple: A tuple containing (rasterio.crs.CRS, rasterio.transform.Affine)
    """
    parts = map_info_str

    # Example map info:
    # {UTM, 1.000, 1.000, 458334.500, 5215383.500, 60.00000000, 60.00000000, 10, North, WGS-84, units=Meters}
    proj_name = parts[0].strip('{')
    ul_x = float(parts[3]) # Easting of upper-left pixel center
    ul_y = float(parts[4]) # Northing of upper-left pixel center
    pixel_size_x = float(parts[5])
    pixel_size_y = -abs(float(parts[6])) # Y pixel size is typically negative

    # Construct the Affine transform
    # This defines the location of the top-left corner of the top-left pixel
    transform = Affine(pixel_size_x, 0.0, ul_x - (pixel_size_x / 2.0),
                       0.0, pixel_size_y, ul_y - (pixel_size_y / 2.0))

    # Construct the CRS
    # This is a bit tricky as ENVI names don't always map directly to EPSG codes
    crs = None
    if proj_name.lower() == 'utm':
        zone = parts[7]
        north_south = parts[8]
        datum = parts[9]
        # Try to build a proj4 string or find an EPSG code
        try:
            crs_proj = PyProjCRS(f"+proj=utm +zone={zone} +datum={datum} +{'north' if north_south.lower()=='north' else 'south'} +units=m +no_defs")
            crs = CRS.from_wkt(crs_proj.to_wkt())
        except Exception as e:
            print(f"Warning: Could not parse UTM projection for CRS: {e}. Falling back.")
            crs = None
    
    if not crs:
        print("Warning: Could not determine a robust CRS. Georeferencing may be inaccurate.")
        # A fallback if pyproj fails
        crs = CRS.from_dict(init='epsg:32610') # User should change this to a sensible default

    return crs, transform

def process_envi_to_hdf5(input_dir, output_hdf5_path):
    """
    Finds ENVI files in a directory, warps them to a common grid,
    stacks them, and saves them to an HDF5 file.

    Args:
        input_dir (str): Path to the directory containing .hdr/.dat files.
        output_hdf5_path (str): Path for the output HDF5 file.
    """
    print(f"Scanning for ENVI .hdr files in: {input_dir}")
    # Find all .hdr files
    hdr_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.lower().endswith('.hdr')]
    if not hdr_files:
        print("Error: No .hdr files found in the specified directory.")
        return

    print(f"Found {len(hdr_files)} files. Reading metadata...")

    # --- 1. Determine the target grid by reading all file headers ---
    all_bounds = []
    source_info = [] # Store info to avoid re-reading files

    for hdr_path in hdr_files:
        header = spectral.io.envi.read_envi_header(hdr_path)
        
        if 'map info' not in header:
            print(f"Warning: Skipping {os.path.basename(hdr_path)} - no 'map info' found in header.")
            continue

        width = int(header['samples'])
        height = int(header['lines'])
        bands = int(header['bands'])
        
        # NOTE: This script now processes ALL bands from each file.
        print(f"  - Found {os.path.basename(hdr_path)} with {bands} band(s).")

        crs, transform = parse_envi_map_info(header['map info'], width, height)
        
        # Calculate bounds for this specific image
        bounds = rasterio.coords.BoundingBox(
            left=transform.c,
            bottom=transform.f + transform.e * height,
            right=transform.c + transform.a * width,
            top=transform.f
        )
        all_bounds.append(bounds)
        source_info.append({'path': hdr_path, 'crs': crs, 'transform': transform, 'width': width, 'height': height, 'bands': bands})

    if not source_info:
        print("Error: No valid, georeferenced ENVI files could be processed.")
        return

    # --- VALIDATION: Check for consistent band counts ---
    num_bands_per_frame = source_info[0]['bands']
    if not all(info['bands'] == num_bands_per_frame for info in source_info):
        raise ValueError("All input ENVI files must have the same number of bands to create a 4D [frame, band, y, x] cube.")
    print(f"\nAll files have a consistent {num_bands_per_frame} bands. Proceeding with 4D cube creation.")

    # Use the CRS of the first valid file as the destination CRS
    dst_crs = source_info[0]['crs']
    
    # Calculate the final output bounds (union of all individual bounds)
    dst_left = min(b.left for b in all_bounds)
    dst_bottom = min(b.bottom for b in all_bounds)
    dst_right = max(b.right for b in all_bounds)
    dst_top = max(b.top for b in all_bounds)
    
    print("\nCalculating common grid...")
    # Determine output grid resolution from the first source (assumes consistent pixel size)
    src_transform0 = source_info[0]['transform']
    res_x = src_transform0.a
    res_y = abs(src_transform0.e)

    # Compute integer output dimensions from union bounds and resolution
    dst_width = int(np.ceil((dst_right - dst_left) / res_x))
    dst_height = int(np.ceil((dst_top - dst_bottom) / res_y))

    # Validate computed dimensions to avoid creating 0-sized datasets
    if dst_width <= 0 or dst_height <= 0:
        raise ValueError(f"Computed destination dimensions are invalid: {dst_width} x {dst_height}. "
                         "Check input file georeferencing and map info parsing.")

    # Construct an Affine transform for the destination grid (top-left corner)
    dst_transform = Affine(res_x, 0.0, dst_left, 0.0, -res_y, dst_top)

    print(f"Using resolution {res_x} x {res_y}. Final grid dimensions (WxH): {dst_width} x {dst_height}")

    # --- 2. Warp each image and stack ---
    print("\nInitializing data cube and warping images...")
    
    # Create an empty 4D NumPy array to hold the final stack
    # Shape is [frame_index, band_index, raster_y, raster_x]
    num_frames = len(source_info)
    print(f"Output cube will have shape: ({num_frames}, {num_bands_per_frame}, {dst_height}, {dst_width}).")
    stacked_cube = np.zeros((num_frames, num_bands_per_frame, dst_height, dst_width), dtype=np.float32)

    for i, src_info in enumerate(source_info):
        print(f"  Processing ({i+1}/{len(source_info)}): {os.path.basename(src_info['path'])}")
        # Open image and iterate through its bands
        img = spectral.open_image(src_info['path'])
        for b in range(src_info['bands']):
            if (b + 1) % 10 == 0 or b == src_info['bands'] - 1 or src_info['bands'] == 1:
                 print(f"    - Warping source band {b+1}/{src_info['bands']} into frame {i+1}")

            source_array = img.read_band(b)

            # Warp the source band onto the destination grid
            reproject(
                source=source_array,
                destination=stacked_cube[i, b], # Reproject into the correct [frame, band] slice
                src_transform=src_info['transform'],
                src_crs=src_info['crs'],
                dst_transform=dst_transform,
                dst_crs=dst_crs,
                resampling=Resampling.bilinear # Or nearest, cubic, etc.
            )

    # --- 3. Save to HDF5 ---
    print(f"\nWarping complete. Saving stacked cube to: {output_hdf5_path}")
    with h5py.File(output_hdf5_path, 'w') as f:
        # Create a dataset for the image stack with explicit shape and dtype
        dset = f.create_dataset('image_stack', 
                                shape=stacked_cube.shape, 
                                dtype=stacked_cube.dtype, 
                                data=stacked_cube, 
                                compression='gzip')

        # Save georeferencing metadata as attributes
        dset.attrs['crs_wkt'] = dst_crs.to_wkt()
        dset.attrs['transform'] = dst_transform.to_gdal()
        
        # Store the original file order, which corresponds to the first dimension (frames)
        original_files = [os.path.basename(info['path']) for info in source_info]
        dset.attrs['source_files'] = original_files

    print("\nProcessing finished successfully!")


# --- Main execution block ---
if __name__ == '__main__':
    # --- PLEASE CONFIGURE THESE PATHS ---
    
    # Directory containing your ENVI .hdr and .dat files
    # Create a folder named 'envi_images' and place your files there, or change this path.
    input_directory = 'C:/satelliteImagery/dragonette/tait-roi'
    
    # Path for the final output HDF5 file.
    output_file = 'C:/satelliteImagery/dragonette/tait-roi/aligned_image_stack.h5'

    # Warn the user if the directory doesn't exist
    if not os.path.exists(input_directory):
        raise FileNotFoundError(f"The specified input directory '{input_directory}' does not exist. Please create it and add ENVI files.")
    
    
    # Run the main processing function
    process_envi_to_hdf5(input_directory, output_file)

