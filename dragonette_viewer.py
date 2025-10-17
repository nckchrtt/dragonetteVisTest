import os
import requests
import rasterio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.patches import Rectangle
import json
import h5py

# --- Configuration ---
# Directory for storing the downloaded files.
DOWNLOAD_DIR = "C:/satelliteImagery/dragonette"

# Default bands to display initially. These are chosen to approximate a
# natural color view. We use 1-based indexing for user display.
INITIAL_BANDS = {
    'red': 5,
    'green': 3,
    'blue': 2,
}

def create_download_dir(directory):
    """Create the directory for storing downloads if it doesn't exist."""
    if not os.path.exists(directory):
        print(f"Creating directory: {directory}")
        os.makedirs(directory)

def parse_stac_item_for_assets(stac_item, base_url):
    """
    Parses a loaded STAC item dictionary to find and return a list of desired assets.
    """
    assets_found = []
    from urllib.parse import urljoin

    # Define the assets we are interested in by their title
    desired_assets = {
        "Cloud optimized GeoTiff": "data",
        "Data Mask": "mask",
        "Pixel Quality Mask": "mask"
    }

    main_data_asset = None
    other_assets = []

    # Find all desired assets
    for asset_key, asset in stac_item.get('assets', {}).items():
        asset_title = asset.get('title')
        if asset_title in desired_assets:
            relative_href = asset.get('href')
            if relative_href:
                asset_info = {
                    'url': urljoin(base_url, relative_href),
                    'filename': os.path.basename(relative_href),
                    'title': asset_title
                }
                if desired_assets[asset_title] == 'data':
                    main_data_asset = asset_info
                else:
                    other_assets.append(asset_info)

    if main_data_asset:
        # Ensure the main data file is first in the list
        assets_found.append(main_data_asset)
        assets_found.extend(other_assets)
    
    return assets_found

def get_asset_info_from_stac_url(stac_json_url, download_directory=DOWNLOAD_DIR):
    """
    Fetches a STAC Item JSON, saves it, and extracts asset info.
    Returns a tuple containing the list of assets and the full STAC item dict.
    """
    stac_item = None
    try:
        json_filename = os.path.basename(stac_json_url)
        json_filepath = os.path.join(download_directory, json_filename)

        if os.path.exists(json_filepath):
            print(f"STAC metadata file already exists. Reading from: {json_filepath}")
            with open(json_filepath, 'r') as f:
                stac_item = json.load(f)
        else:
            print(f"Fetching STAC metadata from: {stac_json_url}")
            response = requests.get(stac_json_url)
            response.raise_for_status()
            stac_item = response.json()
            with open(json_filepath, 'w') as f:
                json.dump(stac_item, f, indent=4)
            print(f"STAC metadata saved to: {json_filepath}")

        print(f"Found STAC Item: {stac_item.get('id', 'N/A')}")
        
        assets_to_download = parse_stac_item_for_assets(stac_item, stac_json_url)

        if assets_to_download:
            print(f"Found {len(assets_to_download)} asset(s) to download.")
            get_asset_info_from_stac_url.last_successful_url = stac_json_url
            return assets_to_download, stac_item
        else:
            print("Could not find the main Cloud Optimized GeoTIFF data asset.")
            return [], stac_item

    except requests.exceptions.RequestException as e:
        print(f"Failed to fetch or parse STAC JSON URL: {e}")
        return [], None
    except json.JSONDecodeError:
        print("Failed to decode JSON from the provided URL.")
        return [], None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return [], None


def download_file(url, filename, download_directory=DOWNLOAD_DIR):
    """Downloads a large file with a progress bar."""
    filepath = os.path.join(download_directory, filename)
    if os.path.exists(filepath):
        print(f"File already exists: {filepath}")
        return filepath

    print(f"Downloading {filename}...")
    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            total_size = int(r.headers.get('content-length', 0))
            bytes_downloaded = 0
            with open(filepath, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
                    bytes_downloaded += len(chunk)
                    done = int(50 * bytes_downloaded / total_size) if total_size > 0 else 0
                    print(f"\r[{'=' * done}{' ' * (50-done)}] {bytes_downloaded/1e6:.2f} / {total_size/1e6:.2f} MB", end='')
        print(f"\nSuccessfully downloaded to {filepath}")
        return filepath
    except requests.exceptions.RequestException as e:
        print(f"\nError downloading file: {e}")
        return None

def integer_normalize(band_array, num_values=256):
    """
    Normalizes an array to 0-(num_values-1) for display, correctly handling NaNs.
    """
    max_val = np.nanmax(band_array)
    if max_val == 0 or np.isnan(max_val):
        max_val = 1
    normalized = (band_array / max_val * (num_values - 1))
    return np.nan_to_num(normalized, nan=0).astype(np.uint8)

def read_geotiff_image(filepath):
    """
    Reads a GeoTIFF file, processes it for display, and returns both the
    image data array and the GeoTIFF's metadata.
    """
    try:
        print("Opening GeoTIFF file. This may take a moment...")
        with rasterio.open(filepath) as src:
            geotiff_metadata = src.profile
            nodata_val = src.nodata
            image_cube = src.read()
            
            if nodata_val is not None:
                print(f"Found no-data value: {nodata_val}. Masking for display.")
                # Convert to a float type that supports NaN if not already
                if image_cube.dtype != np.float32 and image_cube.dtype != np.float64:
                    image_cube = image_cube.astype(np.float32)
                # Replace the no-data value with Not a Number (NaN)
                image_cube[image_cube == nodata_val] = np.nan

            num_bands, height, width = image_cube.shape
            print(f"Image loaded: {width}x{height} pixels with {num_bands} bands.")
            return image_cube, geotiff_metadata
            
    except Exception as e:
        print(f"Failed to read file: {e}")
        return None, None

def read_hdf5_image(filepath):
    """
    Reads a hyperspectral image cube from an HDF5 file.
    If the data is 4D (contains multiple frames), it prompts the user to select one.
    Assumes the data is stored in a dataset named 'image_stack'.
    """
    try:
        print(f"Opening HDF5 file: {filepath}")
        with h5py.File(filepath, 'r') as hf:
            if 'image_stack' in hf:
                dataset = hf['image_stack']
                shape = dataset.shape
                
                image_cube = None

                if len(shape) == 4: # Has frames: (frames, bands, height, width)
                    num_frames = shape[0]
                    print(f"This HDF5 file contains {num_frames} frames.")
                    
                    try:
                        frame_choice = int(input(f"Please select a frame to view (1-{num_frames}): ")) - 1
                        if not 0 <= frame_choice < num_frames:
                            raise ValueError("Frame choice out of range.")
                        
                        print(f"Loading frame {frame_choice + 1}...")
                        image_cube = dataset[frame_choice, :, :, :]

                    except (ValueError, IndexError):
                        print("Invalid frame selection. Exiting.")
                        return None, None

                elif len(shape) == 3: # Single frame: (bands, height, width)
                    print("Loading single frame HDF5 image...")
                    image_cube = dataset[:]
                
                else:
                    print(f"Error: Unsupported data shape in HDF5 file: {shape}")
                    print("Viewer only supports 3D (bands, h, w) or 4D (frames, bands, h, w) data.")
                    return None, None

                num_bands, height, width = image_cube.shape
                print(f"HDF5 Image loaded: {width}x{height} pixels with {num_bands} bands.")
                hdf5_metadata = {'driver': 'HDF5', 'count': num_bands, 'height': height, 'width': width, 'shape': shape}
                return image_cube, hdf5_metadata
            else:
                print(f"Error: Could not find dataset 'image_stack' in {filepath}")
                print(f"Available datasets: {list(hf.keys())}")
                return None, None
    except Exception as e:
        print(f"Failed to read HDF5 file: {e}")
        return None, None

def create_interactive_viewer(image_cube, filename):
    """
    Displays a hyperspectral image array with sliders and a spectral plot.
    """
    if image_cube is None: return

    num_bands, height, width = image_cube.shape
    print("Normalizing image data for display...")
    normalized_cube = np.zeros((num_bands, height, width), dtype=np.uint8)
    for i in range(num_bands):
        normalized_cube[i, :, :] = integer_normalize(image_cube[i, :, :])
    print("Normalization complete.")

    fig, (ax_image, ax_spectrum) = plt.subplots(1, 2, figsize=(12, 7), 
                                                gridspec_kw={'width_ratios': [3, 1], 'wspace': 0.3})
    plt.subplots_adjust(bottom=0.25)

    ax_image.set_title(f'{filename}\n(Click on a pixel to view its spectrum)')
    ax_image.axis('off')
    image_display = ax_image.imshow(np.zeros((height, width, 3), dtype=np.float32))

    ax_spectrum.set_title("Pixel Spectrum")
    ax_spectrum.set_xlabel("Band Number")
    ax_spectrum.set_ylabel("Raw Value")
    spectrum_line, = ax_spectrum.plot([], [])
    ax_spectrum.set_xlim(1, num_bands)
    ax_spectrum.grid(True)

    if np.issubdtype(image_cube.dtype, np.integer):
        dtype_info = np.iinfo(image_cube.dtype)
        ax_spectrum.set_ylim(dtype_info.min, dtype_info.max)
    elif np.issubdtype(image_cube.dtype, np.floating):
        min_val, max_val = np.nanmin(image_cube), np.nanmax(image_cube)
        #ax_spectrum.set_ylim(min_val, max_val)  #Outlier values cause issues
        ax_spectrum.set_ylim(0,150)

    ax_red = plt.axes([0.15, 0.1, 0.5, 0.03])
    ax_green = plt.axes([0.15, 0.05, 0.5, 0.03])
    ax_blue = plt.axes([0.15, 0.0, 0.5, 0.03])
    ax_brightness = plt.axes([0.9, 0.25, 0.02, 0.6]) # Vertical slider on the right

    slider_red = Slider(
        ax=ax_red, label='Red Band', valmin=1, valmax=num_bands,
        valinit=INITIAL_BANDS['red'], valstep=1, color='red'
    )
    slider_green = Slider(
        ax=ax_green, label='Green Band', valmin=1, valmax=num_bands,
        valinit=INITIAL_BANDS['green'], valstep=1, color='green'
    )
    slider_blue = Slider(
        ax=ax_blue, label='Blue Band', valmin=1, valmax=num_bands,
        valinit=INITIAL_BANDS['blue'], valstep=1, color='blue'
    )

    slider_brightness = Slider(
        ax=ax_brightness, label='Bright', valmin=0, valmax=3,
        valinit=1, orientation='vertical'
    )

    def update_image_display(val):
        """Function to be called when a slider value is changed."""
        r_band_idx = int(slider_red.val) - 1
        g_band_idx = int(slider_green.val) - 1
        b_band_idx = int(slider_blue.val) - 1

        r_norm = normalized_cube[r_band_idx, :, :]
        g_norm = normalized_cube[g_band_idx, :, :]
        b_norm = normalized_cube[b_band_idx, :, :]

        # Combine bands and apply brightness
        rgb_image_float = np.dstack((r_norm, g_norm, b_norm)).astype(np.float32)
        
        brightness = slider_brightness.val
        # Apply brightness. Clip to ensure values are in the valid 0-255 range.
        brightened_image = np.clip(rgb_image_float * brightness, 0, 255)
        
        # Update the image data and redraw the plot
        image_display.set_data(brightened_image.astype(np.uint8))
        fig.canvas.draw_idle()

    slider_red.on_changed(update_image_display)
    slider_green.on_changed(update_image_display)
    slider_blue.on_changed(update_image_display)
    slider_brightness.on_changed(update_image_display)
    
    # Variable to store the selection box patch
    selected_pixel_box = None
    
    selected_pixel_box = None
    def on_pixel_click(event):
        """Callback function to handle mouse clicks on the image."""
        nonlocal selected_pixel_box
        if event.inaxes == ax_image and event.button == 1:
            x = int(event.xdata)
            y = int(event.ydata)

            if 0 <= x < width and 0 <= y < height:
                pixel_spectrum = image_cube[:, y, x]
                
                # Update console
                print("\n" + "="*40)
                print(f"Spectral data for pixel at (x={x}, y={y}):")
                print(pixel_spectrum)
                print("="*40)
                
                # Update the spectral plot
                band_numbers = np.arange(1, num_bands + 1)
                spectrum_line.set_data(band_numbers, pixel_spectrum)
                ax_spectrum.relim()
                ax_spectrum.autoscale_view(True, True, True)

                # Remove the old selection box if it exists
                if selected_pixel_box:
                    selected_pixel_box.remove()

                # Create and add a new selection box
                rect = Rectangle((x - 0.5, y - 0.5), 1, 1, 
                                 linewidth=1, edgecolor='r', facecolor='none')
                ax_image.add_patch(rect)
                selected_pixel_box = rect
                
                fig.canvas.draw_idle()

    fig.canvas.mpl_connect('button_press_event', on_pixel_click)
    update_image_display(None)
    plt.show()

def main():
    """Main function to run the application."""
    print("--- Interactive Hyperspectral Data Viewer ---")
    create_download_dir(DOWNLOAD_DIR)
    
    file_type_choice = input("Select data type to view: (S)TAC GeoTIFF or (H)DF5? ").lower()

    if file_type_choice == 's':
        stac_item = None
        main_image_filename = None
        assets_to_download = []
        
        choice = input("Select STAC source: (L)ocal file or (U)RL? ").lower()
        if choice == 'u':
            stac_url = input("Please enter the URL to a STAC Item .json file: ")
            if not stac_url: return print("No URL provided. Exiting.")
            assets_found, stac_item = get_asset_info_from_stac_url(stac_url, DOWNLOAD_DIR)
            if not assets_found: return print("Could not find any assets from the STAC URL. Exiting.")
            main_image_filename = assets_found[0]['filename']
            assets_to_download = assets_found
        elif choice == 'l':
            try:
                json_files = [f for f in os.listdir(DOWNLOAD_DIR) if f.endswith('.json')]
                if not json_files: return print(f"No .json files found in '{DOWNLOAD_DIR}'.")
                print("\nAvailable local STAC files:")
                for i, fname in enumerate(json_files): print(f"  {i+1}: {fname}")
                file_choice = int(input(f"Select a file number (1-{len(json_files)}): ")) - 1
                if not 0 <= file_choice < len(json_files): raise ValueError("Choice out of range.")
                
                json_path = os.path.join(DOWNLOAD_DIR, json_files[file_choice])
                with open(json_path, 'r') as f: stac_item = json.load(f)
                base_url = stac_item.get('assets', {}).get('stac_metadata', {}).get('href', "file:///")
                assets_found = parse_stac_item_for_assets(stac_item, base_url)
                if not assets_found: return print("Could not find a valid GeoTIFF asset in the JSON file.")
                main_image_filename = assets_found[0]['filename']
                for asset in assets_found:
                    if not os.path.exists(os.path.join(DOWNLOAD_DIR, asset['filename'])):
                        assets_to_download.append(asset)
            except (ValueError, IndexError): return print("Invalid selection. Exiting.")
            except Exception as e: return print(f"An error occurred: {e}")
        else: return print("Invalid choice. Exiting.")
        
        if assets_to_download:
            print("\nThe following files are required:")
            for asset in assets_to_download: print(f" - {asset['filename']} ({asset['title']})")
            if input(f"Download all {len(assets_to_download)} file(s)? (y/n): ").lower() == 'y':
                for asset in assets_to_download: download_file(asset['url'], asset['filename'], DOWNLOAD_DIR)
            elif main_image_filename in [a['filename'] for a in assets_to_download]:
                return print("Main image is missing and download was cancelled. Exiting.")

        filepath = os.path.join(DOWNLOAD_DIR, main_image_filename)
        if os.path.exists(filepath):
            hyper_cube, geotiff_meta = read_geotiff_image(filepath)
            if hyper_cube is not None and stac_item and geotiff_meta:
                geospatial_data_package = (hyper_cube, stac_item, geotiff_meta)
                print("\nSuccessfully created data package tuple.")
                create_interactive_viewer(hyper_cube, os.path.basename(filepath))
        else: print(f"Main image file '{main_image_filename}' not found.")

    elif file_type_choice == 'h':
        try:
            hdf5_files = [f for f in os.listdir(DOWNLOAD_DIR) if f.endswith(('.hdf5', '.h5'))]
            if not hdf5_files: return print(f"No .hdf5 or .h5 files found in '{DOWNLOAD_DIR}'.")
            print("\nAvailable local HDF5 files:")
            for i, fname in enumerate(hdf5_files): print(f"  {i+1}: {fname}")
            file_choice = int(input(f"Select a file number (1-{len(hdf5_files)}): ")) - 1
            if not 0 <= file_choice < len(hdf5_files): raise ValueError("Choice out of range.")
            
            filepath = os.path.join(DOWNLOAD_DIR, hdf5_files[file_choice])
            hyper_cube, hdf5_meta = read_hdf5_image(filepath)
            if hyper_cube is not None:
                stac_placeholder = {"id": os.path.basename(filepath)}
                geospatial_data_package = (hyper_cube, stac_placeholder, hdf5_meta)
                print("\nSuccessfully created data package tuple.")
                create_interactive_viewer(hyper_cube, os.path.basename(filepath))
        except (ValueError, IndexError): return print("Invalid selection. Exiting.")
        except Exception as e: return print(f"An error occurred: {e}")

    else:
        print("Invalid choice. Please enter 'S' or 'H'. Exiting.")

if __name__ == "__main__":
    main()
