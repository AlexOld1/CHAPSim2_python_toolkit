import vtk
import numpy as np
from vtkmodules.util.numpy_support import vtk_to_numpy
import pyvista as pv

def data_filepath(folder_path, case, quantity, timestep):
    return f'{folder_path}{case}/1_data/domain1_time_space_averaged_{quantity}_{timestep}.dat'

def load_ts_avg_data(data_filepath):
    try:
        return np.loadtxt(data_filepath)
    except OSError:
        print(f'Error loading data for {data_filepath}')
        return None

def visu_file_paths(folder_path, case, timestep):
    file_names = [
        f'{folder_path}{case}/2_visu/domain1_flow_{timestep}.xdmf',
        f'{folder_path}{case}/2_visu/domain1_time_averaged_flow_{timestep}.xdmf',
        f'{folder_path}{case}/2_visu/domain1_mhd_{timestep}.xdmf'
    ]
    return file_names

def read_xdmf_extract_numpy_arrays(file_names):
    """
    Reads XDMF files and extracts numpy arrays from VTK data.
    
    Args:
        folder_path (str): Path to the folder containing XDMF files
        timestep (str): Timestep identifier
    
    Returns:
        tuple: (all_numpy_arrays dict, grid_info dict)
    """

    all_numpy_arrays = {}
    grid_info = {}
    
    for xdmf_file in file_names:
        try:
            print(f"Opening file: {xdmf_file}")
            
            # Create an XdmfReader
            reader = vtk.vtkXdmfReader()
            reader.SetFileName(xdmf_file)
            
            # Try to update the reader - catch XML parsing errors
            try:
                reader.Update()
                output = reader.GetOutput()
            except Exception as xml_error:
                print(f"XML parsing error in {xdmf_file}: {str(xml_error)}")
                continue
            
            if output and output.GetNumberOfCells() > 0:
                # Extract file type from filename for prefixing
                if 'time_averaged' in xdmf_file:
                    file_type = 'time_averaged'
                elif '_mhd_' in xdmf_file:
                    file_type = 'mhd'  
                else:
                    file_type = 'flow'
                
                # Extract grid information if not already done
                if not grid_info:
                    grid_info = extract_grid_info(output)
                    print(f"Grid info: {grid_info}")
                
                # Get arrays from the dataset
                dataset_arrays = get_vtk_arrays_with_numpy(output, file_type, grid_info)
                all_numpy_arrays.update(dataset_arrays)
                print(f"Successfully extracted {len(dataset_arrays)} arrays from {file_type} file")
            else:
                print(f"Warning: No valid output from {xdmf_file}, file missing or empty")
                
        except Exception as e:
            print(f"Error processing {xdmf_file}: {str(e)}")
            continue
    
    return all_numpy_arrays, grid_info

def extract_grid_info(dataset):
    """
    Extract grid dimensions and spacing information from VTK dataset.
    
    Args:
        dataset: VTK dataset object
    
    Returns:
        dict: Grid information including dimensions and spacing
    """
    grid_info = {}
    
    try:
        # For structured grids
        if hasattr(dataset, 'GetDimensions'):
            dims = dataset.GetDimensions()
            grid_info['node_dimensions'] = dims
            grid_info['cell_dimensions'] = (dims[0]-1, dims[1]-1, dims[2]-1)
            print(f"Node dimensions: {dims}")
            print(f"Cell dimensions: {grid_info['cell_dimensions']}")
            
        # Get bounds
        if hasattr(dataset, 'GetBounds'):
            bounds = dataset.GetBounds()
            grid_info['bounds'] = bounds
            print(f"Domain bounds: x=[{bounds[0]:.3f}, {bounds[1]:.3f}], "
                  f"y=[{bounds[2]:.3f}, {bounds[3]:.3f}], z=[{bounds[4]:.3f}, {bounds[5]:.3f}]")
        
        # Calculate average spacing if possible
        if 'node_dimensions' in grid_info and 'bounds' in grid_info:
            dx = (bounds[1] - bounds[0]) / (dims[0] - 1)
            dy = (bounds[3] - bounds[2]) / (dims[1] - 1)
            dz = (bounds[5] - bounds[4]) / (dims[2] - 1)
            grid_info['average_spacing'] = (dx, dy, dz)
            print(f"Average Grid spacing: dx={dx:.6f}, dy={dy:.6f}, dz={dz:.6f}")
            
    except Exception as e:
        print(f"Could not extract complete grid info: {str(e)}")
        
    return grid_info

def get_vtk_arrays_with_numpy(dataset, file_type="", grid_info=None):
    """
    Extracts VTK arrays and converts them to NumPy arrays.
    
    Args:
        dataset: VTK dataset object
        file_type (str): Type of file for prefixing array names
        grid_info (dict): Grid information for reshaping arrays
    
    Returns:
        dict: Dictionary of numpy arrays
    """
    arrays = {}
    
    # Process Point Data
    if dataset.GetPointData() and dataset.GetPointData().GetNumberOfArrays() > 0:
        print(f"\nPoint Data Arrays ({file_type}):")
        for i in range(dataset.GetPointData().GetNumberOfArrays()):
            array = dataset.GetPointData().GetArray(i)
            name = array.GetName()
            numpy_array = vtk_to_numpy(array)
            
            # Reshape to 3D if grid info available
            if grid_info and 'node_dimensions' in grid_info:
                try:
                    dims = grid_info['node_dimensions']
                    if numpy_array.size == dims[0] * dims[1] * dims[2]:
                        numpy_array = numpy_array.reshape(dims[2], dims[1], dims[0])  # VTK uses different ordering
                        print(f"  {name}: Shape - {numpy_array.shape} (3D), Type - {numpy_array.dtype}")
                    else:
                        print(f"  {name}: Shape - {numpy_array.shape} (1D - size mismatch), Type - {numpy_array.dtype}")
                except:
                    print(f"  {name}: Shape - {numpy_array.shape} (1D - reshape failed), Type - {numpy_array.dtype}")
            else:
                print(f"  {name}: Shape - {numpy_array.shape} (1D), Type - {numpy_array.dtype}")
            
            key = f"{file_type}_point_{name}" if file_type else f"point_{name}"
            arrays[key] = numpy_array
    
    # Process Cell Data
    if dataset.GetCellData() and dataset.GetCellData().GetNumberOfArrays() > 0:
        print(f"\nCell Data Arrays ({file_type}):")
        for i in range(dataset.GetCellData().GetNumberOfArrays()):
            array = dataset.GetCellData().GetArray(i)
            name = array.GetName()
            numpy_array = vtk_to_numpy(array)
            
            # Reshape to 3D if grid info available
            if grid_info and 'cell_dimensions' in grid_info:
                try:
                    dims = grid_info['cell_dimensions']
                    if numpy_array.size == dims[0] * dims[1] * dims[2]:
                        numpy_array = numpy_array.reshape(dims[2], dims[1], dims[0])  # VTK uses different ordering
                        print(f"  {name}: Shape - {numpy_array.shape} (3D), Type - {numpy_array.dtype}")
                    else:
                        print(f"  {name}: Shape - {numpy_array.shape} (1D - size mismatch), Type - {numpy_array.dtype}")
                except:
                    print(f"  {name}: Shape - {numpy_array.shape} (1D - reshape failed), Type - {numpy_array.dtype}")
            else:
                print(f"  {name}: Shape - {numpy_array.shape} (1D), Type - {numpy_array.dtype}")
            
            key = f"{file_type}_cell_{name}" if file_type else f"cell_{name}"
            arrays[key] = numpy_array
    
    # Extract geometry coordinates
    try:
        if hasattr(dataset, 'GetPoints') and dataset.GetPoints() is not None:
            points = dataset.GetPoints()
            points_array = vtk_to_numpy(points.GetData())
            
            # Try to reshape coordinates to 3D grid
            if grid_info and 'node_dimensions' in grid_info:
                try:
                    dims = grid_info['node_dimensions']
                    if points_array.shape[0] == dims[0] * dims[1] * dims[2]:
                        # Separate x, y, z coordinates and reshape each
                        x_coords = points_array[:, 0].reshape(dims[2], dims[1], dims[0])
                        y_coords = points_array[:, 1].reshape(dims[2], dims[1], dims[0])
                        z_coords = points_array[:, 2].reshape(dims[2], dims[1], dims[0])
                        
                        key_prefix = f"{file_type}_geometry" if file_type else "geometry"
                        arrays[f"{key_prefix}_x"] = x_coords
                        arrays[f"{key_prefix}_y"] = y_coords
                        arrays[f"{key_prefix}_z"] = z_coords
                        
                        print(f"\nGeometry Coordinates: X, Y, Z each with shape {x_coords.shape} (3D)")
                    else:
                        key = f"{file_type}_geometry_points" if file_type else "geometry_points"
                        arrays[key] = points_array
                        print(f"\nGeometry Points: Shape - {points_array.shape} (1D)")
                except Exception as reshape_error:
                    key = f"{file_type}_geometry_points" if file_type else "geometry_points"
                    arrays[key] = points_array
                    print(f"\nGeometry Points: Shape - {points_array.shape} (reshape failed: {str(reshape_error)})")
            else:
                key = f"{file_type}_geometry_points" if file_type else "geometry_points"
                arrays[key] = points_array
                print(f"\nGeometry Points: Shape - {points_array.shape}")
                
    except Exception as e:
        print(f"Could not extract geometry points: {str(e)}")
        # Try alternative method for geometry
        try:
            if hasattr(dataset, 'GetNumberOfPoints') and dataset.GetNumberOfPoints() > 0:
                print(f"Dataset has {dataset.GetNumberOfPoints()} points but could not extract coordinates")
        except:
            pass
    
    return arrays

def reader_output_summary(arrays_dict):
    """
    Provides a summary analysis of the extracted arrays.
    
    Args:
        arrays_dict (dict): Dictionary of numpy arrays
    """
    print("\n" + "="*60)
    print("READER OUTPUT SUMMARY")
    print("="*60)
    
    for key, array in arrays_dict.items():
        print(f"{key}:")
        print(f"  Shape: {array.shape},  Min value: {np.min(array):.6e},  Max value: {np.max(array):.6e}   Mean value: {np.mean(array):.6e}")
        print("-" * 40)

def visualise_domain_qx(output, all_numpy_arrays):
    pv_mesh = pv.wrap(output)

    if pv_mesh:
                # Let's assume you want to plot the 'qx_velocity' cell data
                    if "cell_data_qx_velocity" in all_numpy_arrays:
                        qx_velocity = all_numpy_arrays["cell_data_qx_velocity"]

                        # Add the NumPy array as cell data to the PyVista mesh
                        pv_mesh.cell_data["qx_velocity"] = qx_velocity

                        # Plot the mesh, coloring by the 'qx_velocity' cell data
                        plotter = pv.Plotter()
                        plotter.add_mesh(pv_mesh, scalars="qx_velocity", show_edges=True, cmap="viridis")
                        plotter.show()
                    else:
                        print("Error: 'cell_data_qx_velocity' not found in the extracted arrays.")
    else:
                    print("Error: Could not wrap VTK output to PyVista mesh.")