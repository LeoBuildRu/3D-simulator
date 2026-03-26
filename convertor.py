import open3d as o3d
from pathlib import Path

# Define input and output directories
input_folder = Path("ToConvert")
output_folder = Path("Output")

# Create output folder if it doesn't exist
output_folder.mkdir(parents=True, exist_ok=True)

# Iterate over all .txt files in the input folder
for txt_file in input_folder.glob("*.txt"):
    # Construct output path: same name, but .ply extension
    output_file = output_folder / f"{txt_file.stem}.ply"
    
    try:
        # Read point cloud from XYZ-formatted text file
        pcd = o3d.io.read_point_cloud(str(txt_file), format="xyz")
        
        # Write to PLY format
        o3d.io.write_point_cloud(str(output_file), pcd)
        
        print(f"✓ Converted: {txt_file.name} → {output_file.name}")
        
    except Exception as e:
        print(f"✗ Error converting {txt_file.name}: {e}")