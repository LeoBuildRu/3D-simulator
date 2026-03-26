import open3d as o3d
from pathlib import Path

# Define input and output directories
input_folder = Path("ToConvert")
output_folder = Path("Output")

# Create output folder if it doesn't exist
output_folder.mkdir(parents=True, exist_ok=True)

# Recursively iterate over all .txt files
for txt_file in input_folder.rglob("*.txt"):
    try:
        # Get path relative to input folder
        relative_path = txt_file.relative_to(input_folder)
        
        # Extract top-level folder (first part of the path)
        top_folder = relative_path.parts[0] if len(relative_path.parts) > 1 else "root"
        
        # Create corresponding output subfolder
        target_folder = output_folder / top_folder
        target_folder.mkdir(parents=True, exist_ok=True)
        
        # Output file goes into that folder (flattened)
        output_file = target_folder / f"{txt_file.stem}.ply"
        
        # Read point cloud
        pcd = o3d.io.read_point_cloud(str(txt_file), format="xyz")
        
        # Write to PLY
        o3d.io.write_point_cloud(str(output_file), pcd)
        
        print(f"✓ Converted: {txt_file} → {output_file}")
        
    except Exception as e:
        print(f"✗ Error converting {txt_file}: {e}")