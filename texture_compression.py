import os
from pathlib import Path
from PIL import Image
import shutil

# =========================
# SETTINGS
# =========================

ROOT_FOLDER = r"G:\IQoko\bruuuh\gltf_optimized"   # Folder A
TARGET_MB_PER_C_FOLDER = 19

# texture priorities
HIGH_QUALITY_KEYWORDS = [
    "color",
    "albedo",
    "diffuse",
    "basecolor"
]

LOW_QUALITY_KEYWORDS = [
    "roughness",
    "specular",
    "metallic",
    "metalness",
    "ao",
    "ambientocclusion",
    "normal",
    "height",
    "displacement",
    "mask"
]

# backup originals?
CREATE_BACKUP = False

# =========================
# HELPERS
# =========================

def folder_size_mb(folder):
    total = 0
    for root, _, files in os.walk(folder):
        for f in files:
            fp = os.path.join(root, f)
            total += os.path.getsize(fp)
    return total / (1024 * 1024)

def is_high_quality(name):
    n = name.lower()
    return any(k in n for k in HIGH_QUALITY_KEYWORDS)

def is_low_quality(name):
    n = name.lower()
    return any(k in n for k in LOW_QUALITY_KEYWORDS)

def compress_png(path, aggressive=False, very_aggressive=False):
    img = Image.open(path)

    # Preserve alpha if present
    has_alpha = img.mode in ("RGBA", "LA") or (
        img.mode == "P" and "transparency" in img.info
    )

    # Convert modes safely
    if has_alpha:
        img = img.convert("RGBA")
    else:
        img = img.convert("RGB")

    width, height = img.size

    # =========================
    # Resize logic
    # =========================

    scale = 1.0

    if very_aggressive:
        scale = 0.5
    elif aggressive:
        scale = 0.75

    if scale < 1.0:
        new_size = (
            max(1, int(width * scale)),
            max(1, int(height * scale))
        )
        img = img.resize(new_size, Image.LANCZOS)

    # =========================
    # PNG palette reduction
    # =========================

    if very_aggressive:
        colors = 32
    elif aggressive:
        colors = 128
    else:
        colors = 256

    img = img.quantize(colors=colors)

    # Save optimized PNG
    img.save(
        path,
        optimize=True,
        compress_level=9
    )

def process_c_folder(folder):
    pngs = list(Path(folder).glob("*.png"))

    if not pngs:
        return

    print(f"\nProcessing: {folder}")

    # Optional backup
    if CREATE_BACKUP:
        backup_dir = Path(folder) / "_backup_originals"
        backup_dir.mkdir(exist_ok=True)

        for p in pngs:
            backup_path = backup_dir / p.name
            if not backup_path.exists():
                shutil.copy2(p, backup_path)

    # First pass
    for tex in pngs:
        name = tex.name.lower()

        if is_high_quality(name):
            compress_png(tex, aggressive=False)

        elif is_low_quality(name):
            compress_png(tex, aggressive=True)

        else:
            compress_png(tex, aggressive=True)

    current_size = folder_size_mb(folder)

    print(f"Current size: {current_size:.2f} MB")

    # Second pass if still too large
    if current_size > TARGET_MB_PER_C_FOLDER:
        print("Still too large -> aggressive recompression")

        low_priority = []

        for tex in pngs:
            if not is_high_quality(tex.name):
                low_priority.append(tex)

        # Compress low priority harder first
        for tex in low_priority:
            compress_png(tex, very_aggressive=True)

            current_size = folder_size_mb(folder)

            print(f"Size now: {current_size:.2f} MB")

            if current_size <= TARGET_MB_PER_C_FOLDER:
                break

    # Final emergency pass
    current_size = folder_size_mb(folder)

    if current_size > TARGET_MB_PER_C_FOLDER:
        print("Emergency pass on ALL textures")

        for tex in pngs:
            compress_png(tex, aggressive=True, very_aggressive=True)

            current_size = folder_size_mb(folder)

            print(f"Size now: {current_size:.2f} MB")

            if current_size <= TARGET_MB_PER_C_FOLDER:
                break

    print(f"Final size: {folder_size_mb(folder):.2f} MB")


def main():
    root = Path(ROOT_FOLDER)

    # Find all "C" folders automatically
    for dirpath, dirnames, filenames in os.walk(root):

        png_files = [f for f in filenames if f.lower().endswith(".png")]

        # If folder contains pngs -> treat as C folder
        if png_files:
            process_c_folder(dirpath)


if __name__ == "__main__":
    main()