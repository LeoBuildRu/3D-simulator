# panda_widget.py
import tkinter as tk
from tkinter import filedialog

class Panda3DWidget:
    def __init__(self):
        self.app = None
        self.tk_root = tk.Tk()
        self.tk_root.withdraw()
        self.tk_root.attributes('-topmost', True)
        
    def load_model_dialog(self):
        file_path = filedialog.askopenfilename(
            title="Select .gltf model",
            filetypes=[("GLTF files", "*.gltf"), ("GLB files", "*.glb"), ("All files", "*.*")]
        )
        
        if file_path:
            return file_path
        return None