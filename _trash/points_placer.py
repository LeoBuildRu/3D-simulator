"""
Simple interactive Python utility to place exactly 4 keypoints on a .png image
and save them into a .json file. Repeat for additional images.

Features:
- Open a .png file
- Click 4 points on the image (canvas will show points and order numbers)
- Save keypoints to a JSON next to the image (default) or choose save path
- Reset points or open another image and repeat

Dependencies: Pillow (PIL). tkinter is used from the stdlib.
Install Pillow with: pip install pillow

Run: python image_keypoint_labeler.py

JSON format produced (example):
{
  "image": "C:/path/to/image.png",
  "image_size": [1920, 1080],
  "display_size": [960, 540],
  "scale": 0.5,
  "points": [
    {"x": 120, "y": 300},
    {"x": 400, "y": 310},
    {"x": 700, "y": 200},
    {"x": 800, "y": 600}
  ],
  "timestamp": "2026-03-02T12:34:56"
}

Behavior notes:
- Clicks are converted to original image coordinates (so saved points map to the full-resolution image).
- After 4 clicks the UI will auto-enable the Save button; you can still Reset if you want to re-place points.
- Use the buttons for Open, Reset, Save, Quit.

"""

import json
import os
from datetime import datetime
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

MAX_DISPLAY_W = 1200
MAX_DISPLAY_H = 800
POINT_RADIUS = 4


class KeypointLabeler:
    def __init__(self, root):
        self.root = root
        root.title("PNG 4-Keypoint Labeler")

        self.image_path = None
        self.orig_image = None  # PIL Image at original size
        self.display_image = None  # PIL Image resized for display
        self.tk_image = None
        self.scale = 1.0
        self.points = []  # list of (orig_x, orig_y)
        self.point_handles = []  # canvas ids for drawings

        # Top frame for buttons
        top = tk.Frame(root)
        top.pack(side=tk.TOP, fill=tk.X, padx=6, pady=6)

        btn_open = tk.Button(top, text="Open .png", command=self.open_image)
        btn_open.pack(side=tk.LEFT, padx=4)

        self.btn_reset = tk.Button(top, text="Reset Points", command=self.reset_points, state=tk.DISABLED)
        self.btn_reset.pack(side=tk.LEFT, padx=4)

        self.btn_save = tk.Button(top, text="Save JSON", command=self.save_json, state=tk.DISABLED)
        self.btn_save.pack(side=tk.LEFT, padx=4)

        btn_quit = tk.Button(top, text="Quit", command=root.quit)
        btn_quit.pack(side=tk.RIGHT, padx=4)

        # Info / instructions
        self.info_var = tk.StringVar()
        self.info_var.set("Open a .png file to begin. Click 4 points on the image.")
        info_label = tk.Label(root, textvariable=self.info_var)
        info_label.pack(side=tk.TOP, fill=tk.X, padx=6)

        # Canvas for image display
        self.canvas = tk.Canvas(root, bg="black")
        self.canvas.pack(expand=True, fill=tk.BOTH)
        self.canvas.bind("<Button-1>", self.on_click)

        # Right-side frame to show points list
        right = tk.Frame(root)
        right.pack(side=tk.RIGHT, fill=tk.Y, padx=6, pady=6)
        tk.Label(right, text="Points (original image coords):").pack(anchor=tk.NW)
        self.points_text = tk.Text(right, width=28, height=10, state=tk.DISABLED)
        self.points_text.pack()

    def open_image(self):
        filepath = filedialog.askopenfilename(filetypes=[("PNG images", "*.png")])
        if not filepath:
            return
        try:
            img = Image.open(filepath).convert("RGBA")
        except Exception as e:
            messagebox.showerror("Open image", f"Failed to open image:\n{e}")
            return

        self.image_path = filepath
        self.orig_image = img
        self.points = []
        self.point_handles = []

        ow, oh = img.size
        # compute scale to fit display
        scale = min(1.0, MAX_DISPLAY_W / ow, MAX_DISPLAY_H / oh)
        self.scale = scale
        dw = int(ow * scale)
        dh = int(oh * scale)
        if scale < 1.0:
            disp = img.resize((dw, dh), Image.LANCZOS)
        else:
            disp = img.copy()

        self.display_image = disp
        self.tk_image = ImageTk.PhotoImage(disp)

        # resize canvas to image size
        self.canvas.config(width=dw, height=dh)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)

        self.btn_reset.config(state=tk.NORMAL)
        self.btn_save.config(state=tk.DISABLED)

        self.info_var.set(f"Opened: {os.path.basename(filepath)} — click 4 points (scale={self.scale:.3f}).")
        self.update_points_display()

    def on_click(self, event):
        if self.orig_image is None:
            return
        if len(self.points) >= 4:
            messagebox.showinfo("4 points placed", "You already placed 4 points. Reset to place again or Save JSON.")
            return

        # canvas coords
        cx, cy = event.x, event.y
        # clamp to image bounds
        dw, dh = self.display_image.size
        cx = max(0, min(cx, dw - 1))
        cy = max(0, min(cy, dh - 1))

        # convert to original coords
        ox = int(round(cx / self.scale))
        oy = int(round(cy / self.scale))
        self.points.append((ox, oy))

        # draw a small circle and label on canvas (use display coords)
        handle = self.draw_point(cx, cy, len(self.points))
        self.point_handles.append(handle)

        self.info_var.set(f"Placed point {len(self.points)}/4 at ({ox}, {oy}).")
        self.update_points_display()

        if len(self.points) == 4:
            self.btn_save.config(state=tk.NORMAL)
            messagebox.showinfo("4 points placed", "4 points placed. Click Save JSON to export or Reset to redo.")

    def draw_point(self, cx, cy, idx):
        r = POINT_RADIUS
        oval = self.canvas.create_oval(cx - r, cy - r, cx + r, cy + r, outline="yellow", width=2)
        text = self.canvas.create_text(cx + r + 6, cy - r - 6, text=str(idx), anchor=tk.NW, fill="yellow", font=(None, 12, "bold"))
        return (oval, text)

    def reset_points(self):
        self.points = []
        # delete handles
        for h in self.point_handles:
            for id_ in h:
                try:
                    self.canvas.delete(id_)
                except Exception:
                    pass
        self.point_handles = []
        self.btn_save.config(state=tk.DISABLED)
        if self.orig_image:
            self.info_var.set(f"Points reset — click 4 points on {os.path.basename(self.image_path)}.")
        else:
            self.info_var.set("Points reset.")
        self.update_points_display()

    def update_points_display(self):
        self.points_text.config(state=tk.NORMAL)
        self.points_text.delete("1.0", tk.END)
        for i, (x, y) in enumerate(self.points, start=1):
            self.points_text.insert(tk.END, f"{i}: ({x}, {y})\n")
        self.points_text.config(state=tk.DISABLED)

    def save_json(self):
        if len(self.points) != 4:
            messagebox.showwarning("Not enough points", "Place exactly 4 points before saving.")
            return

        # default path: same folder, same base name + _keypoints.json
        base_dir = os.path.dirname(self.image_path)
        base_name = os.path.splitext(os.path.basename(self.image_path))[0]
        default_name = os.path.join(base_dir, base_name + "_keypoints.json")

        save_path = filedialog.asksaveasfilename(title="Save keypoints JSON",
                                                 defaultextension=".json",
                                                 initialfile=os.path.basename(default_name),
                                                 initialdir=base_dir,
                                                 filetypes=[("JSON files", "*.json")])
        if not save_path:
            return

        ow, oh = self.orig_image.size
        dw, dh = self.display_image.size
        data = {
            "image": os.path.abspath(self.image_path),
            "image_size": [ow, oh],
            "display_size": [dw, dh],
            "scale": float(self.scale),
            "points": [{"x": int(x), "y": int(y)} for (x, y) in self.points],
            "timestamp": datetime.utcnow().isoformat()
        }

        try:
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            messagebox.showerror("Save JSON", f"Failed to save JSON:\n{e}")
            return

        messagebox.showinfo("Saved", f"Saved keypoints to:\n{save_path}")
        # optionally automatically reset and allow user to open another image
        self.reset_points()


if __name__ == "__main__":
    root = tk.Tk()
    app = KeypointLabeler(root)
    root.mainloop()
