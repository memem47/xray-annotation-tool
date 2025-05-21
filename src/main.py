"""X‑ray Annotation Tool – single‑file prototype

Features
--------
1. Load real X‑ray images (8‑bit grayscale)
2. Generate synthetic phantom images that mimic X‑ray appearance
3. Free‑hand polygon annotation with left‑click vertices & right‑click close
4. Export binary mask (PNG) + vertex list (JSON)
"""
from __future__ import annotations

import json
import os
import tkinter as tk
from tkinter import filedialog, messagebox

import cv2
import numpy as np
from PIL import Image, ImageTk

from synthetic_xray import generate_phantom

class AnnotationApp(tk.Tk):
    """Main application window."""

    def __init__(self) -> None:
        super().__init__()
        self.title("X‑ray Annotation Tool")
        self.geometry("1024x768")

        # Internal state
        self.image: np.ndarray | None = None          # original grayscale image (uint8)
        self.photo: ImageTk.PhotoImage | None = None  # rendered Tk image
        self.display_size: tuple[int, int] | None = None  # (w, h) after scaling to canvas
        self.points: list[tuple[int, int]] = []
        self.mask: np.ndarray | None = None           # binary mask (same size as self.image)
        self.img_path: str | None = None

        # ---- UI ----
        self._build_toolbar()
        self.canvas = tk.Canvas(self, bg="black")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Mouse bindings (attach after canvas exists)
        self.canvas.bind("<Button-1>", self._on_left_click)
        self.canvas.bind("<Button-3>", self._on_right_click)

    # ------------------------------------------------------------------ UI helpers
    def _build_toolbar(self) -> None:
        bar = tk.Frame(self, bd=2, relief=tk.RIDGE)
        bar.pack(side=tk.TOP, fill=tk.X)
        tk.Button(bar, text="Open Image", command=self._open_image).pack(side=tk.LEFT)
        tk.Button(bar, text="Generate Sim", command=self._generate_sim).pack(side=tk.LEFT)
        tk.Button(bar, text="Clear Points", command=self._clear_points).pack(side=tk.LEFT)
        tk.Button(bar, text="Save Mask", command=self._save_mask).pack(side=tk.LEFT)

    # ------------------------------------------------------------------ core actions
    def _open_image(self) -> None:
        path = filedialog.askopenfilename(
            title="Choose X‑ray image",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.tif *.tiff")],
        )
        if path:
            self._load_image(path)

    def _load_image(self, path: str) -> None:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            messagebox.showerror("Error", f"Failed to read image: {path}")
            return
        self.image = img
        self.img_path = path
        self._display_image(img)

    def _generate_sim(self) -> None:
        """Create a quick synthetic phantom (512×512)"""
        self.image = generate_phantom(size=512)
        self.img_path = None
        self._display_image(self.image)

    # ------------------------------------------------------------------ drawing helpers
    def _display_image(self, img: np.ndarray) -> None:
        # Scale down if too large for screen (keep aspect)
        h, w = img.shape
        max_dim = 900
        scale = min(1.0, max_dim / max(h, w))
        disp_w, disp_h = int(w * scale), int(h * scale)
        self.display_size = (disp_w, disp_h)
        resized = cv2.resize(img, (disp_w, disp_h), interpolation=cv2.INTER_AREA)
        rgba = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGBA)
        pil_img = Image.fromarray(rgba)
        self.photo = ImageTk.PhotoImage(pil_img)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
        self.points.clear()
        self.mask = None

    def _on_left_click(self, event: tk.Event) -> None:
        if self.photo is None:
            return  # nothing displayed
        self.points.append((event.x, event.y))
        r = 3
        self.canvas.create_oval(event.x - r, event.y - r, event.x + r, event.y + r, fill="red")
        if len(self.points) > 1:
            self.canvas.create_line(*self.points[-2], *self.points[-1], fill="red", width=2)

    def _on_right_click(self, event: tk.Event) -> None:
        if len(self.points) < 3:
            messagebox.showinfo("Info", "Need at least 3 points to form a polygon.")
            return
        # close polygon visually
        self.canvas.create_line(*self.points[-1], *self.points[0], fill="red", width=2)
        self._create_mask_from_points()
        messagebox.showinfo("Mask", "Mask created – you can now click 'Save Mask'.")

    def _clear_points(self) -> None:
        if self.image is not None:
            self._display_image(self.image)

    # ------------------------------------------------------------------ mask export
    def _create_mask_from_points(self) -> None:
        if self.image is None or self.display_size is None:
            return
        disp_w, disp_h = self.display_size
        # Create mask in display resolution first
        disp_mask = np.zeros((disp_h, disp_w), dtype=np.uint8)
        cv2.fillPoly(disp_mask, [np.array(self.points, dtype=np.int32)], 255)
        # Rescale mask back to original resolution using nearest‑neighbor
        h0, w0 = self.image.shape
        self.mask = cv2.resize(disp_mask, (w0, h0), interpolation=cv2.INTER_NEAREST)

    def _save_mask(self) -> None:
        if self.mask is None:
            messagebox.showerror("Error", "No mask found – draw a polygon and right‑click first.")
            return
        default_name = "mask.png" if not self.img_path else os.path.splitext(os.path.basename(self.img_path))[0] + "_mask.png"
        path = filedialog.asksaveasfilename(defaultextension=".png", initialfile=default_name)
        if not path:
            return
        cv2.imwrite(path, self.mask)
        # Save raw vertices for reproducibility
        verts_path = os.path.splitext(path)[0] + ".json"
        with open(verts_path, "w", encoding="utf-8") as fp:
            json.dump(self.points, fp, indent=2)
        messagebox.showinfo("Saved", f"Mask PNG and vertices JSON saved to:\n{path}\n{verts_path}")

    # ------------------------------------------------------------------ synthetic image helper
    @staticmethod
    def _create_simulated_xray(size: int = 512) -> np.ndarray:
        """Very coarse phantom generator (replace with domain‑specific model if needed)."""
        rng = np.random.default_rng()
        base = rng.normal(loc=220, scale=10, size=(size, size)).astype(np.float32)
        # bones (bright regions)
        for _ in range(6):
            center = rng.integers(low=size * 0.2, high=size * 0.8, size=2)
            radius = rng.integers(low=size * 0.05, high=size * 0.12)
            cv2.circle(base, tuple(center), int(radius), 40 + rng.integers(10), -1)
        # soft tissue shadow
        cv2.GaussianBlur(base, (0, 0), sigmaX=5, dst=base)
        img_uint8 = np.clip(base, 0, 255).astype(np.uint8)
        return img_uint8


if __name__ == "__main__":
    AnnotationApp().mainloop()