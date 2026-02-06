from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def run_gui(initial_config: Dict[str, object]) -> Optional[Dict[str, object]]:
    """Launch a simple Tkinter GUI to collect run settings."""
    try:
        import tkinter as tk
        from tkinter import ttk
        from PIL import Image, ImageTk
    except Exception as exc:
        raise RuntimeError("Tkinter is required for GUI mode.") from exc

    root = tk.Tk()
    root.title("Active Learning")
    root.geometry("1200x700")

    fields = [
        ("project_name", "Project Name"),
        ("data_dir", "Data Dir"),
        ("model_dir", "Model Dir"),
        ("output_dir", "Output Dir"),
        ("round", "Round"),
        ("query_size", "Query Size"),
        ("strategy", "Strategy"),
        ("seed", "Seed"),
        ("device", "Device"),
        ("model_name", "Model Name"),
        ("tile_size", "Tile Size"),
    ]

    values: Dict[str, tk.StringVar] = {}
    for key, label in fields:
        values[key] = tk.StringVar(value=str(initial_config.get(key, "")))

    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=1)

    panels = ttk.Panedwindow(root, orient=tk.HORIZONTAL)
    panels.grid(row=0, column=0, sticky="nsew")

    left_panel = ttk.Frame(panels, padding=12)
    middle_panel = ttk.Frame(panels, padding=12)
    right_panel = ttk.Frame(panels, padding=12)

    panels.add(left_panel, weight=1)
    panels.add(middle_panel, weight=2)
    panels.add(right_panel, weight=1)

    left_panel.columnconfigure(1, weight=1)
    for row_idx, (key, label) in enumerate(fields):
        ttk.Label(left_panel, text=label).grid(row=row_idx, column=0, sticky="w", padx=4, pady=4)
        ttk.Entry(left_panel, textvariable=values[key], width=32).grid(
            row=row_idx, column=1, sticky="ew", padx=4, pady=4
        )

    dropdown_row = len(fields)
    ttk.Separator(left_panel).grid(row=dropdown_row, column=0, columnspan=2, sticky="ew", pady=8)
    ttk.Label(left_panel, text="Map Year").grid(
        row=dropdown_row + 1, column=0, sticky="w", padx=4, pady=4
    )
    year_var = tk.StringVar(value="")
    year_combo = ttk.Combobox(
        left_panel, textvariable=year_var, state="readonly", width=28, values=[]
    )
    year_combo.grid(row=dropdown_row + 1, column=1, sticky="ew", padx=4, pady=4)

    info_row = dropdown_row + 2
    ttk.Separator(left_panel).grid(row=info_row, column=0, columnspan=2, sticky="ew", pady=8)
    ttk.Label(left_panel, text="Run Info").grid(
        row=info_row + 1, column=0, columnspan=2, sticky="w", padx=4, pady=(4, 2)
    )
    info_box = tk.Text(left_panel, height=10, width=32)
    info_box.insert("1.0", "Status: Ready\nSelected tiles: 0\n")
    info_box.configure(state="disabled")
    info_box.grid(row=info_row + 2, column=0, columnspan=2, sticky="nsew", padx=4, pady=4)
    left_panel.rowconfigure(info_row + 2, weight=1)

    ttk.Label(middle_panel, text="Loaded Map").grid(row=0, column=0, sticky="w")
    map_canvas = tk.Canvas(middle_panel, background="#1f2933", height=500)
    map_canvas.grid(row=1, column=0, sticky="nsew", pady=6)
    placeholder_id = map_canvas.create_text(
        10,
        10,
        anchor="nw",
        fill="#cbd5e1",
        text="Select a year to load the map.",
    )
    map_image_id: Optional[int] = None
    map_photo: Optional[ImageTk.PhotoImage] = None
    map_path: Optional[Path] = None
    map_meta: Dict[str, float] = {}
    year_files: Dict[str, Path] = {}
    middle_panel.rowconfigure(1, weight=1)
    middle_panel.columnconfigure(0, weight=1)

    ttk.Label(right_panel, text="Selected Tiles (Temporal)").grid(row=0, column=0, sticky="w")
    tiles_canvas = tk.Canvas(right_panel, highlightthickness=0)
    tiles_scroll = ttk.Scrollbar(right_panel, orient="vertical", command=tiles_canvas.yview)
    tiles_frame = ttk.Frame(tiles_canvas)
    tiles_canvas.configure(yscrollcommand=tiles_scroll.set)
    tiles_canvas.grid(row=1, column=0, sticky="nsew", pady=6)
    tiles_scroll.grid(row=1, column=1, sticky="ns", pady=6)
    right_panel.rowconfigure(1, weight=1)
    right_panel.columnconfigure(0, weight=1)

    tiles_window_id = tiles_canvas.create_window((0, 0), window=tiles_frame, anchor="nw")
    tiles_images: List[ImageTk.PhotoImage] = []

    result: Dict[str, object] = {}
    cancelled = {"value": False}

    def _parse_years_from_imgs(img_dir: Path) -> List[Tuple[str, Path]]:
        if not img_dir.exists():
            return []
        exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
        pairs: List[Tuple[str, Path]] = []
        for path in sorted(img_dir.iterdir()):
            if path.suffix.lower() not in exts:
                continue
            match = re.search(r"(19\d{2}|20\d{2})", path.stem)
            if match:
                pairs.append((match.group(1), path))
        return pairs

    def _refresh_years() -> None:
        nonlocal year_files
        data_dir = values["data_dir"].get()
        img_dir = Path(data_dir) / "imgs"
        year_pairs = _parse_years_from_imgs(img_dir)
        year_files = {year: path for year, path in year_pairs}
        years = sorted({year for year, _ in year_pairs})
        year_combo["values"] = years
        year_var.set("")
        year_combo.set("")

    def _render_map(target: Path) -> None:
        nonlocal map_image_id, map_photo, map_path, map_meta
        map_path = target
        image = Image.open(target)
        down_w = max(1, image.width // 4)
        down_h = max(1, image.height // 4)
        if down_w < image.width or down_h < image.height:
            image = image.resize((down_w, down_h), Image.BILINEAR)
        orig_w, orig_h = image.size
        canvas_width = map_canvas.winfo_width() or 800
        canvas_height = map_canvas.winfo_height() or 500
        image.thumbnail((canvas_width - 20, canvas_height - 20))
        render_w, render_h = image.size
        x0 = (canvas_width - render_w) / 2
        y0 = (canvas_height - render_h) / 2
        map_meta = {
            "orig_w": float(orig_w),
            "orig_h": float(orig_h),
            "render_w": float(render_w),
            "render_h": float(render_h),
            "x0": float(x0),
            "y0": float(y0),
        }
        map_photo = ImageTk.PhotoImage(image)
        map_canvas.delete("grid")
        map_canvas.delete("selection")
        if map_image_id is not None:
            map_canvas.delete(map_image_id)
        map_canvas.delete(placeholder_id)
        map_image_id = map_canvas.create_image(
            x0, y0, image=map_photo, anchor="nw"
        )
        _draw_grid()

    def _draw_grid() -> None:
        if not map_meta:
            return
        try:
            tile_size = int(values["tile_size"].get())
        except (ValueError, TypeError):
            return
        if tile_size <= 0:
            return
        map_canvas.delete("grid")
        scale_x = map_meta["render_w"] / map_meta["orig_w"]
        scale_y = map_meta["render_h"] / map_meta["orig_h"]
        x0 = map_meta["x0"]
        y0 = map_meta["y0"]
        render_w = map_meta["render_w"]
        render_h = map_meta["render_h"]
        step_x = tile_size * scale_x
        step_y = tile_size * scale_y
        if step_x <= 1 or step_y <= 1:
            return
        x = x0
        while x <= x0 + render_w:
            map_canvas.create_line(x, y0, x, y0 + render_h, fill="#ef4444", tags="grid")
            x += step_x
        y = y0
        while y <= y0 + render_h:
            map_canvas.create_line(x0, y, x0 + render_w, y, fill="#ef4444", tags="grid")
            y += step_y

    def _load_map_for_year(year: str) -> None:
        nonlocal map_path
        target = year_files.get(year)
        if target is None:
            return
        _render_map(target)

    def on_run() -> None:
        for key, _ in fields:
            result[key] = values[key].get()
        result["selected_year"] = year_var.get()
        root.destroy()

    def on_cancel() -> None:
        cancelled["value"] = True
        root.destroy()

    buttons = ttk.Frame(left_panel)
    buttons.grid(row=info_row + 3, column=0, columnspan=2, sticky="e", pady=(12, 0))
    ttk.Button(buttons, text="Run", command=on_run).grid(row=0, column=0, padx=4)
    ttk.Button(buttons, text="Cancel", command=on_cancel).grid(row=0, column=1, padx=4)

    def _on_year_selected(event: object) -> None:
        year = year_var.get()
        if year:
            _load_map_for_year(year)

    def _on_data_dir_change(*_args: object) -> None:
        _refresh_years()

    def _on_canvas_resize(event: object) -> None:
        if map_path is None:
            return
        _render_map(map_path)

    def _on_tile_size_change(*_args: object) -> None:
        if map_path is None:
            return
        _render_map(map_path)

    def _on_canvas_click(event: object) -> None:
        if not map_meta:
            return
        try:
            tile_size = int(values["tile_size"].get())
        except (ValueError, TypeError):
            return
        if tile_size <= 0:
            return
        x0 = map_meta["x0"]
        y0 = map_meta["y0"]
        render_w = map_meta["render_w"]
        render_h = map_meta["render_h"]
        if not (x0 <= event.x <= x0 + render_w and y0 <= event.y <= y0 + render_h):
            return

        scale_x = map_meta["render_w"] / map_meta["orig_w"]
        scale_y = map_meta["render_h"] / map_meta["orig_h"]
        img_x = (event.x - x0) / scale_x
        img_y = (event.y - y0) / scale_y
        col = int(img_x // tile_size)
        row = int(img_y // tile_size)

        tile_x0 = x0 + col * tile_size * scale_x
        tile_y0 = y0 + row * tile_size * scale_y
        tile_x1 = tile_x0 + tile_size * scale_x
        tile_y1 = tile_y0 + tile_size * scale_y
        map_canvas.delete("selection")
        map_canvas.create_rectangle(
            tile_x0,
            tile_y0,
            tile_x1,
            tile_y1,
            fill="#fcd34d",
            outline="#f59e0b",
            stipple="gray25",
            tags="selection",
        )

        _update_tiles_panel(row, col)

    def _get_tile_bbox(row: int, col: int, tile_size: int, width: int, height: int) -> Tuple[int, int, int, int]:
        left = col * tile_size
        upper = row * tile_size
        right = min(left + tile_size, width)
        lower = min(upper + tile_size, height)
        return left, upper, right, lower

    def _open_label_window(year: str, image: Image.Image) -> None:
        window = tk.Toplevel(root)
        window.title(f"Label Tile {year}")
        window.geometry("800x700")

        header = ttk.Label(
            window,
            text="Left click to add vertices. Double click to close polygon. Clear removes all.",
        )
        header.pack(anchor="w", padx=12, pady=8)

        canvas = tk.Canvas(window, background="#0f172a")
        canvas.pack(fill="both", expand=True, padx=12, pady=8)

        img = image.copy()
        img.thumbnail((760, 520))
        photo = ImageTk.PhotoImage(img)
        canvas.image = photo
        canvas.create_image(10, 10, image=photo, anchor="nw")

        points: List[Tuple[int, int]] = []
        polygons: List[List[Tuple[int, int]]] = []
        current_line_ids: List[int] = []

        def _draw_point(x: int, y: int) -> None:
            canvas.create_oval(x - 3, y - 3, x + 3, y + 3, fill="#f59e0b", outline="")

        def _add_vertex(event: object) -> None:
            x, y = int(event.x), int(event.y)
            points.append((x, y))
            _draw_point(x, y)
            if len(points) > 1:
                x0, y0 = points[-2]
                line_id = canvas.create_line(x0, y0, x, y, fill="#f59e0b", width=2)
                current_line_ids.append(line_id)

        def _close_polygon(event: object) -> None:
            if len(points) < 3:
                return
            x0, y0 = points[-1]
            x1, y1 = points[0]
            canvas.create_line(x0, y0, x1, y1, fill="#f59e0b", width=2)
            polygons.append(list(points))
            points.clear()
            current_line_ids.clear()

        def _clear() -> None:
            canvas.delete("all")
            canvas.create_image(10, 10, image=photo, anchor="nw")
            points.clear()
            polygons.clear()
            current_line_ids.clear()

        button_row = ttk.Frame(window)
        button_row.pack(fill="x", padx=12, pady=(0, 12))
        ttk.Button(button_row, text="Clear", command=_clear).pack(side="left")
        ttk.Button(button_row, text="Done", command=window.destroy).pack(side="right")

        canvas.bind("<Button-1>", _add_vertex)
        canvas.bind("<Double-Button-1>", _close_polygon)

    def _update_tiles_panel(row: int, col: int) -> None:
        nonlocal tiles_images
        for child in tiles_frame.winfo_children():
            child.destroy()
        tiles_images = []

        data_dir = values["data_dir"].get()
        if not data_dir:
            return
        img_dir = Path(data_dir) / "imgs"
        if not img_dir.exists():
            return

        try:
            tile_size = int(values["tile_size"].get())
        except (ValueError, TypeError):
            return
        if tile_size <= 0:
            return

        for idx, year in enumerate(sorted(year_files.keys())):
            path = year_files[year]
            try:
                image = Image.open(path)
            except Exception:
                continue
            bbox = _get_tile_bbox(row, col, tile_size, image.width, image.height)
            tile = image.crop(bbox)
            tile.thumbnail((220, 220))
            photo = ImageTk.PhotoImage(tile)
            tiles_images.append(photo)

            tile_frame = ttk.Frame(tiles_frame)
            tile_frame.grid(row=idx, column=0, sticky="ew", pady=4)
            label = ttk.Label(tile_frame, text=year, width=8)
            label.grid(row=0, column=0, sticky="w", padx=4)
            btn = ttk.Button(
                tile_frame,
                image=photo,
                command=lambda y=year, img=tile.copy(): _open_label_window(y, img),
            )
            btn.grid(row=0, column=1, sticky="w", padx=4)

        tiles_frame.update_idletasks()
        tiles_canvas.configure(scrollregion=tiles_canvas.bbox("all"))

    def _on_tiles_canvas_resize(event: object) -> None:
        tiles_canvas.itemconfigure(tiles_window_id, width=tiles_canvas.winfo_width())

    year_combo.bind("<<ComboboxSelected>>", _on_year_selected)
    values["data_dir"].trace_add("write", _on_data_dir_change)
    values["tile_size"].trace_add("write", _on_tile_size_change)
    map_canvas.bind("<Configure>", _on_canvas_resize)
    map_canvas.bind("<Button-1>", _on_canvas_click)
    tiles_canvas.bind("<Configure>", _on_tiles_canvas_resize)
    _refresh_years()

    root.mainloop()

    if cancelled["value"]:
        return None

    return result
