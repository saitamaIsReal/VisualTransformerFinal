import os
import numpy as np
from typing import Union
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image, ImageDraw, ImageFont
from shapiq.approximator.regression import RegressionFSII
from shapiq.interaction_values import InteractionValues
from shapiq.plot.upset import upset_plot
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from PIL import Image
from shapiq.interaction_values import InteractionValues

def create_heatmap_overlay(image: Image.Image, shapley_values: np.ndarray, n_patches_per_row: int, cell: int) -> Image.Image:
    # Normalize values to [-1,1]
    min_val, max_val = shapley_values.min(), shapley_values.max()
    norm_vals = shapley_values / (max(abs(min_val), abs(max_val)) + 1e-8)

    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    for idx, val in enumerate(norm_vals):
        r, c = divmod(idx, n_patches_per_row)
        x1, y1 = c * cell, r * cell
        x2, y2 = x1 + cell, y1 + cell
        if val > 0:
            color = (int(val * 255), 0, 0, 120)
        elif val < 0:
            color = (0, 0, int(-val * 255), 120)
        else:
            color = (0, 0, 0, 0)
        draw.rectangle([x1, y1, x2, y2], fill=color)

    return Image.alpha_composite(image.convert("RGBA"), overlay)


def draw_grid(img: Image.Image, n_patches_per_row: int, cell: int) -> Image.Image:
    draw = ImageDraw.Draw(img)
    for r in range(n_patches_per_row):
        for c in range(n_patches_per_row):
            x1, y1 = c * cell, r * cell
            x2, y2 = x1 + cell, y1 + cell
            draw.rectangle([x1, y1, x2, y2], outline="gray", width=1)
            idx = r * n_patches_per_row + c
            try:
                font = ImageFont.truetype("arial.ttf", size=7)
            except:
                font = ImageFont.load_default()
            draw.text((x1 + 2, y1 + 2), str(idx), fill="gray", font=font)
    return img


def mask_image_grid(
    img: Image.Image,
    coalition: np.ndarray,
    image_size: int,
    n_patches_per_row: int,
    cell: int
) -> Image.Image:
    arr = np.array(img.resize((image_size, image_size))).copy()
    coal = np.asarray(coalition, dtype=bool)
    for i, keep in enumerate(coal):
        if not keep:
            r, c = divmod(i, n_patches_per_row)
            y1, y2 = r*cell, (r+1)*cell
            x1, x2 = c*cell, (c+1)*cell
            arr[y1:y2, x1:x2] = 128
    return Image.fromarray(arr)


def value_function(
    coalitions: np.ndarray,
    processor,
    model,
    device,
    image,
    predicted_class,
    image_size,
    n_patches_per_row,
    cell
) -> np.ndarray:
    out = []
    for coalition in coalitions:
        masked = mask_image_grid(image, coalition, image_size, n_patches_per_row, cell)
        batch = processor(images=masked, return_tensors="pt").to(device)
        with torch.no_grad():
            logit = model(**batch).logits[0, predicted_class].item()
        out.append(logit)
    return np.array(out)


def build_upset_plot(item_list, name_suffix):
    pairs = [item[0] for item in item_list]
    values = [item[1] for item in item_list]
    used_ids = sorted(set(i for pair in pairs for i in pair))
    id2new = {real: idx for idx, real in enumerate(used_ids)}
    remapped_pairs = [tuple(sorted((id2new[i], id2new[j]))) for (i, j) in pairs]

    subset = InteractionValues(
        values=np.array(values),
        index="sparse",
        n_players=len(used_ids),
        max_order=2,
        min_order=2,
        interaction_lookup={pair: idx for idx, pair in enumerate(remapped_pairs)},
        baseline_value=0.0
    )

    fig = upset_plot(subset, show=False, feature_names=[str(fid) for fid in used_ids])
    fig.subplots_adjust(bottom=0.25)
    canvas = FigureCanvas(fig)
    canvas.draw()
    w, h = canvas.get_width_height()
    img = Image.frombuffer("RGBA", (w, h), canvas.buffer_rgba(), "raw", "RGBA", 0, 1)
    return img

def get_top3_patch_ids(fsii_path: str, kind: str):
    result = InteractionValues.load(fsii_path)
    second = result.get_n_order(order=2).dict_values.items()
    if kind == "pos":
        items = [x for x in second if x[1] > 0]
    elif kind == "neg":
        items = [x for x in second if x[1] < 0]
    else:
        items = list(second)
    items = sorted(items, key=lambda x: abs(x[1]), reverse=True)[:3]
    return [pid for (pair, _) in items for pid in pair]


def overlay_top3_combined(patch_groups, colors, image, n_patches_per_row, cell):
    img_overlay = image.copy().convert("RGBA")
    draw = ImageDraw.Draw(img_overlay, "RGBA")
    for idx, (patches, color) in enumerate(zip(patch_groups, colors)):
        inset = idx * 2
        rgba = (*color, 180)
        for patch in patches:
            r, c = divmod(patch, n_patches_per_row)
            x1 = c*cell + inset
            y1 = r*cell + inset
            x2 = (c+1)*cell - inset
            y2 = (r+1)*cell - inset
            draw.rectangle([x1, y1, x2, y2], outline=rgba, width=3)
    return img_overlay
