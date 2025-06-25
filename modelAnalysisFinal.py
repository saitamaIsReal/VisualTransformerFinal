#!/usr/bin/env python3
import sys
import os
import torch
import numpy as np
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image, ImageDraw
import requests
from io import BytesIO
from typing import Union
from shapiq.approximator.regression import RegressionFSII
from shapiq.interaction_values import InteractionValues
from IPython.display import display
from shapiq.plot.upset import upset_plot
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from PIL import ImageFont
import seaborn as sns

# === Logging Setup ===
import time
# === Logging Setup ===
os.makedirs("output", exist_ok=True)
timestamp = time.strftime("%Y%m%d_%H%M%S")
log_path = f"output/log_run_{timestamp}.txt"

# Öffne Logdatei im "line-buffered"-Modus
log_file = open(log_path, "w", encoding="utf-8", buffering=1)
sys.stdout = log_file
sys.stderr = log_file
print(f"[INFO] Logging gestartet um {timestamp}")


# Ordnerstruktur anlegen
output_dirs = [
    "output/heatmaps",
    "output/histograms",
    "output/histograms_shapleyValue",
    "output/upsets_2er",
    "output/upsets_1er2er",
    "output/top_interactions",
]

for d in output_dirs:
    os.makedirs(d, exist_ok=True)

image_paths = [
    "images/cat1.jpg",
    "images/cat2.jpg",
    "images/cat3.jpg",
    "images/cat4.jpg",
    "images/lucky.jpeg",
    "images/dog2.jpg",
    "images/dog3.jpg",
    "images/dog4.jpg",

]

# Model selection
models = [
    "google/vit-base-patch32-384",
    "facebook/deit-tiny-patch16-224",
    "akahana/vit-base-cats-vs-dogs",
]

device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Running on device:", device)
if device.type == "cuda":
    print(" CUDA device name:", torch.cuda.get_device_name())


# Helper Functions
def create_heatmap_overlay(image: Image.Image, shapley_values: np.ndarray, n_patches_per_row: int, cell: int) -> Image.Image:
    """
    Erstellt eine Heatmap, die positive Werte in Rot und negative Werte in Blau zeigt.
    """
    min_val, max_val = shapley_values.min(), shapley_values.max()
    norm_vals = shapley_values / max(abs(min_val), abs(max_val) + 1e-8)  # Normalisierung auf [-1, 1]

    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    for idx, val in enumerate(norm_vals):
        r, c = divmod(idx, n_patches_per_row)
        x1, y1 = c * cell, r * cell
        x2, y2 = x1 + cell, y1 + cell

        if val > 0:
            color = (int(val * 255), 0, 0, 120)  # Rot für positive Werte
        elif val < 0:
            color = (0, 0, int(-val * 255), 120)  # Blau für negative Werte
        else:
            color = (0, 0, 0, 0)  # Keine Farbe für 0
        draw.rectangle([x1, y1, x2, y2], fill=color)

    return Image.alpha_composite(image.convert("RGBA"), overlay)

def combine_side_by_side(images):
    total_width = sum(img.width for img in images)
    max_height = max(img.height for img in images)
    combined = Image.new("RGBA", (total_width, max_height))
    x_offset = 0
    for img in images:
        combined.paste(img, (x_offset, 0))
        x_offset += img.width
    return combined

def mask_image_grid(
    img: Image.Image,
    coalition: Union[np.ndarray, list],
    image_size: int,
    n_patches_per_row: int,
    cell: int
) -> Image.Image:
    """
    Graying out patches where coalition[i] is False (0).
    """
    arr  = np.array(img.resize((image_size, image_size))).copy()
    coal = np.asarray(coalition, dtype=bool)
    for i, keep in enumerate(coal):
        if not keep:
            r, c = divmod(i, n_patches_per_row)
            y1, y2 = r*cell, (r+1)*cell
            x1, x2 = c*cell, (c+1)*cell
            arr[y1:y2, x1:x2] = 128
    return Image.fromarray(arr)

def draw_grid(
    img: Image.Image,
    n_patches_per_row: int,
    cell: int
):
    """
    Draw numeric patch-grid for visualization.
    """
    draw = ImageDraw.Draw(img)
    for r in range(n_patches_per_row):
        for c in range(n_patches_per_row):
            x1, y1 = c*cell, r*cell
            x2, y2 = x1+cell, y1+cell
            draw.rectangle([x1,y1,x2,y2], outline="gray", width=1)
            idx = r*n_patches_per_row + c
            try:
                font = ImageFont.truetype("arial.ttf", size=7)
            except:
                font = ImageFont.load_default()
            draw.text((x1+2, y1+2), str(idx), fill="gray", font=font)

    display(img)


def build_upset_plot(item_list, name_suffix):
    if not item_list:
        print(f"[WARN] Keine Einträge für {name_suffix}")
        return

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
    fig.suptitle(f"{model_name} – {name_suffix}", fontsize=12, weight='bold')
    fig.subplots_adjust(bottom=0.25)
    
    canvas = FigureCanvas(fig)
    canvas.draw()
    w, h = canvas.get_width_height()
    img = Image.frombuffer("RGBA", (w, h), canvas.buffer_rgba(), "raw", "RGBA", 0, 1)
    plt.close(fig)

    # Speichern
    filename = f"output/upsets_2er/{img_idx}_{model_name.replace('/', '_')}_{name_suffix}.png"
    img.save(filename)
    print(f"[SAVED] {filename}")
    return img


def get_top3_patch_ids(fsii_path, kind):
    result = InteractionValues.load(fsii_path)
    second_order = result.get_n_order(order=2)
    top_items = sorted(second_order.dict_values.items(), key=lambda x: abs(x[1]), reverse=True)[:10]
    if kind == "pos":
        top_items = [item for item in top_items if item[1] > 0][:3]
    elif kind == "neg":
        top_items = [item for item in top_items if item[1] < 0][:3]
    else:
        top_items = top_items[:3]
    return [i for pair, _ in top_items for i in pair]

def overlay_top3_combined(patch_sets, colors, image, n_patches_per_row, cell):
    img_overlay = image.copy().convert("RGBA")
    draw = ImageDraw.Draw(img_overlay, "RGBA")

    # Zuerst das Grid
    draw_grid(img_overlay, n_patches_per_row, cell)

    # Dann die Kästchen mit leichtem Inset und Transparenz
    for idx, (color, patches) in enumerate(zip(colors, patch_sets)):
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
    """
    Given array (n_coalitions, n_patches), return logits for `predicted_class`.
    """
    out = []
    for coalition in coalitions:
        masked = mask_image_grid(image, coalition, image_size, n_patches_per_row, cell)
        batch  = processor(images=masked, return_tensors="pt").to(device)
        with torch.no_grad():
            logit = model(**batch).logits[0, predicted_class].item()
        out.append(logit)
    return np.array(out)

#=======MAIN========

heatmaps_all = []
histograms_all = []
upset2_all = []
upset12_all = []
top_patches_all = []
upset_plots_2er_only = []
upset_plots_combined = []
histograms = []
shapley_histograms_all = []

# Farbcodes für Modelle (eine Farbe pro Modell)
model_colors = [(255, 0, 0), (0, 180, 0), (0, 128, 255)]
top_interaction_patches = []  
top_interaction_patches_all = []

for img_idx, image_path in enumerate(image_paths):
    heatmaps_per_image = []
    histograms_per_image = []
    upsets2_per_image = []
    upsets12_per_image = []
    top_patches_per_image = []
    shapley_histograms_per_image = []

    print(f"\n===== Bild {img_idx+1}/{len(image_paths)}: {image_path} =====")
    

    for model_name in models:
        print(f"\n--- Evaluating {model_name} ---")

    # 1) load processor & model
        processor = AutoImageProcessor.from_pretrained(model_name)
        model     = AutoModelForImageClassification.from_pretrained(model_name).to(device).eval()

    # 2) extract patch & image config
        patch_size        = model.config.patch_size
        image_size        = model.config.image_size
        n_patches_per_row = image_size // patch_size
        n_patches         = n_patches_per_row**2
        cell = patch_size

        image = Image.open(image_path).convert("RGB").resize((image_size, image_size))

    # 3) load & show grid
        draw_grid(image.copy(), n_patches_per_row, cell)

    # 4) determine target class
        inputs = processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            logits = model(**inputs).logits
        predicted_class = int(logits.argmax(-1))
        print("Predicted class:", model.config.id2label[predicted_class])
        print("predicted logits:", logits[0,predicted_class])

    # 5) quick-check full vs. empty
        full  = value_function(
            np.array([np.ones(n_patches, bool)]),
            processor, model, device, image,
            predicted_class,
            image_size, n_patches_per_row, cell
        )[0]
        empty = value_function(
            np.array([np.zeros(n_patches, bool)]),
            processor, model, device, image,
            predicted_class,
            image_size, n_patches_per_row, cell
        )[0]
        print(f" Logit full:  {full:.2f}")
        print(f" Logit empty: {empty:.2f}")


    # 6) FSII approx
        approximator = RegressionFSII(
            n=n_patches,
            max_order=2,
            pairing_trick=False,
            random_state=42,
        )
        result = approximator.approximate(
            budget = 64000,
            game=lambda c: value_function(
                c,
                processor, model, device, image,
                predicted_class,
                image_size, n_patches_per_row, cell
            )
        )

        # === SAVE FSII-RESULTS ===
        save_dir = "output/results"
        os.makedirs(save_dir, exist_ok=True)
        result_path = os.path.join(save_dir, f"{img_idx}_{model_name.replace('/', '_')}.fsii")
        result.save(path=result_path)

        # === Heatmap (nur 1st-Order Shapley Values) ===
        first_order = result.get_n_order(order=1)
        shapley_values = np.array([first_order[(i,)] for i in range(n_patches)])
        heatmap_img = create_heatmap_overlay(image, shapley_values, n_patches_per_row, cell)
        # Grid über Heatmap legen
        heatmap_with_grid = heatmap_img.copy()
        draw_grid(heatmap_with_grid, n_patches_per_row, cell)


        
        # === UpSet Plot (2nd-Order Interactions) ===
        second_order = result.get_n_order(order=2)

        # Top 10 nach Betrag
        top10_items = sorted(second_order.dict_values.items(), key=lambda x: abs(x[1]), reverse=True)[:10]

        top10_pos = [item for item in top10_items if item[1] > 0]
        top10_neg = [item for item in top10_items if item[1] < 0]

        img_mixed = build_upset_plot(top10_items, "mixed")


        # Top 3 Interaktionen extrahieren
        top3_pairs = top10_items[:3]
        patch_ids = [i for pair, _ in top3_pairs for i in pair]
        top_interaction_patches.append(patch_ids)

        # Logging der Top10
        top10_pairs = [item[0] for item in top10_items]
        top10_values = [item[1] for item in top10_items]
        for (i, j), val in zip(top10_pairs, top10_values):
            print(f"yes shapiq: Interaction ({i}, {j}): {val:.4f}")

        #========= UpSet Plot: 1er + 2er kombiniert ===========

        # 1) Alle Werte holen
        all_items = result.dict_values.items()

        # 2) Nur (i,) und (i,j) behalten
        filtered_items = [(k, v) for k, v in all_items if len(k) in [1, 2]]

        # 3) Top 10 nach Betrag
        top10_items_combined = sorted(filtered_items, key=lambda x: abs(x[1]), reverse=True)[:10]

        # 4) Feature-IDs extrahieren
        used_ids_combined = sorted(set(i for key, _ in top10_items_combined for i in key))
        id2new_combined = {real: idx for idx, real in enumerate(used_ids_combined)}

        # 5) Remapping anwenden
        remapped_keys_combined = [
            tuple(sorted(id2new_combined[i] for i in key))
            for key, _ in top10_items_combined
        ]
        top10_values_combined = [v for _, v in top10_items_combined]

        # 6) Neues InteractionValues-Objekt bauen
        subset_combined = InteractionValues(
            values=np.array(top10_values_combined),
            index="sparse",
            n_players=len(used_ids_combined),
            min_order=1,
            max_order=2,
            interaction_lookup={k: i for i, k in enumerate(remapped_keys_combined)},
            baseline_value=0.0,
        )

        # 8) Plot erzeugen
        fig_combined = upset_plot(subset_combined, show=False, feature_names=[str(fid) for fid in used_ids_combined])
        fig_combined.suptitle(f"{model_name} (1er + 2er)", fontsize=12, weight='bold')
        fig_combined.subplots_adjust(bottom=0.25)

        # 9) In PIL umwandeln
        canvas_combined = FigureCanvas(fig_combined)
        canvas_combined.draw()
        width_c, height_c = canvas_combined.get_width_height()
        img_combined = Image.frombuffer("RGBA", (width_c, height_c), canvas_combined.buffer_rgba(), "raw", "RGBA", 0, 1)
        upset_plots_combined.append(img_combined)
        plt.close(fig_combined)


        # === VISUALISIERUNG DER TOP 3 INTERACTIONS (GEMISCHT, POSITIV, NEGATIV) ===
        # Farbcodes
        color_mixed = (255, 0, 0)     # Rot
        color_pos   = (0, 200, 0)     # Grün
        color_neg   = (0, 100, 255)   # Blau

        # Patch IDs extrahieren
        top3_mixed = top10_items[:3]
        top3_pos   = top10_pos[:3]
        top3_neg   = top10_neg[:3]

        def format_top3(pairs):
            return "; ".join(f"({i},{j})" for (i, j) in pairs)

        print(f"🔴 mixed – {model_name}: {format_top3(top3_mixed)}")
        print(f"🟢 pos   – {model_name}: {format_top3(top3_pos)}")
        print(f"🔵 neg   – {model_name}: {format_top3(top3_neg)}")

        patches_mixed = [i for pair, _ in top3_mixed for i in pair]
        patches_pos   = [i for pair, _ in top3_pos for i in pair]
        patches_neg   = [i for pair, _ in top3_neg for i in pair]

        #========= HISTOGRAMM (1er + 2er) ==========
        # Alle Shapley- und Interaktionswerte bis Order 2
        all_items = result.dict_values.items()
        values_1_2 = [abs(v) for k, v in all_items if len(k) in [1, 2]]

        # Histogramm mit seaborn
        fig_hist = plt.figure(figsize=(6, 4))
        sns.histplot(values_1_2, bins=30, kde=False, color='royalblue')
        plt.title(f"Histogramm – Shapley + Interaktionen\n({model_name})", fontsize=10)
        plt.xlabel("Wert (absolut)")
        plt.ylabel("Anzahl")
        plt.tight_layout()

        # Speichern oder als PIL
        from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
        canvas_hist = FigureCanvas(fig_hist)
        canvas_hist.draw()
        w, h = canvas_hist.get_width_height()
        hist_img = Image.frombuffer("RGBA", (w, h), canvas_hist.buffer_rgba(), "raw", "RGBA", 0, 1)
        plt.close(fig_hist)

        # Zur Liste hinzufügen
        heatmaps_per_image.append(heatmap_with_grid)
        heatmap_with_grid.save(f"output/heatmaps/{img_idx}_{model_name.replace('/', '_')}.png")
        histograms_per_image.append(hist_img)
        hist_img.save(f"output/histograms/{img_idx}_{model_name.replace('/', '_')}.png")
        upsets2_per_image.append(img_mixed)
        img_mixed.save(f"output/upsets_2er/{img_idx}_{model_name.replace('/', '_')}_mixed.png")
        upsets12_per_image.append(img_combined)
        img_combined.save(f"output/upsets_1er2er/{img_idx}_{model_name.replace('/', '_')}.png")
        top_patches_per_image.append(patch_ids)

        heatmaps_all.append(heatmaps_per_image)
        histograms_all.append(histograms_per_image)
        upset2_all.append(upsets2_per_image)
        upset12_all.append(upsets12_per_image)
        top_patches_all.append(top_patches_per_image)
        shapley_histograms_all.append(shapley_histograms_per_image)


        #=========HEAT PRINT==========
        # 🔁 Standardgröße definieren (z. B. 384x384, weil ViT damit arbeitet)
        standard_size = (384, 384)
        # 📐 Alle Heatmaps auf gleiche Größe bringen
        resized_heatmaps = [img.resize(standard_size) for img in heatmaps_per_image]
        # 🔗 Jetzt sauber nebeneinander kombinieren
        final_img = combine_side_by_side(resized_heatmaps)
        final_img.save("vergleich_heatmaps.png")
        final_img.show()

        #=========UPSET PLOT (2er only) BILDER KOMBINIEREN==========
        upset_standard_size = img_mixed.size
        resized_2er = [img.resize(upset_standard_size) for img in upset_plots_2er_only if img is not None]
        if resized_2er:
            combined_2er = combine_side_by_side(resized_2er)
            combined_2er.save("vergleich_upsetplots_2er.png")
            combined_2er.show()
        else:
            print("[SKIPPED] Kein kombinierter 2er-Vergleichsplot (global), da keine Upset-Plots vorhanden.")

        #=========UPSET PLOT (1er + 2er) BILDER KOMBINIEREN==========
        # ========== Kombinierter UpSet-Plot (1er + 2er) pro Bild ==========
        if upsets12_per_image:
            resized_combined = [img.resize(upset_standard_size) for img in upsets12_per_image]
            combined_combined = combine_side_by_side(resized_combined)
            combined_combined.save(f"output/upsets_1er2er/{img_idx}_vergleich_upsetplots_1er_2er.png")
            print(f"[SAVED] output/upsets_1er2er/{img_idx}_vergleich_upsetplots_1er_2er.png")


        #=========HISTOGRAMM-BILDER KOMBINIEREN==========

        # Einheitliche Größe (z. B. wie erstes Histogramm)
        histogram_standard_size = hist_img.size
        resized_histograms = [img.resize(histogram_standard_size) for img in histograms_all[0]]
        combined_histogram_img = combine_side_by_side(resized_histograms)
        combined_histogram_img.save("vergleich_histogramme.png")
        combined_histogram_img.show()

        # ========= HISTOGRAMM NUR FÜR SHAPLEY VALUES (ORDER 1) =========
        fig_shap = plt.figure(figsize=(6, 4))
        sns.histplot(np.abs(shapley_values), bins=30, kde=False, color='darkorange')
        plt.title(f"Histogramm – Shapley-Werte\n({model_name})", fontsize=10)
        plt.xlabel("Wert (absolut)")
        plt.ylabel("Anzahl")
        plt.tight_layout()

        canvas_shap = FigureCanvas(fig_shap)
        canvas_shap.draw()
        w, h = canvas_shap.get_width_height()
        hist_img_shap = Image.frombuffer("RGBA", (w, h), canvas_shap.buffer_rgba(), "raw", "RGBA", 0, 1)
        hist_img_shap.save(f"output/histograms_shapleyValue/{img_idx}_{model_name.replace('/', '_')}_shapley_only.png")
        print(f"[SAVED] output/histograms_shapleyValue/{img_idx}_{model_name.replace('/', '_')}_shapley_only.png")
        shapley_histograms_per_image.append(hist_img_shap)
        plt.close(fig_shap)

        # ======= Gemeinsame Top-3-Visualisierung (mixed/pos/neg) für alle 3 Modelle =======

    if len(models) == 3:
        image_orig = Image.open(image_paths[img_idx]).convert("RGB").resize((image_size, image_size))
        patchgroups_mixed = []
        patchgroups_pos = []
        patchgroups_neg = []

        for model_name in models:
            fsii_path = f"output/results/{img_idx}_{model_name.replace('/', '_')}.fsii"
            patchgroups_mixed.append(get_top3_patch_ids(fsii_path, "mixed"))
            patchgroups_pos.append(get_top3_patch_ids(fsii_path, "pos"))
            patchgroups_neg.append(get_top3_patch_ids(fsii_path, "neg"))

        img_comb_mixed = overlay_top3_combined(patchgroups_mixed, model_colors, image_orig, n_patches_per_row, cell)
        img_comb_pos   = overlay_top3_combined(patchgroups_pos,   model_colors, image_orig, n_patches_per_row, cell)
        img_comb_neg   = overlay_top3_combined(patchgroups_neg,   model_colors, image_orig, n_patches_per_row, cell)

        img_comb_mixed.save(f"output/top_interactions/{img_idx}_top3_combined_mixed.png")
        img_comb_pos.save(f"output/top_interactions/{img_idx}_top3_combined_pos.png")
        img_comb_neg.save(f"output/top_interactions/{img_idx}_top3_combined_neg.png")
        print("[SAVED] 3× kombinierte Top-3 Interaktionen pro Bild gespeichert")







