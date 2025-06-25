#!/usr/bin/env python3
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
from shapiq.approximator.regression import RegressionFSII
from shapiq.interaction_values import InteractionValues
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from IPython.display import display
from shapiq.plot.upset import upset_plot
from helper import (
    create_heatmap_overlay,
    draw_grid,
    value_function,
    get_top3_patch_ids,
    overlay_top3_combined
)

# Basis-Ordner und Unterverzeichnisse
base = "sanitycheck"
subdirs = {
    "results":      os.path.join(base, "results"),
    "heatmaps":     os.path.join(base, "heatmaps"),
    "upsets_mixed": os.path.join(base, "top_interactions", "mixed"),
    "upsets_pos":   os.path.join(base, "top_interactions", "pos"),
    "upsets_neg":   os.path.join(base, "top_interactions", "neg"),
    "histograms":   os.path.join(base, "histograms")
}

import time, sys
# === Logging Setup ===
log_dir = os.path.join(base, "logs")
os.makedirs(log_dir, exist_ok=True)
timestamp = time.strftime("%Y%m%d_%H%M%S")
log_path = os.path.join(log_dir, f"log_run_{timestamp}.txt")
log_file = open(log_path, "w", encoding="utf-8", buffering=1)
sys.stdout = log_file
sys.stderr = log_file
print(f"[INFO] Sanity-Check gestartet um {timestamp}")


# === Konfiguration ===
models = [
    "google/vit-base-patch32-384",
    "facebook/deit-tiny-patch16-224",
    "akahana/vit-base-cats-vs-dogs",
]
device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
image_path = "images/airplane.jpg"

for path in subdirs.values():
    os.makedirs(path, exist_ok=True)

# Speicherplatz für Patch-Gruppen aller Modelle
all_mixed = []
all_pos   = []
all_neg   = []

# === Schleife über die Modelle ===
for model_name in models:
    print(f"=== Sanity-Check für {model_name} ===")
    processor = AutoImageProcessor.from_pretrained(model_name)
    model     = AutoModelForImageClassification.from_pretrained(model_name).to(device).eval()

    # Bild laden und auf Modelgröße bringen
    img = Image.open(image_path).convert("RGB")
    patch_size = model.config.patch_size
    image_size = model.config.image_size
    img = img.resize((image_size, image_size))
    n_row = image_size // patch_size
    n_patches = n_row * n_row

    inputs = processor(images=img, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
    predicted_class = int(logits.argmax(-1))
    print("Predicted class:", model.config.id2label[predicted_class])
    print("Predicted logit:", logits[0, predicted_class].item())

    # 1) FSII Approximation
    approximator = RegressionFSII(n=n_patches, max_order=2, pairing_trick=False, random_state=42)
    result = approximator.approximate(
        budget=64000,
        game=lambda c: value_function(
            c, processor, model, device, img,
            int(model(**processor(images=img, return_tensors="pt").to(device))
                .logits.argmax()),
            image_size, n_row, patch_size
        )
    )

    # === FSII-Ergebnis speichern ===
    fsii_path = os.path.join(subdirs["results"], f"{model_name.replace('/', '_')}.fsii")
    result.save(path=fsii_path)

     # → Top-10 2nd-Order Interactions ausgeben
    second_items = result.get_n_order(order=2).dict_values.items()
    top10 = sorted(second_items, key=lambda x: abs(x[1]), reverse=True)[:10]
    print(f"\n--- Top-10 2nd-Order Interactions für {model_name} ---")
    for (i, j), val in top10:
        print(f"({i},{j}): {val:.4f}")


    # 2) Heatmap (1st Order)
    first_order = result.get_n_order(order=1)
    shap_vals   = np.array([first_order[(i,)] for i in range(n_patches)])
    hm = create_heatmap_overlay(img, shap_vals, n_row, patch_size)
    draw_grid(hm, n_row, patch_size)
    hm.save(os.path.join(subdirs["heatmaps"], f"{model_name.replace('/', '_')}_heatmap.png"))

    # → Upset-Plot (2nd-Order) mit Remapping speichern
    # 1) Originale Patch-IDs extrahieren und sortieren
    used_ids = sorted({i for (i,j),_ in top10} | {j for (i,j),_ in top10})
    # 2) Mapping orig → new_index
    id2new   = {orig: new for new, orig in enumerate(used_ids)}
    # 3) Paare remappen
    remapped_pairs = [(id2new[i], id2new[j]) for (i,j),_ in top10]
    # 4) Werte extrahieren
    values = [v for _,v in top10]
    # 5) InteractionValues anlegen
    subset2 = InteractionValues(
        values=np.array(values),
        index="sparse",
        n_players=len(used_ids),
        min_order=2,
        max_order=2,
        interaction_lookup={pair: idx for idx, pair in enumerate(remapped_pairs)},
        baseline_value=0.0
    )
    # 6) Feature-Namen in Original-ID-Reihenfolge
    feature_names = [str(orig) for orig in used_ids]
    # 7) Upset-Plot zeichnen und speichern
    fig2 = upset_plot(subset2, show=False, feature_names=feature_names)
    fig2.subplots_adjust(bottom=0.25)
    canvas2 = FigureCanvas(fig2); canvas2.draw()
    img2 = Image.frombuffer("RGBA", canvas2.get_width_height(),
                            canvas2.buffer_rgba(), "raw", "RGBA", 0, 1)
    img2.save(os.path.join(subdirs["heatmaps"],
                        f"{model_name.replace('/', '_')}_upset2er.png"))
    plt.close(fig2)

    # → Upset-Plot (1er + 2er) speichern
    # 1) Alle Order-1 und Order-2 Paare sammeln und Top-10 nach Betrag auswählen
    all_items = [(k, v) for k, v in result.dict_values.items() if len(k) in (1, 2)]
    top10_all = sorted(all_items, key=lambda x: abs(x[1]), reverse=True)[:10]

    # 2) Originale Feature-IDs extrahieren (flach)
    used_ids_all = sorted({i for pair, _ in top10_all for i in pair})

    # 3) Mapping original → neuer Index
    id2new_all = {orig: new for new, orig in enumerate(used_ids_all)}

    # 4) Remappte Keys erstellen (funktioniert für Order 1 und Order 2)
    remapped_keys_all = [
        tuple(sorted(id2new_all[i] for i in key))
        for key, _ in top10_all
    ]

    # 5) Werte extrahieren
    values_all = [v for _, v in top10_all]

    # 6) InteractionValues-Objekt bauen
    subset_all = InteractionValues(
        values=np.array(values_all),
        index="sparse",
        n_players=len(used_ids_all),
        min_order=1,
        max_order=2,
        interaction_lookup={key: idx for idx, key in enumerate(remapped_keys_all)},
        baseline_value=0.0
    )


    # 7) Feature-Namen in Original-ID-Reihenfolge
    feature_names_all = [str(orig) for orig in used_ids_all]

    # 8) Plot erzeugen und speichern
    fig12 = upset_plot(subset_all, show=False, feature_names=feature_names_all)
    fig12.subplots_adjust(bottom=0.25)
    canvas12 = FigureCanvas(fig12)
    canvas12.draw()
    img12 = Image.frombuffer(
        "RGBA",
        canvas12.get_width_height(),
        canvas12.buffer_rgba(),
        "raw",
        "RGBA",
        0,
        1
    )
    img12.save(os.path.join(
        subdirs["heatmaps"],
        f"{model_name.replace('/', '_')}_upset1er2er.png"
    ))
    plt.close(fig12)


    # 3) Top-3 Patch-IDs extrahieren
    mixed_ids = get_top3_patch_ids(fsii_path, kind="mixed")
    pos_ids   = get_top3_patch_ids(fsii_path, kind="pos")
    neg_ids   = get_top3_patch_ids(fsii_path, kind="neg")
    all_mixed.append(mixed_ids)
    all_pos.append(pos_ids)
    all_neg.append(neg_ids)

    # 4) Histogramm (1. + 2. Order)
    vals = [abs(v) for k, v in result.dict_values.items() if len(k) in (1, 2)]
    plt.figure(figsize=(6, 4))
    sns.histplot(vals, bins=30, stat="count")
    plt.title(f"Histogramm – Shapley+Interaktionen ({model_name})")
    plt.xlabel("Wert (absolut)"); plt.ylabel("Anzahl")
    plt.tight_layout()
    plt.savefig(os.path.join(subdirs["histograms"], f"{model_name.replace('/', '_')}_hist.png"))
    plt.close()

    print(f"→ Outputs für {model_name} in '{base}/' abgelegt")

# 5) Gemeinsame Top-3 Overlays (Mixed / Pos / Neg) über alle Modelle
colors = [(255, 0, 0), (0, 200, 0), (0, 128, 255)]  # google, facebook, akahana

# Verwende für alle Overlays immer dasselbe Originalbild
orig = Image.open(image_path).convert("RGB")
orig = orig.resize((image_size, image_size))

# Mixed
overlay = overlay_top3_combined(all_mixed, colors, orig, n_row, patch_size)
overlay = draw_grid(overlay, n_row, patch_size)
overlay.save(os.path.join(subdirs["upsets_mixed"], "combined_mixed.png"))

# Positiv
overlay = overlay_top3_combined(all_pos, colors, orig, n_row, patch_size)
overlay = draw_grid(overlay, n_row, patch_size)
overlay.save(os.path.join(subdirs["upsets_pos"], "combined_pos.png"))

# Negativ
overlay = overlay_top3_combined(all_neg, colors, orig, n_row, patch_size)
overlay = draw_grid(overlay, n_row, patch_size)
overlay.save(os.path.join(subdirs["upsets_neg"], "combined_neg.png"))

print("Sanity-Check abgeschlossen. Alle Grafiken in 'sanitycheck/' gespeichert.")
