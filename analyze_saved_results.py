#!/usr/bin/env python3
import os
from shapiq.interaction_values import InteractionValues

FSII_DIR = "output/results"

def test_fsii(path):
    iv = InteractionValues.load(path)
    # interaction_lookup: dict mapping tuple of patches -> index in iv.values
    lookup = iv.interaction_lookup

    # 1st-Order (= Shapley) und 2nd-Order
    fo = {k: iv.values[idx] for k, idx in lookup.items() if len(k) == 1}
    so = {k: iv.values[idx] for k, idx in lookup.items() if len(k) == 2}

    # Ausgabe
    print(f"\n--- {os.path.basename(path)} ---")
    print(f"Shapley-Einträge:    {len(fo)}  (expected = #patches)")
    print(f"2nd-Order-Paare:     {len(so)}  (expected = n_patches*(n_patches-1)/2)\n")

    # erste 10 Shapley-Werte
    print("Erste 10 Shapley-Werte:")
    for (i,), val in list(fo.items())[:10]:
        print(f"  Patch {i:3d}: {val: .4f}")
    # top 5 Interaktionen
    print("\nTop 5 2nd-Order-Interaktionen:")
    top5 = sorted(so.items(), key=lambda kv: abs(kv[1]), reverse=True)[:5]
    for (i,j), val in top5:
        print(f"  Patches ({i:3d},{j:3d}): {val: .4f}")

if __name__ == "__main__":
    for fn in sorted(os.listdir(FSII_DIR)):
        if not fn.endswith(".fsii"):
            continue
        path = os.path.join(FSII_DIR, fn)
        test_fsii(path)
    print("\n✅ Fertig!")
