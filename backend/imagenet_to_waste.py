"""ImageNet class index -> waste category mapping.

Curated subset of ImageNet1K covering items that plausibly appear as household
waste. Indices not present here contribute to the `unknown` bucket.

Каждый раз когда видим неправильное определение в проде — добавляем правильный
imagenet idx -> waste категорию. Изначально карта намеренно консервативная,
лучше unknown чем неправильный класс.

Limitations:
- ImageNet has very weak coverage of empty soda/beer cans → metal sparse here.
- Crumpled / dirty waste rarely matches ImageNet samples.
- Bottles are ambiguous between glass/plastic; defaults follow modern usage.
"""

IMAGENET_TO_WASTE: dict[int, str] = {
    # ── Glass ────────────────────────────────────────────────
    440: "glass",      # beer_bottle
    441: "glass",      # beer_glass
    907: "glass",      # wine_bottle

    # ── Plastic ──────────────────────────────────────────────
    463: "plastic",    # bucket, pail (modern usage usually plastic)
    720: "plastic",    # pill_bottle
    728: "plastic",    # plastic_bag
    737: "plastic",    # pop_bottle, soda_bottle (PET)
    798: "plastic",    # shower_curtain (vinyl)
    898: "plastic",    # water_bottle (PET)
    899: "plastic",    # water_jug (modern usage)

    # ── Paper ────────────────────────────────────────────────
    549: "paper",      # envelope
    921: "paper",      # book_jacket, dust_cover
    922: "paper",      # menu
    999: "paper",      # toilet_tissue, toilet_paper, bathroom_tissue

    # ── Metal ────────────────────────────────────────────────
    # ImageNet does not have a "soda can" / "aluminium can" class.
    # Add candidate indices after empirical testing on real photos.
    # Possible weak proxies (uncomment cautiously):
    # 868: "metal",    # tray (often metal but ambiguous)
}
