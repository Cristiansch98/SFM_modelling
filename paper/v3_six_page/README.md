# V3-derived six-page research paper

Deliverable: emv_yielding_v3_6pages.pdf. Editable source: emv_yielding_v3_6pages.tex.

Exactly six A4 pages including four vector figures, two tables, and references, in IEEE conference format with standard 10-point body text.

This edition develops the v3 argument using the existing unified six-page synthesis and its implementation-aware corrections. It strengthens the research question, the connection between shared lane resistance and staged identification, the derivation of characteristic clearance, and the distinction between empirical fit and internal simulation tests. Original manuscripts are preserved.

Reported results and graphics are inherited from the project; no new calibration or experimental validation was performed. The discussion explicitly scopes the escape approximation, selected width-analysis conditions, EV evaluation using new simulation seeds, and frame-based overlap diagnostic. References are inherited from v3; a targeted literature check covered emergency interaction and cooperative pre-clearing work, not a complete bibliography audit. Relevant publisher record: https://www.sciencedirect.com/science/article/pii/S0191261520304033

Source materials:
- ../itec_emv_sfm_fullpaper_v3.tex
- ../revised/itec_emv_sfm_fullpaper_v3.tex
- ../unified_conference/itec_emv_sfm_unified_6pages.tex and figs/
- ../code_based_six_page/README.md

Rebuild from this directory:

```sh
latexmk -pdf -interaction=nonstopmode -halt-on-error emv_yielding_v3_6pages.tex
```

Verification: successful LaTeX build; six-page count confirmed with PyMuPDF; page montage and final-page layout visually inspected; no unresolved references. TeX reports a minor 1.12-point vertical overfull box; no visible clipping was observed.
