# ✅ Milestone 3 — Update Summary

## 1. Addition of New Visualization Content (Report Pages 7–8)

In Milestone 3, we expanded the report by adding a new section on pages **7 and 8** introducing the custom network-visualization package **`complex_NX`**, which was developed to address scalability and interpretability limitations found in default NetworkX and Plotly graph-drawing tools.

This new section includes:

- a conceptual overview of the package design  
- explanations of the 2D and 3D plotting modes  
- discussion of adaptive transparency, selective edge rendering, and community-based coloring  
- visual examples generated directly using the package  

The goal of this addition is to clearly demonstrate how `complex_NX` enables high-performance rendering of large repository-interaction networks, particularly those found in Topic 3.

Readers are referred to:

- `FURTHER_EXPLORATION/complex_NX.py` — full implementation of the visualization tool  

---

## 2. Instructions for Using the `complex_NX` Package

Users who wish to reproduce the visualizations or integrate the package into their own workflows can refer to:

- **`FURTHER_EXPLORATION/complex_NX.py`** — source code  
- **`FURTHER_EXPLORATION/README.md`** — usage instructions, examples, and integration notes  

The README includes:

- example import statements  
- demonstration code for both 2D and 3D graph plotting  
- parameter descriptions (edge filtering, transparency controls, community-aware coloring)  
- recommended workflows for high-density or large-scale network visualization  

These instructions support reproducibility and extensibility for users applying the visualization method shown in the main report.

---

## 3. Additional Data Cleaning for Topic 3

As part of Milestone 3, additional data-cleaning procedures were applied to the **Topic 3 `repo_activity.csv`** dataset.  
This extended cleaning includes:

- removal of malformed or incomplete repository-interaction records  
- normalization of contributor-count–based edge weights  
- filtering of low-activity repositories  
- restructuring of the dataset to better support visualization and downstream modeling  

The cleaned dataset has been uploaded as part of the Milestone 3 submission.

---

## 4. Updated Milestone 3 Submission Contents

The Milestone 3 submission now includes:

- 📄 Updated project report (with new Section on pages **7–8**)  
- 📂 Cleaned **`repo_activity.csv`** for Topic 3  
- 🧩 **`complex_NX.py`** visualization package  
- 📘 Updated README instructions for using `complex_NX`  

---

