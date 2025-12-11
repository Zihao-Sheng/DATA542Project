# **complex_NX — Advanced NetworkX Visualization Toolkit**

This toolkit extends standard NetworkX visualization with high-performance 2D and 3D graph rendering, adaptive edge-transparency, community coloring, weight-based filtering, and multiple interactive backends (**Plotly + PyVis**).

It is designed for large real-world graphs where:

- Node count may reach **thousands**
- Edge count may reach **hundreds of thousands or millions**
- Standard NetworkX drawing becomes unreadable or fails due to memory limits

---

## ✨ **Key Features**

---

### 🔵 **1. Smart Edge Transparency & Highlighting**

- Low-weight edges automatically become lighter + more transparent  
- High-weight edges are emphasized using colormaps  
- Prevents dense *“hairball”* appearance in large graphs  
- Includes **weak-edge filtering**, allowing you to:
  - Limit the number of low-weight edges per node  
  - **Or hide all weak edges entirely (`simple=True`)**

---

### 🔵 **2. Two Visualization Modes**

| Mode | Backend | Capabilities |
|------|---------|--------------|
| **3D** | Plotly | Rotatable, zoomable, smooth Bézier curve edges, colorbar, HTML export |
| **2D** | PyVis | Physics simulation, draggable nodes, community coloring, interactive tooltips |

Select via:

```python
mode="3d"   # or "2d"
```
### 🔵 **3. Community-Based Coloring (2D)**

The 2D PyVis backend supports community-based node coloring.  
Provide a dictionary mapping each node to a community index:

```python
community_partition = {node_id: community_index}
```

Nodes in the same community will automatically receive distinct discrete colors using the Tab20 colormap.

### 🔵 **4. Flexible Edge Shapes (3D)**

In 3D mode, edges can be visualized using different geometric styles:

- **"straight"** — standard linear edges connecting node coordinates  
- **"arc"** — smooth quadratic Bézier curves in 3D space  

Arc-shaped edges are particularly useful for:

- Dense graphs with heavy edge overlap  
- Highlighting important structural patterns  
- Producing more aesthetic, presentation-quality visualizations  

---

### 🔵 **5. Safe Handling of Node IDs**

PyVis requires that node identifiers must be either `int` or `str`.  
To avoid common errors such as:

```
ValueError: Node id of type XYZ is not supported
```

### 🔵 **6. Jupyter Notebook Friendly**

The entire visualization toolkit is designed to work seamlessly inside common research environments such as **Jupyter Notebook** and **JupyterLab**.

Key advantages:

- 3D mode returns a **Plotly Figure** object, allowing:
  - further layout customization  
  - adding annotations, camera settings, or legends  
  - exporting to static images (`.png`, `.svg`) or HTML  

- 2D mode (PyVis) produces an **interactive HTML graph** that:
  - supports dragging nodes  
  - shows tooltips  
  - runs physics-based layout (or static layout if physics=False)

Both backends support:

- inline display in notebooks  
- HTML export for embedding into reports or sharing with collaborators  

---

## 📦 **Installation**

Place the module in your project:

```
your_project/
├── complex_NX.py
└── your_notebook.ipynb
```

Then import it normally:

```python
from complex_NX import NX_style
```

## 🚀 Quick Start Examples

Below are minimal, ready-to-run examples demonstrating the core functionalities of `NX_style`.

---

### **Example 1 — 3D Weighted Graph (Plotly)**

```python
import networkx as nx
import numpy as np
from complex_NX import NX_style

# Build sample graph
G = nx.fast_gnp_random_graph(20, 0.3)
for u, v in G.edges():
    G[u][v]["weight"] = np.random.randint(1, 10)

fig = NX_style(
    G,
    mode="3d",
    node_size="degree",        # nodes sized by degree
    node_color="degree",       # nodes colored by degree
    edge_color_attr="weight",  # edge color based on weight
    edge_shape="arc",          # smoother curved edges
    title="3D Weighted Graph"
)
```

### **Example 2 — 2D Community Visualization (PyVis)**

This example demonstrates how `NX_style` automatically colors nodes by community and produces an interactive 2D PyVis graph.

```python
import networkx as nx
import community   # python-louvain
from complex_NX import NX_style

# Load a sample graph
G = nx.karate_club_graph()

# Detect communities using the Louvain algorithm
partition = community.best_partition(G)

# Visualize in 2D with community coloring
NX_style(
    G,
    mode="2d",
    community_partition=partition,   # assign color by community
    physics=False,                   # stable static layout (recommended for reports)
    edge_colormap=True,              # optional: color edges by weight or degree
    height=800,
    width=1200,
)
```

### **Example 3 — Filtering Weak Edges (Large Graph Handling)**

This example shows how to improve readability in dense graphs by treating low-weight edges as “weak” and either limiting or fading them.

```python
from complex_NX import NX_style

NX_style(
    G,
    mode="3d",                 # or "2d"
    weak_weights=[1],          # treat edges with weight=1 as weak
    weak_max_per_node=[2],     # draw at most 2 weak edges per node
    simple=False,              # keep weak edges, but render them thin + transparent
    title="Graph with Weak Edge Filtering"
)
```

### **Example 4 — Exporting Interactive HTML**

Both Plotly (3D) and PyVis (2D) visualizations can be exported as standalone interactive HTML files.  
This allows sharing, archiving, or embedding the graph in reports or web pages.

```python
from complex_NX import NX_style

NX_style(
    G,
    mode="3d",                  # or "2d"
    output_html="graph_output.html"   # name of the exported HTML file
)
```

The exported HTML file:

- opens in any web browser

- remains fully interactive

- supports rotation, zooming, and panning (3D Plotly)

- supports node dragging and tooltips (2D PyVis)

- requires no Python environment to view

