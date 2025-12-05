import networkx as nx
import numpy as np
from collections import defaultdict
import random
import plotly.graph_objects as go
import matplotlib.pyplot as plt


def mpl_to_plotly(cmap, n=256):
    """Convert a matplotlib colormap to a Plotly colorscale."""
    xs = np.linspace(0, 1, n)
    colors = cmap(xs)
    scale = []
    for x, (r, g, b, a) in zip(xs, colors):
        scale.append([
            float(x),
            f"rgb({int(r*255)},{int(g*255)},{int(b*255)})"
        ])
    return scale


def get_mpl_cmap(style_name):
    """Return a matplotlib colormap by string name."""
    name = (style_name or "").lower()
    cmap_dict = {
        "turbo":   plt.cm.turbo,
        "viridis": plt.cm.viridis,
        "plasma":  plt.cm.plasma,
        "inferno": plt.cm.inferno,
        "magma":   plt.cm.magma,
        "cividis": plt.cm.cividis,
        "cool":    plt.cm.cool,
        "hot":     plt.cm.hot,
        "spring":  plt.cm.spring,
        "winter":  plt.cm.winter,
    }
    return cmap_dict.get(name, plt.cm.viridis)  # default


def NX_3D_style(
    G,
    weak_weights=None,
    weak_max_per_node=None,
    simple=False,
    node_size=None,       # e.g. "Popularity"
    node_color=None,      # e.g. "Efficiency"
    node_style="turbo",
    edge_style="inferno",
    seed=42,
    k_layout=1.0,
    iterations=75,
    main='NX_3D_style Graph'
):
    """
    Plot a 3D repo graph (Plotly) with flexible weak edges and node styling.

    Parameters
    ----------
    G : nx.Graph
        Graph with:
          - edge attribute 'weight'
          - optional node attributes for sizing / coloring
    weak_weights : list[int or float] or None
        Weight values considered "weak" edges.
        These edges are drawn as gray lines (unless simple=True).
    weak_max_per_node : list[int] or None
        Same length as weak_weights.
        weak_max_per_node[i] = max number of incident edges (per node)
        with weight == weak_weights[i] that will be drawn.
        (Per-node limit.)
    simple : bool
        If True:
          - do NOT draw edges whose weight is in weak_weights
          - remove isolated nodes (nodes with degree 0 in the drawn edges)
        If False:
          - draw weak edges in gray (with per-node sampling)
          - draw all nodes (even isolated ones)
    node_size : str or None
        Name of node attribute to control marker size.
        Example: "Popularity".
        If None -> use a constant default size.
    node_color : str or None
        Name of node attribute to control marker color.
        Example: "Efficiency".
        If None -> use a constant color (no colorbar).
        If not None -> create a node colorbar labelled with this name.
    node_style : str
        Colormap style for nodes. One of:
        "turbo", "viridis", "plasma", "inferno", "magma",
        "cividis", "cool", "hot", "spring", "winter".
    edge_style : str
        Colormap style for strong edges (by weight). Same choices as node_style.
    seed : int
        Random seed for layout & sampling.
    k_layout : float
        'k' parameter for spring_layout (controls spacing).
    iterations : int
        Number of iterations in spring_layout.
    """

    random.seed(seed)
    np.random.seed(seed)

    # ---------- 0. layouts & edge weights ----------
    pos2d = nx.spring_layout(G, seed=seed, k=k_layout, iterations=iterations)

    pos3d = {}
    for n, (x, y) in pos2d.items():
        z = np.random.uniform(-1, 1)
        pos3d[n] = (x, y, z)

    edges_data = list(G.edges(data=True))

    # For edge weight normalization & edge colorbar
    if edges_data:
        all_w = np.array([d.get("weight", 1) for _, _, d in edges_data], dtype=float)
        w_min, w_max = all_w.min(), all_w.max()
    else:
        w_min, w_max = 0.0, 1.0

    edge_cmap = get_mpl_cmap(edge_style)
    edge_colorscale = mpl_to_plotly(edge_cmap)

    # ---------- 1. weak vs strong edges (per-node sampling) ----------
    if weak_weights is not None or weak_max_per_node is not None:
        if not weak_weights or not weak_max_per_node:
            raise ValueError("weak_weights and weak_max_per_node must both be given or both None.")
        if len(weak_weights) != len(weak_max_per_node):
            raise ValueError("weak_weights and weak_max_per_node must have the same length.")

        weak_set = set(weak_weights)
        weak_limit_map = dict(zip(weak_weights, weak_max_per_node))
    else:
        weak_set = set()
        weak_limit_map = {}

    # No weak weights: everything is strong
    if not weak_set:
        strong_edges = edges_data
        weak_edges = []
    else:
        edge_attr = {}
        # weak_incident[w][node] = list of edge_keys incident on node (for that weight)
        weak_incident = {w: defaultdict(list) for w in weak_set}
        strong_edges = []

        for u, v, d in edges_data:
            w = d.get("weight", 1)
            key = tuple(sorted((u, v)))
            edge_attr[key] = (u, v, d)

            if w in weak_set:
                weak_incident[w][u].append(key)
                weak_incident[w][v].append(key)
            else:
                strong_edges.append((u, v, d))

        chosen_weak_keys = set()
        for w in weak_set:
            max_per_node = weak_limit_map[w]
            for node, edge_keys in weak_incident[w].items():
                if len(edge_keys) <= max_per_node:
                    chosen = edge_keys
                else:
                    chosen = random.sample(edge_keys, max_per_node)
                chosen_weak_keys.update(chosen)

        weak_edges = [edge_attr[k] for k in chosen_weak_keys]

    # simple=True: do not draw weak edges at all
    if simple and weak_set:
        weak_edges = []

    # ---------- 2. decide which nodes to draw ----------
    all_nodes = list(G.nodes())

    if simple:
        nodes_in_edges = set()
        for u, v, _ in strong_edges + weak_edges:
            nodes_in_edges.add(u)
            nodes_in_edges.add(v)
        draw_nodes = [n for n in all_nodes if n in nodes_in_edges]
    else:
        draw_nodes = all_nodes

    # ---------- 3. build edge traces ----------
    def norm_w(w):
        return (w - w_min) / (w_max - w_min + 1e-9)

    # weak edges: single gray trace
    data_traces = []

    if weak_edges:
        edge_x_weak, edge_y_weak, edge_z_weak = [], [], []
        for u, v, d in weak_edges:
            x0, y0, z0 = pos3d[u]
            x1, y1, z1 = pos3d[v]
            edge_x_weak += [x0, x1, None]
            edge_y_weak += [y0, y1, None]
            edge_z_weak += [z0, z1, None]

        edge_trace_weak = go.Scatter3d(
            x=edge_x_weak,
            y=edge_y_weak,
            z=edge_z_weak,
            mode='lines',
            line=dict(width=1, color='rgba(200,200,200,0.25)'),
            hoverinfo='none',
            showlegend=False
        )
        data_traces.append(edge_trace_weak)

    # strong edges: one trace per edge with inferno-like colors
    for u, v, d in strong_edges:
        x0, y0, z0 = pos3d[u]
        x1, y1, z1 = pos3d[v]
        w = d.get("weight", 1)
        nw = norm_w(w)

        rgba = edge_cmap(nw)
        color_str = f"rgba({int(rgba[0]*255)},{int(rgba[1]*255)},{int(rgba[2]*255)},{rgba[3]:.2f})"

        trace = go.Scatter3d(
            x=[x0, x1],
            y=[y0, y1],
            z=[z0, z1],
            mode='lines',
            line=dict(
                width=2.0 + 4 * nw,
                color=color_str
            ),
            hoverinfo='none',
            showlegend=False
        )
        data_traces.append(trace)

    # ---------- 4. nodes ----------
    x_nodes = [pos3d[n][0] for n in draw_nodes]
    y_nodes = [pos3d[n][1] for n in draw_nodes]
    z_nodes = [pos3d[n][2] for n in draw_nodes]

    # node size mapping
    if node_size is None or len(draw_nodes) == 0:
        size_values = np.ones(len(draw_nodes)) * 10.0 if draw_nodes else []
    else:
        vals = np.array([G.nodes[n].get(node_size, 0.0) for n in draw_nodes], dtype=float)
        vmin, vmax = vals.min(), vals.max()
        size_values = 5 + 25 * (vals - vmin) / (vmax - vmin + 1e-9)

    # node color mapping
    marker_kwargs = {}
    if node_color is None or len(draw_nodes) == 0:
        marker_kwargs["color"] = "royalblue"
    else:
        cmap_node = get_mpl_cmap(node_style)
        node_colorscale = mpl_to_plotly(cmap_node)

        vals_c = np.array([G.nodes[n].get(node_color, 0.0) for n in draw_nodes], dtype=float)
        cmin, cmax = vals_c.min(), vals_c.max()
        norm_c = (vals_c - cmin) / (cmax - cmin + 1e-9)

        marker_kwargs["color"] = norm_c
        marker_kwargs["colorscale"] = node_colorscale
        marker_kwargs["cmin"] = 0
        marker_kwargs["cmax"] = 1
        # show node colorbar only when node_color is provided
        marker_kwargs["colorbar"] = dict(
            title=node_color,
            x=1.05
        )

    # hover text
    hover_text = []
    for n in draw_nodes:
        eff = G.nodes[n].get("Efficiency", None)
        pop = G.nodes[n].get("Popularity", None)
        parts = [f"id: {n}"]
        if eff is not None:
            parts.append(f"Efficiency: {eff:.2f}")
        if pop is not None:
            parts.append(f"Popularity: {pop:.2f}")
        hover_text.append("<br>".join(parts))

    node_trace = go.Scatter3d(
        x=x_nodes,
        y=y_nodes,
        z=z_nodes,
        mode='markers',
        hoverinfo='text',
        text=hover_text,
        marker=dict(
            size=size_values,
            line=dict(width=0.5, color='black'),
            **marker_kwargs
        ),
        showlegend=False
    )
    data_traces.append(node_trace)

    # Edge Weight colorbar (dummy trace)
    edge_colorbar_dummy = go.Scatter3d(
        x=[None],
        y=[None],
        z=[None],
        mode='markers',
        marker=dict(
            size=0,
            color=[w_min, w_max],
            colorscale=edge_colorscale,
            cmin=w_min,
            cmax=w_max,
            colorbar=dict(
                title="Edge Weight",
                x=1.18
            )
        ),
        showlegend=False,
        hoverinfo='none'
    )
    data_traces.append(edge_colorbar_dummy)

    # ---------- 5. assemble ----------
    fig = go.Figure(data=data_traces)

    fig.update_layout(
        width=1000,
        height=1000,
        title=main,
        showlegend=False,
        scene=dict(
            xaxis=dict(showbackground=False, visible=False),
            yaxis=dict(showbackground=False, visible=False),
            zaxis=dict(showbackground=False, visible=False),
            aspectmode='data'
        ),
    )

    fig.show()
