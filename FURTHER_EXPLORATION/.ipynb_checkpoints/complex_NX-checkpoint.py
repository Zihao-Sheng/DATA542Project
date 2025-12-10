from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import math
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

try:
    import plotly.graph_objects as go
except ImportError as e:  # pragma: no cover - environment-dependent
    go = None  # type: ignore

try:
    from pyvis.network import Network
    _PYVIS_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    Network = None  # type: ignore
    _PYVIS_AVAILABLE = False


ColorStyle = str


def _mpl_to_plotly(cmap, n: int = 256) -> List[List[Union[float, str]]]:
    """Convert a matplotlib colormap to a Plotly colorscale."""
    xs = np.linspace(0, 1, n)
    colors = cmap(xs)
    scale: List[List[Union[float, str]]] = []
    for x, (r, g, b, a) in zip(xs, colors):
        scale.append(
            [float(x), f"rgb({int(r * 255)},{int(g * 255)},{int(b * 255)})"]
        )
    return scale


def _get_mpl_cmap(style_name: ColorStyle):
    """Return a matplotlib colormap by string name, with a safe default."""
    name = (style_name or "").lower()
    cmap_dict = {
        "turbo": plt.cm.turbo,
        "viridis": plt.cm.viridis,
        "plasma": plt.cm.plasma,
        "inferno": plt.cm.inferno,
        "magma": plt.cm.magma,
        "cividis": plt.cm.cividis,
        "cool": plt.cm.cool,
        "hot": plt.cm.hot,
        "spring": plt.cm.spring,
        "winter": plt.cm.winter,
    }
    return cmap_dict.get(name, plt.cm.viridis)


def _normalize_values(
    values: np.ndarray, out_min: float, out_max: float
) -> np.ndarray:
    """Linearly map a 1D array to [out_min, out_max]."""
    if values.size == 0:
        return np.array([])
    vmin = float(np.nanmin(values))
    vmax = float(np.nanmax(values))
    if math.isclose(vmin, vmax):
        return np.full_like(values, (out_min + out_max) / 2.0)
    norm = (values - vmin) / (vmax - vmin)
    return out_min + norm * (out_max - out_min)


def _normalize_0_1(values: np.ndarray) -> np.ndarray:
    """Normalize a 1D array to [0, 1]."""
    if values.size == 0:
        return np.array([])
    vmin = float(np.nanmin(values))
    vmax = float(np.nanmax(values))
    if math.isclose(vmin, vmax):
        return np.zeros_like(values)
    return (values - vmin) / (vmax - vmin)


def _build_hover_text(
    node_id: Any,
    attrs: Dict[str, Any],
    hover_attrs: Optional[Union[str, Iterable[str]]],
) -> str:
    """
    Build a multi-line hover string for a node.

    hover_attrs:
        - "all": show all attributes (sorted by key)
        - iterable: only those keys
        - None: only show id
    """
    lines: List[str] = [f"id: {node_id}"]

    if hover_attrs is None:
        return "<br>".join(lines)

    if isinstance(hover_attrs, str):
        if hover_attrs.lower() != "all":
            # single attribute name
            key = hover_attrs
            if key in attrs:
                lines.append(f"{key}: {attrs[key]}")
            return "<br>".join(lines)
        # "all"
        for key in sorted(attrs.keys()):
            lines.append(f"{key}: {attrs[key]}")
        return "<br>".join(lines)

    # iterable of keys
    for key in hover_attrs:
        if key in attrs:
            lines.append(f"{key}: {attrs[key]}")
    return "<br>".join(lines)


def NX_style(
    G: nx.Graph,
    mode: str = "3d",  # "3d" (Plotly) or "2d" (PyVis)
    # ---- node visuals ----
    node_size: Optional[str] = None,
    node_color: Optional[str] = None,
    node_style: ColorStyle = "turbo",
    node_size_range: Tuple[float, float] = (5.0, 30.0),
    # ---- edge visuals ----
    edge_color_attr: Optional[str] = "weight",
    edge_style: ColorStyle = "inferno",
    edge_width_range: Tuple[float, float] = (1.0, 6.0),
    weak_weights: Optional[List[Union[int, float]]] = None,
    weak_max_per_node: Optional[List[int]] = None,
    simple: bool = False,
    # ---- filtering & layout ----
    hide_isolates: Optional[bool] = None,
    layout: str = "spring",
    seed: int = 42,
    # ---- display & export ----
    hover_attrs: Optional[Union[str, Iterable[str]]] = "all",
    output_html: Optional[str] = None,
    notebook: bool = True,
    show: bool = True,
    # ---- global size & 2D resources & community coloring ----
    height: int = 800,
    width: int = 1000,
    cdn_resources: str = "in_line",
    community_partition: Optional[Dict[Any, int]] = None,
    # ---- 2D edge colormap switch & physics switch ----
    edge_colormap: bool = True,
    physics: bool = False,
    # ---- 3D edge shape ----
    edge_shape: str = "straight",  # "straight" or "arc"
):
    """
    Unified entry point for interactive NetworkX graph visualization.

    Parameters
    ----------
    G :
        NetworkX graph (Graph / DiGraph / etc.).
    mode :
        "3d" for Plotly 3D, "2d" for PyVis 2D interactive.
    node_size :
        Node attribute name used to control marker size.
    node_color :
        Node attribute name used to control marker color.
    node_style :
        Colormap style name for node colors.
    node_size_range :
        (min_size, max_size) for node marker sizes.
    edge_color_attr :
        Edge attribute name used to color strong edges. Default "weight".
    edge_style :
        Colormap style for edges (when edge_colormap=True).
    edge_width_range :
        (min_width, max_width) for strong edge line widths.
    weak_weights / weak_max_per_node :
        Define which weights are "weak" and how many per node to draw.
    simple :
        If True, weak edges are not drawn at all.
    hide_isolates :
        If None, defaults to hide_isolates = simple.
    layout :
        Currently only "spring" is supported.
    seed :
        Random seed for layout and sampling.
    hover_attrs :
        Which node attributes appear in hover text.
    output_html :
        If not None, write a stand-alone HTML.
    notebook :
        For PyVis 2D integration in Jupyter.
    show :
        If True, display the figure/HTML.
    height, width :
        Global size of the visualization.
    cdn_resources :
        PyVis CDN mode: "in_line", "remote", or "local".
    community_partition :
        Dict[node -> community_id] for discrete community coloring (2D).
    edge_colormap :
        2D only:
        - True  (default): strong edges use colormap by edge_color_attr
        - False: strong edges are uniform gray, only width varies.
    physics :
        2D only: control vis.js physics (True=dynamic, False=static).
    edge_shape :
        3D only:
        - "straight": straight line edges
        - "arc": smooth Bezier curves
    """
    if mode not in {"3d", "2d"}:
        raise ValueError("mode must be '3d' or '2d'")

    if edge_shape not in {"straight", "arc"}:
        raise ValueError("edge_shape must be 'straight' or 'arc'")

    # Determine effective hide_isolates if not explicitly set
    if hide_isolates is None:
        hide_isolates_effective = simple
    else:
        hide_isolates_effective = hide_isolates

    # Work on an undirected view for visualization
    if isinstance(G, (nx.DiGraph, nx.MultiDiGraph)):
        H = G.to_undirected()
    else:
        H = G

    # --- 1. classify edges into strong / weak and determine which edges to draw ---
    rng = np.random.RandomState(seed)
    edges_data = list(H.edges(data=True))

    # Determine which weights are weak
    if weak_weights is None or len(weak_weights) == 0:
        weak_set = set()
    else:
        weak_set = set(weak_weights)

    strong_edges: List[Tuple[Any, Any, Dict[str, Any]]] = []
    weak_edges_all: List[Tuple[Any, Any, Dict[str, Any]]] = []

    from collections import defaultdict
    weak_incident: Dict[Union[int, float], Dict[Any, List[Tuple[Any, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )

    for u, v, d in edges_data:
        w = d.get("weight", 1)
        if w in weak_set:
            weak_edges_all.append((u, v, d))
            weak_incident[w][u].append((u, v))
            weak_incident[w][v].append((u, v))
        else:
            strong_edges.append((u, v, d))

    # Decide which weak edges are actually drawn (sampling per node if requested)
    weak_edges_drawn: List[Tuple[Any, Any, Dict[str, Any]]] = []

    if weak_set and not simple:
        if weak_max_per_node is not None:
            if len(weak_max_per_node) != len(weak_weights):
                raise ValueError(
                    "weak_max_per_node must have the same length as weak_weights"
                )
            weight_to_cap = dict(zip(weak_weights, weak_max_per_node))
        else:
            weight_to_cap = {w: float("inf") for w in weak_set}

        chosen_keys = set()
        for w in weak_set:
            cap = weight_to_cap.get(w, float("inf"))
            incident_dict = weak_incident[w]
            for node, edge_pairs in incident_dict.items():
                if len(edge_pairs) <= cap:
                    chosen = edge_pairs
                else:
                    idx = rng.choice(len(edge_pairs), size=int(cap), replace=False)
                    chosen = [edge_pairs[i] for i in idx]
                for e in chosen:
                    chosen_keys.add(tuple(sorted(e)))

        edge_lookup = {
            (min(u, v), max(u, v)): (u, v, d) for (u, v, d) in weak_edges_all
        }
        for key in chosen_keys:
            weak_edges_drawn.append(edge_lookup[key])

    # --- 2. determine which nodes to draw based on hide_isolates ---
    if hide_isolates_effective:
        incident_nodes = set()
        for u, v, _ in strong_edges + weak_edges_drawn:
            incident_nodes.add(u)
            incident_nodes.add(v)
        nodes_to_draw = [n for n in H.nodes() if n in incident_nodes]
    else:
        nodes_to_draw = list(H.nodes())

    # --- 3. backend ---
    if mode == "3d":
        if go is None:
            raise ImportError(
                "plotly is required for mode='3d'. Please install plotly."
            )
        fig = _plot_3d_plotly(
            H,
            nodes_to_draw,
            strong_edges,
            weak_edges_drawn,
            node_size=node_size,
            node_color=node_color,
            node_style=node_style,
            node_size_range=node_size_range,
            edge_color_attr=edge_color_attr,
            edge_style=edge_style,
            edge_width_range=edge_width_range,
            layout=layout,
            seed=seed,
            hover_attrs=hover_attrs,
            output_html=output_html,
            show=show,
            height=height,
            width=width,
            edge_shape=edge_shape,
        )
        return fig

    # mode == "2d"
    if not _PYVIS_AVAILABLE:
        raise ImportError(
            "PyVis is required for mode='2d'. Please install pyvis "
            "(`pip install pyvis`)."
        )
    net = _plot_2d_pyvis(
        H,
        nodes_to_draw,
        strong_edges,
        weak_edges_drawn,
        node_size=node_size,
        node_color=node_color,
        node_style=node_style,
        node_size_range=node_size_range,
        edge_color_attr=edge_color_attr,
        edge_style=edge_style,
        edge_width_range=edge_width_range,
        layout=layout,
        seed=seed,
        hover_attrs=hover_attrs,
        output_html=output_html,
        notebook=notebook,
        show=show,
        height=height,
        width=width,
        cdn_resources=cdn_resources,
        community_partition=community_partition,
        edge_colormap=edge_colormap,
        physics=physics,
    )
    return net


def _bezier_curve_3d(
    p0: Tuple[float, float, float],
    p1: Tuple[float, float, float],
    bend: float = 0.15,
    n_points: int = 15,
) -> Tuple[List[float], List[float], List[float]]:
    """
    Generate a smooth 3D quadratic Bezier curve between p0 and p1.

    - bend: how strong the bowing is (0.1 ~ 0.2 usually looks good)
    - n_points: how many samples along the curve
    """
    p0 = np.array(p0, dtype=float)
    p1 = np.array(p1, dtype=float)
    v = p1 - p0
    if np.allclose(v, 0):
        # degenerate edge
        return [p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]]

    # middle point
    mid = 0.5 * (p0 + p1)

    # pick a reference axis that is not parallel to v
    ref = np.array([0.0, 0.0, 1.0])
    if np.allclose(np.cross(v, ref), 0):
        ref = np.array([0.0, 1.0, 0.0])

    # normal direction roughly orthogonal to v
    n = np.cross(v, ref)
    n_norm = np.linalg.norm(n)
    if n_norm < 1e-9:
        # fallback: no good normal, use straight line
        return [p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]]
    n = n / n_norm

    # control point: mid + offset along normal
    length = np.linalg.norm(v)
    control = mid + bend * length * n

    ts = np.linspace(0.0, 1.0, n_points)
    xs, ys, zs = [], [], []
    for t in ts:
        # quadratic Bezier: B(t) = (1-t)^2 p0 + 2(1-t)t control + t^2 p1
        p = (1 - t) ** 2 * p0 + 2 * (1 - t) * t * control + t ** 2 * p1
        xs.append(float(p[0]))
        ys.append(float(p[1]))
        zs.append(float(p[2]))
    return xs, ys, zs


def _plot_3d_plotly(
    G: nx.Graph,
    nodes_to_draw: List[Any],
    strong_edges: List[Tuple[Any, Any, Dict[str, Any]]],
    weak_edges_drawn: List[Tuple[Any, Any, Dict[str, Any]]],
    *,
    node_size: Optional[str],
    node_color: Optional[str],
    node_style: ColorStyle,
    node_size_range: Tuple[float, float],
    edge_color_attr: Optional[str],
    edge_style: ColorStyle,
    edge_width_range: Tuple[float, float],
    layout: str,
    seed: int,
    hover_attrs: Optional[Union[str, Iterable[str]]],
    output_html: Optional[str],
    show: bool,
    height: int,
    width: int,
    edge_shape: str,
):
    """Backend: Plotly 3D visualization."""
    if layout != "spring":
        raise ValueError("Currently only layout='spring' is supported for 3D.")

    if nodes_to_draw:
        H_sub = G.subgraph(nodes_to_draw)
    else:
        H_sub = G.subgraph([])

    pos2d = nx.spring_layout(H_sub, seed=seed, k=None)

    rng = np.random.RandomState(seed)
    pos3d: Dict[Any, Tuple[float, float, float]] = {}
    for n, (x, y) in pos2d.items():
        z = rng.uniform(-1.0, 1.0)
        pos3d[n] = (x, y, z)

    for n in nodes_to_draw:
        if n not in pos3d:
            pos3d[n] = (0.0, 0.0, 0.0)

    # --- node sizes ---
    if node_size is not None and nodes_to_draw:
        vals = np.array(
            [G.nodes[n].get(node_size, 0.0) for n in nodes_to_draw], dtype=float
        )
        size_values = _normalize_values(vals, node_size_range[0], node_size_range[1])
    else:
        size_values = np.full(
            len(nodes_to_draw),
            (node_size_range[0] + node_size_range[1]) / 2.0,
        )

    # --- node colors ---
    marker_kwargs: Dict[str, Any] = {}
    if node_color is not None and nodes_to_draw:
        cmap_node = _get_mpl_cmap(node_style)
        node_colorscale = _mpl_to_plotly(cmap_node)

        vals_c = np.array(
            [G.nodes[n].get(node_color, 0.0) for n in nodes_to_draw], dtype=float
        )
        norm_c = _normalize_0_1(vals_c)

        marker_kwargs["color"] = norm_c
        marker_kwargs["colorscale"] = node_colorscale
        marker_kwargs["cmin"] = 0
        marker_kwargs["cmax"] = 1
        marker_kwargs["colorbar"] = dict(title=node_color, x=1.05)
    else:
        marker_kwargs["color"] = "royalblue"

    x_nodes = [pos3d[n][0] for n in nodes_to_draw]
    y_nodes = [pos3d[n][1] for n in nodes_to_draw]
    z_nodes = [pos3d[n][2] for n in nodes_to_draw]

    hover_texts = [
        _build_hover_text(n, dict(G.nodes[n]), hover_attrs) for n in nodes_to_draw
    ]

    node_trace = go.Scatter3d(
        x=x_nodes,
        y=y_nodes,
        z=z_nodes,
        mode="markers",
        hoverinfo="text",
        text=hover_texts,
        marker=dict(
            size=size_values,
            line=dict(width=0.5, color="black"),
            **marker_kwargs,
        ),
        showlegend=False,
    )

    # --- edge colors (for strong edges) with robust normalization ---
    strong_edges_for_color = strong_edges
    if edge_color_attr is not None and strong_edges_for_color:
        edge_vals = np.array(
            [
                e[2].get(edge_color_attr, 1.0)
                for e in strong_edges_for_color
            ],
            dtype=float,
        )
        # robust normalization: avoid range 太大 / 太小 导致全透明
        vmin = float(np.nanmin(edge_vals))
        vmax = float(np.nanmax(edge_vals))
        if math.isclose(vmin, vmax):
            edge_norm = np.zeros_like(edge_vals)
        else:
            q_low = float(np.quantile(edge_vals, 0.10))
            q_high = float(np.quantile(edge_vals, 0.90))
            if math.isclose(q_low, q_high):
                # fallback to min/max
                edge_norm = _normalize_0_1(edge_vals)
            else:
                edge_norm = (edge_vals - q_low) / (q_high - q_low)
                edge_norm = np.clip(edge_norm, 0.0, 1.0)
    else:
        edge_norm = np.zeros(len(strong_edges_for_color))

    edge_cmap = _get_mpl_cmap(edge_style)

    # 淡色 + 透明度修正：低 weight 越靠近浅灰 + 更透明（但不消失）
    light_rgb = (230, 230, 230)

    def edge_rgba_for_index(idx: int) -> str:
        t = float(edge_norm[idx])  # 0~1 after robust norm
        if edge_color_attr is None:
            base_rgba = edge_cmap(0.5)
        else:
            base_rgba = edge_cmap(t)

        br, bg, bb, _ = base_rgba
        br, bg, bb = int(br * 255), int(bg * 255), int(bb * 255)

        # light_strength: 低 t -> 更靠近浅灰；高 t -> 更靠近原色
        light_strength = (1.0 - t) ** 0.7
        lr, lg, lb = light_rgb

        rr = int((1 - light_strength) * br + light_strength * lr)
        rg = int((1 - light_strength) * bg + light_strength * lg)
        rb = int((1 - light_strength) * bb + light_strength * lb)

        # alpha: 低 t → 0.40, 高 t → 1.0 左右，不会透明到没了
        alpha = 0.40 + 0.60 * t
        return f"rgba({rr},{rg},{rb},{alpha:.2f})"

    # strong edges traces（支持直线 / 弧线）
    strong_traces = []
    for idx, (u, v, d) in enumerate(strong_edges_for_color):
        if (u not in pos3d) or (v not in pos3d):
            continue
        p0 = pos3d[u]
        p1 = pos3d[v]

        if edge_shape == "arc":
            xs, ys, zs = _bezier_curve_3d(
                p0,
                p1,
                bend=0.15,
                n_points=16,
            )
        else:
            # straight line
            xs = [p0[0], p1[0]]
            ys = [p0[1], p1[1]]
            zs = [p0[2], p1[2]]

        color_str = edge_rgba_for_index(idx)

        if edge_color_attr is not None:
            base_width = _normalize_values(
                np.array([d.get(edge_color_attr, 1.0)]),
                edge_width_range[0], edge_width_range[1]
            )[0]
        else:
            base_width = (
                edge_width_range[0] + edge_width_range[1]
            ) / 2.0

        strong_traces.append(
            go.Scatter3d(
                x=xs,
                y=ys,
                z=zs,
                mode="lines",
                line=dict(width=base_width, color=color_str),
                hoverinfo="none",
                showlegend=False,
            )
        )

    # weak edges: single gray trace
    if weak_edges_drawn:
        edge_x_weak, edge_y_weak, edge_z_weak = [], [], []
        for u, v, d in weak_edges_drawn:
            if (u not in pos3d) or (v not in pos3d):
                continue
            x0, y0, z0 = pos3d[u]
            x1, y1, z1 = pos3d[v]
            edge_x_weak += [x0, x1, None]
            edge_y_weak += [y0, y1, None]
            edge_z_weak += [z0, z1, None]

        weak_trace = go.Scatter3d(
            x=edge_x_weak,
            y=edge_y_weak,
            z=edge_z_weak,
            mode="lines",
            line=dict(width=edge_width_range[0],
                      color="rgba(210,210,210,0.30)"),
            hoverinfo="none",
            showlegend=False,
        )
    else:
        weak_trace = None

    # Edge colorbar (dummy marker)
    if edge_color_attr is not None and strong_edges_for_color:
        vals = np.array(
            [d.get(edge_color_attr, 1.0) for _, _, d in strong_edges_for_color],
            dtype=float,
        )
        vmin = float(np.nanmin(vals))
        vmax = float(np.nanmax(vals))
        colorscale_edge = _mpl_to_plotly(edge_cmap)
        edge_colorbar_dummy = go.Scatter3d(
            x=[None],
            y=[None],
            z=[None],
            mode="markers",
            marker=dict(
                size=0,
                color=[vmin, vmax],
                colorscale=colorscale_edge,
                cmin=vmin,
                cmax=vmax,
                colorbar=dict(title=edge_color_attr, x=1.18),
            ),
            hoverinfo="none",
            showlegend=False,
        )
    else:
        edge_colorbar_dummy = None

    data = []
    if weak_trace is not None:
        data.append(weak_trace)
    data.extend(strong_traces)
    data.append(node_trace)
    if edge_colorbar_dummy is not None:
        data.append(edge_colorbar_dummy)

    fig = go.Figure(data=data)
    fig.update_layout(
        width=width,
        height=height,
        title="Network (3D)",
        showlegend=False,
        scene=dict(
            xaxis=dict(showbackground=False, visible=False),
            yaxis=dict(showbackground=False, visible=False),
            zaxis=dict(showbackground=False, visible=False),
            aspectmode="data",
        ),
    )

    if output_html is not None:
        fig.write_html(output_html, include_plotlyjs="cdn")

    if show:
        fig.show()
        return None

    return fig


def _plot_2d_pyvis(
    G: nx.Graph,
    nodes_to_draw: List[Any],
    strong_edges: List[Tuple[Any, Any, Dict[str, Any]]],
    weak_edges_drawn: List[Tuple[Any, Any, Dict[str, Any]]],
    *,
    node_size: Optional[str],
    node_color: Optional[str],
    node_style: ColorStyle,
    node_size_range: Tuple[float, float],
    edge_color_attr: Optional[str],
    edge_style: ColorStyle,
    edge_width_range: Tuple[float, float],
    layout: str,
    seed: int,
    hover_attrs: Optional[Union[str, Iterable[str]]],
    output_html: Optional[str],
    notebook: bool,
    show: bool,
    height: int,
    width: int,
    cdn_resources: str,
    community_partition: Optional[Dict[Any, int]],
    edge_colormap: bool,
    physics: bool,
):
    """Backend: PyVis 2D interactive visualization."""
    if not _PYVIS_AVAILABLE:
        raise ImportError(
            "PyVis is required for mode='2d'. Please install pyvis "
            "(`pip install pyvis`)."
        )

    height_str = f"{height}px" if isinstance(height, (int, float)) else str(height)
    width_str = f"{width}px" if isinstance(width, (int, float)) else str(width)

    net = Network(
        height=height_str,
        width=width_str,
        notebook=notebook,
        cdn_resources=cdn_resources,
    )

    # ---------- Map original node IDs to PyVis-safe IDs ----------
    id_map: Dict[Any, Union[int, str]] = {}
    for n in nodes_to_draw:
        if isinstance(n, (int, str)):
            id_map[n] = n
        elif isinstance(n, np.integer):
            id_map[n] = int(n)
        else:
            id_map[n] = str(n)

    # ---------- 2D layout ----------
    if layout != "spring":
        raise ValueError("Currently only layout='spring' is supported for 2D.")

    if nodes_to_draw:
        H_sub = G.subgraph(nodes_to_draw)
        pos = nx.spring_layout(H_sub, seed=seed, k=None)
    else:
        pos = {}

    # Node size values
    if node_size is not None and nodes_to_draw:
        vals = np.array(
            [G.nodes[n].get(node_size, 0.0) for n in nodes_to_draw],
            dtype=float,
        )
        size_values = _normalize_values(vals, node_size_range[0], node_size_range[1])
    else:
        size_values = np.full(
            len(nodes_to_draw),
            (node_size_range[0] + node_size_range[1]) / 2.0,
        )

    # Node colors
    colors: List[str] = []

    if community_partition is not None and nodes_to_draw:
        cmap_disc = plt.cm.tab20
        coms = [community_partition.get(n, -1) for n in nodes_to_draw]
        unique_coms = sorted(set(coms))
        com_to_idx = {c: i for i, c in enumerate(unique_coms)}
        n_colors = max(len(unique_coms), 1)

        for n in nodes_to_draw:
            c = community_partition.get(n, -1)
            idx = com_to_idx[c]
            v = idx / max(n_colors - 1, 1)
            r, g, b, a = cmap_disc(v)
            colors.append(f"rgb({int(r*255)},{int(g*255)},{int(b*255)})")

    elif node_color is not None and nodes_to_draw:
        cmap_node = _get_mpl_cmap(node_style)
        vals_c = np.array(
            [G.nodes[n].get(node_color, 0.0) for n in nodes_to_draw], dtype=float
        )
        norm_c = _normalize_0_1(vals_c)
        for v in norm_c:
            r, g, b, a = cmap_node(v)
            colors.append(f"rgb({int(r * 255)},{int(g * 255)},{int(b * 255)})")

    else:
        colors = ["#4169E1"] * len(nodes_to_draw)

    # Add nodes
    for n, size, color in zip(nodes_to_draw, size_values, colors):
        attrs = dict(G.nodes[n])
        title = _build_hover_text(n, attrs, hover_attrs)
        nid = id_map[n]
        x, y = pos.get(n, (0.0, 0.0))
        net.add_node(
            nid,
            label=str(n),
            title=title,
            size=float(size),
            color=color,
            x=float(x),
            y=float(y),
        )

    node_set = set(nodes_to_draw)
    strong_edges_for_color = [
        (u, v, d)
        for (u, v, d) in strong_edges
        if (u in node_set and v in node_set)
    ]
    weak_edges_for_color = [
        (u, v, d)
        for (u, v, d) in weak_edges_drawn
        if (u in node_set and v in node_set)
    ]

    # Strong edge widths
    if strong_edges_for_color:
        if edge_color_attr is not None:
            vals_e = np.array(
                [d.get(edge_color_attr, 1.0) for _, _, d in strong_edges_for_color],
                dtype=float,
            )
            min_w = max(edge_width_range[0], 1.2)
            max_w = max(edge_width_range[1], min_w + 1.0)
            width_vals = _normalize_values(vals_e, min_w, max_w)
        else:
            width_vals = np.full(
                len(strong_edges_for_color),
                max((edge_width_range[0] + edge_width_range[1]) / 2.0, 1.5),
            )
    else:
        width_vals = np.array([])

    strong_edge_colors: List[str] = []

    if edge_colormap and edge_color_attr is not None and strong_edges_for_color:
        edge_cmap = _get_mpl_cmap(edge_style)
        vals_e = np.array(
            [d.get(edge_color_attr, 1.0) for _, _, d in strong_edges_for_color],
            dtype=float,
        )
        norm_e = _normalize_0_1(vals_e)
        for v in norm_e:
            r, g, b, a = edge_cmap(v)
            strong_edge_colors.append(
                f"rgba({int(r*255)},{int(g*255)},{int(b*255)},{a:.2f})"
            )
    else:
        strong_edge_colors = ["rgba(160,160,160,0.8)"] * len(strong_edges_for_color)

    # Add strong edges
    for (u, v, d), w_width, col in zip(
        strong_edges_for_color, width_vals, strong_edge_colors
    ):
        uid = id_map.get(u, u)
        vid = id_map.get(v, v)
        net.add_edge(
            uid,
            vid,
            value=float(w_width),
            color=col,
        )

    # Weak edges: 灰色、略细
    weak_width = max(edge_width_range[0] * 0.9, 1.0)
    for u, v, d in weak_edges_for_color:
        uid = id_map.get(u, u)
        vid = id_map.get(v, v)
        net.add_edge(
            uid,
            vid,
            value=weak_width,
            color="rgba(210,210,210,0.5)",
        )

    physics_enabled_str = "true" if physics else "false"

    net.set_options(
        f"""
        {{
          "physics": {{
            "enabled": {physics_enabled_str},
            "solver": "barnesHut",
            "stabilization": {{
              "enabled": true,
              "iterations": 150,
              "fit": true
            }},
            "minVelocity": 1
          }},
          "interaction": {{
            "dragNodes": true,
            "dragView": true,
            "zoomView": true
          }}
        }}
        """
    )

    if output_html is not None:
        filename = output_html
    else:
        filename = "_nx_2d_temp.html"

    try:
        html = net.generate_html(name=filename, notebook=notebook)
    except TypeError:
        html = net.generate_html(notebook=notebook)

    with open(filename, "w", encoding="utf-8") as f:
        f.write(html)

    if show and notebook:
        try:
            from IPython.display import HTML, display
            display(HTML(html))
        except Exception:
            pass
    elif show and not notebook:
        import os, webbrowser
        webbrowser.open(os.path.abspath(filename))

    return net


def NX_3d_style(
    G: nx.Graph,
    **kwargs: Any,
):
    """
    Compatibility wrapper for older code using NX_3d_style.

    Simply forwards to NX_style(..., mode="3d", **kwargs).
    """
    return NX_style(G, mode="3d", **kwargs)
