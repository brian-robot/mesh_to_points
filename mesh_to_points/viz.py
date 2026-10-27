"""Visualisation: interactive Plotly HTML and a static matplotlib preview."""
from __future__ import annotations

import numpy as np

from .segment import REGION_COLORS, REGION_NAMES, UNREACHABLE_COLOR


def preview_png(result, path: str, max_arrows: int = 300, dpi: int = 110, title: str | None = None):
    """Static 4-panel preview (top / front / side / 3-D): mesh context in grey, points
    coloured by order, strokes as lines, transitions dashed, a subset of arrows."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import cm

    pts = result.points
    mesh = result.mesh
    n = len(pts)
    spacing = result.params["spacing"]
    fig = plt.figure(figsize=(15, 11))
    views = [("top (XY)", 0, 1), ("front (XZ)", 0, 2), ("side (YZ)", 1, 2)]
    colors = cm.viridis(np.linspace(0, 1, max(n, 2)))[:n]
    rng = np.random.default_rng(0)
    vsub = mesh.vertices[rng.choice(len(mesh.vertices), min(6000, len(mesh.vertices)), replace=False)]
    sel = result.seg.select(result.params["include"], result.params["exclude"], result.params["reachable_only"])
    csel = mesh.triangles_center[sel]
    csel = csel[rng.choice(len(csel), min(6000, len(csel)), replace=False)] if len(csel) else csel
    arrow_idx = np.linspace(0, n - 1, min(max_arrows, n)).astype(int) if n else np.zeros(0, int)
    for k, (name, a, b) in enumerate(views):
        ax = fig.add_subplot(2, 2, k + 1)
        ax.scatter(vsub[:, a], vsub[:, b], s=1, c="#d0d0d0", alpha=0.5, linewidths=0)
        if len(csel):
            ax.scatter(csel[:, a], csel[:, b], s=1, c="#9ecae1", alpha=0.5, linewidths=0)
        if n:
            _draw_path_2d(ax, pts, a, b)
            ax.scatter(pts.pos[:, a], pts.pos[:, b], s=6, c=colors, linewidths=0, zorder=3)
            ax.quiver(pts.pos[arrow_idx, a], pts.pos[arrow_idx, b], pts.tool_dir[arrow_idx, a] * spacing,
                      pts.tool_dir[arrow_idx, b] * spacing, color="#d62728", angles="xy", scale_units="xy",
                      scale=1, width=0.002, zorder=4)
            ax.quiver(pts.pos[arrow_idx, a], pts.pos[arrow_idx, b], pts.normal[arrow_idx, a] * spacing,
                      pts.normal[arrow_idx, b] * spacing, color="#2ca02c", angles="xy", scale_units="xy",
                      scale=1, width=0.002, zorder=4)
            ax.scatter(pts.pos[:1, a], pts.pos[:1, b], s=80, marker="*", c="k", zorder=5)
        ax.set_aspect("equal")
        ax.set_title(name)
        ax.grid(alpha=0.3)
    ax = fig.add_subplot(2, 2, 4, projection="3d")
    if n:
        for sid in np.unique(pts.stroke):
            m = pts.stroke == sid
            ax.plot(pts.pos[m, 0], pts.pos[m, 1], pts.pos[m, 2], "-", lw=0.6, color="#555555")
        ax.scatter(pts.pos[:, 0], pts.pos[:, 1], pts.pos[:, 2], s=5, c=colors, linewidths=0)
        ax.quiver(pts.pos[arrow_idx, 0], pts.pos[arrow_idx, 1], pts.pos[arrow_idx, 2],
                  pts.normal[arrow_idx, 0], pts.normal[arrow_idx, 1], pts.normal[arrow_idx, 2],
                  length=spacing, color="#2ca02c", linewidth=0.5)
    ax.scatter(vsub[::4, 0], vsub[::4, 1], vsub[::4, 2], s=1, c="#d0d0d0", alpha=0.3, linewidths=0)
    ext = mesh.extents.max()
    c = mesh.bounds.mean(axis=0)
    ax.set_xlim(c[0] - ext / 2, c[0] + ext / 2)
    ax.set_ylim(c[1] - ext / 2, c[1] + ext / 2)
    ax.set_zlim(c[2] - ext / 2, c[2] + ext / 2)
    ax.set_title("3D (colour = order, green = normal)")
    s = result.stats
    fig.suptitle(title or f"{result.params['source']} | {result.params['pattern']} | {n} points, "
                 f"{s.get('n_strokes', 0)} strokes, path {s.get('path_length', 0):.2f} m | "
                 f"red = tool_dir, green = norm_dir, star = start", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path


def _draw_path_2d(ax, pts, a, b):
    p = pts.pos
    same = pts.stroke[1:] == pts.stroke[:-1]
    for i in np.where(same)[0]:
        pass
    # vectorised line segments: within-stroke solid, jumps dashed
    from matplotlib.collections import LineCollection
    segs = np.stack([p[:-1][:, [a, b]], p[1:][:, [a, b]]], axis=1)
    ax.add_collection(LineCollection(segs[same], colors="#555555", linewidths=0.5, zorder=2))
    if (~same).any():
        ax.add_collection(LineCollection(segs[~same], colors="#e41a1c", linewidths=0.6, linestyles="dashed", zorder=2))


def plotly_html(result, path: str, arrow_scale: float = 1.0, slider: bool = True, slider_steps: int = 40,
                labels_every: int | None = None, embed_js: bool = True, title: str | None = None,
                open_browser: bool = False) -> str:
    """Interactive viewer (standalone HTML).

    Layers (toggle in the legend): mesh per region (unreachable parts darker), points
    coloured by order, path (solid within a stroke, dashed transitions), tool_dir
    cones (red, into the surface), norm_dir cones (green, out of the surface) and
    order labels.  Hover a point for index / region / level / stroke.  The slider
    replays the order.
    """
    import plotly.graph_objects as go

    from .sample import SubMesh

    pts = result.points
    mesh = result.mesh
    seg = result.seg
    n = len(pts)
    spacing = result.params["spacing"]
    selected = seg.select(result.params["include"], result.params["exclude"], result.params["reachable_only"])
    fig = go.Figure()
    r5 = lambda a: np.round(a, 5)

    # --- mesh, one trace per region (and unreachable part) ---------------------------
    for rid, name in enumerate(REGION_NAMES):
        for reach in (True, False):
            m = (seg.labels == rid) & (seg.reachable == reach)
            if not m.any():
                continue
            sub = SubMesh(mesh, np.where(m)[0]).mesh
            V, F = r5(sub.vertices), sub.faces
            color = REGION_COLORS[name] if reach else UNREACHABLE_COLOR
            is_sel = bool(selected[m].any())
            fig.add_trace(go.Mesh3d(
                x=V[:, 0], y=V[:, 1], z=V[:, 2], i=F[:, 0], j=F[:, 1], k=F[:, 2],
                color=color, opacity=0.5 if is_sel else 0.25, flatshading=True,
                lighting=dict(ambient=0.55, diffuse=0.7, specular=0.05, roughness=0.9),
                name=f"{name}{'' if reach else ' (unreachable)'}" + (" ✓" if is_sel else ""),
                hoverinfo="name", showlegend=True, legendgroup="mesh", legendgrouptitle_text="mesh regions"))

    if n == 0:
        fig.update_layout(title="no points generated")
        fig.write_html(path, include_plotlyjs=True if embed_js else "cdn")
        return path

    P = r5(pts.pos)
    names = pts.region_names
    hover = [f"#{i} | {names[i]} L{pts.level[i]} | stroke {pts.stroke[i]}" for i in range(n)]
    order = np.arange(n)

    # --- path (solid within strokes, dashed between) -------------------------------------
    same = pts.stroke[1:] == pts.stroke[:-1]
    px, py, pz = [], [], []
    for i in range(n - 1):
        if same[i]:
            px += [P[i, 0], P[i + 1, 0], None]
            py += [P[i, 1], P[i + 1, 1], None]
            pz += [P[i, 2], P[i + 1, 2], None]
    fig.add_trace(go.Scatter3d(x=px, y=py, z=pz, mode="lines", line=dict(color="#444444", width=3),
                               name="path (strokes)", hoverinfo="skip", legendgroup="path"))
    jx, jy, jz = [], [], []
    for i in np.where(~same)[0]:
        jx += [P[i, 0], P[i + 1, 0], None]
        jy += [P[i, 1], P[i + 1, 1], None]
        jz += [P[i, 2], P[i + 1, 2], None]
    fig.add_trace(go.Scatter3d(x=jx, y=jy, z=jz, mode="lines", line=dict(color="#e41a1c", width=3, dash="dash"),
                               name="transitions (lift)", hoverinfo="skip", legendgroup="path"))

    # --- points coloured by order ----------------------------------------------------------
    idx_points = len(fig.data)
    fig.add_trace(go.Scatter3d(
        x=P[:, 0], y=P[:, 1], z=P[:, 2], mode="markers",
        marker=dict(size=3.5, color=order, colorscale="Viridis", colorbar=dict(title="order", thickness=12, len=0.5),
                    showscale=True),
        text=hover, hoverinfo="text", name=f"points ({n})", legendgroup="points"))
    fig.add_trace(go.Scatter3d(x=P[:1, 0], y=P[:1, 1], z=P[:1, 2], mode="markers",
                               marker=dict(size=9, color="#000000", symbol="diamond"),
                               name="start", hoverinfo="name", legendgroup="points"))

    # --- direction cones ---------------------------------------------------------------------
    L = spacing * 0.8 * arrow_scale
    cone_kw = dict(sizemode="raw", sizeref=1.0, anchor="tail", showscale=False, hoverinfo="skip")
    try:
        fig.add_trace(go.Cone(x=P[:, 0], y=P[:, 1], z=P[:, 2], u=pts.tool_dir[:, 0] * L, v=pts.tool_dir[:, 1] * L,
                              w=pts.tool_dir[:, 2] * L, colorscale=[[0, "#d62728"], [1, "#d62728"]],
                              name="tool_dir (into surface)", legendgroup="dirs", **cone_kw))
        fig.add_trace(go.Cone(x=P[:, 0], y=P[:, 1], z=P[:, 2], u=pts.normal[:, 0] * L, v=pts.normal[:, 1] * L,
                              w=pts.normal[:, 2] * L, colorscale=[[0, "#2ca02c"], [1, "#2ca02c"]],
                              name="norm_dir (out of surface)", legendgroup="dirs", **cone_kw))
    except ValueError:  # older plotly without sizemode='raw'
        cone_kw.update(sizemode="absolute", sizeref=L)
        fig.add_trace(go.Cone(x=P[:, 0], y=P[:, 1], z=P[:, 2], u=pts.tool_dir[:, 0], v=pts.tool_dir[:, 1],
                              w=pts.tool_dir[:, 2], colorscale=[[0, "#d62728"], [1, "#d62728"]],
                              name="tool_dir (into surface)", legendgroup="dirs", **cone_kw))
        fig.add_trace(go.Cone(x=P[:, 0], y=P[:, 1], z=P[:, 2], u=pts.normal[:, 0], v=pts.normal[:, 1],
                              w=pts.normal[:, 2], colorscale=[[0, "#2ca02c"], [1, "#2ca02c"]],
                              name="norm_dir (out of surface)", legendgroup="dirs", **cone_kw))

    # --- order labels ------------------------------------------------------------------------
    k = labels_every or max(1, int(round(n / 40)))
    li = np.arange(0, n, k)
    fig.add_trace(go.Scatter3d(x=P[li, 0], y=P[li, 1], z=P[li, 2], mode="text", text=[str(i) for i in li],
                               textfont=dict(size=9, color="#222222"), name=f"order labels (every {k})",
                               hoverinfo="skip", visible="legendonly" if n > 4000 else True, legendgroup="points"))

    # --- replay slider -------------------------------------------------------------------------
    if slider and n > 1:
        steps = np.unique(np.linspace(1, n, min(slider_steps, n)).astype(int))
        frames, slider_steps_cfg = [], []
        for kk in steps:
            fx, fy, fz = [], [], []
            for i in range(kk - 1):
                if same[i]:
                    fx += [P[i, 0], P[i + 1, 0], None]
                    fy += [P[i, 1], P[i + 1, 1], None]
                    fz += [P[i, 2], P[i + 1, 2], None]
            frames.append(go.Frame(
                name=str(kk),
                data=[go.Scatter3d(x=fx, y=fy, z=fz, mode="lines", line=dict(color="#444444", width=3)),
                      go.Scatter3d(x=P[:kk, 0], y=P[:kk, 1], z=P[:kk, 2], mode="markers",
                                   marker=dict(size=3.5, color=order[:kk], colorscale="Viridis",
                                               cmin=0, cmax=n - 1, showscale=True,
                                               colorbar=dict(title="order", thickness=12, len=0.5)),
                                   text=hover[:kk], hoverinfo="text")],
                traces=[idx_points - 2, idx_points]))
            slider_steps_cfg.append(dict(method="animate", label=str(kk),
                                         args=[[str(kk)], dict(mode="immediate", frame=dict(duration=0, redraw=True),
                                                               transition=dict(duration=0))]))
        fig.frames = frames
        fig.update_layout(sliders=[dict(active=len(steps) - 1, currentvalue=dict(prefix="points shown: "),
                                        pad=dict(t=30), steps=slider_steps_cfg)],
                          updatemenus=[dict(type="buttons", showactive=False, x=0.0, y=0.0, xanchor="left",
                                            buttons=[dict(label="▶ replay", method="animate",
                                                          args=[None, dict(frame=dict(duration=120, redraw=True),
                                                                           fromcurrent=False, mode="immediate",
                                                                           transition=dict(duration=0))])])])

    s = result.stats
    fig.update_layout(
        title=title or (f"{result.params['source']} | pattern={result.params['pattern']} order={result.params['order']} | "
                        f"{n} points, {s['n_strokes']} strokes, path {s['path_length']:.2f} m | "
                        f"spacing {spacing} m, row {result.params['row_spacing']} m"),
        scene=dict(aspectmode="data", xaxis_title="x [m]", yaxis_title="y [m]", zaxis_title="z [m]"),
        legend=dict(itemsizing="constant", groupclick="toggleitem"),
        margin=dict(l=0, r=0, t=50, b=0), height=850)
    fig.write_html(path, include_plotlyjs=True if embed_js else "cdn", auto_open=open_browser)
    return path
