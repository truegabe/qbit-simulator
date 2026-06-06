"""Visualization helpers — probability bars, Bloch sphere, ASCII circuit."""

from __future__ import annotations

import numpy as np

# Use non-interactive backend by default so headless / script runs work.
import matplotlib
matplotlib.use("Agg", force=False)
import matplotlib.pyplot as plt  # noqa: E402
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401, E402

from .qubit import Qubit
from .circuit import QuantumCircuit


# ---------- probability bars ----------

def plot_probabilities(
    qc: QuantumCircuit | np.ndarray,
    title: str = "Computational basis probabilities",
    threshold: float = 1e-4,
    ax=None,
):
    state = qc.state if isinstance(qc, QuantumCircuit) else np.asarray(qc)
    n = int(np.log2(len(state)))
    probs = np.abs(state) ** 2

    mask = probs > threshold
    indices = np.arange(len(probs))[mask]
    labels = [format(i, f"0{n}b") for i in indices]
    values = probs[mask]

    if ax is None:
        fig, ax = plt.subplots(figsize=(max(6, 0.4 * len(indices) + 2), 4))
    else:
        fig = ax.figure

    bars = ax.bar(range(len(indices)), values, color="#4C72B0")
    ax.set_xticks(range(len(indices)))
    ax.set_xticklabels(labels, rotation=45 if n > 3 else 0, ha="right")
    ax.set_ylabel("Probability")
    ax.set_ylim(0, 1.05)
    ax.set_title(title)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01,
                f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    return fig


def plot_counts(counts: dict[str, int], title: str = "Measurement counts", ax=None):
    keys = sorted(counts.keys())
    vals = [counts[k] for k in keys]

    if ax is None:
        fig, ax = plt.subplots(figsize=(max(6, 0.4 * len(keys) + 2), 4))
    else:
        fig = ax.figure

    ax.bar(range(len(keys)), vals, color="#55A868")
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels(keys, rotation=45 if len(keys[0]) > 3 else 0, ha="right")
    ax.set_ylabel("Counts")
    ax.set_title(title)
    fig.tight_layout()
    return fig


# ---------- Bloch sphere (single qubit) ----------

def bloch_coords(q: Qubit | np.ndarray) -> tuple[float, float, float]:
    state = q.state if isinstance(q, Qubit) else np.asarray(q)
    a, b = state[0], state[1]
    x = 2 * (a.conjugate() * b).real
    y = 2 * (a.conjugate() * b).imag
    z = (a.conjugate() * a - b.conjugate() * b).real
    return float(x), float(y), float(z)


def plot_bloch(q: Qubit | np.ndarray, title: str = "Bloch sphere"):
    x, y, z = bloch_coords(q)

    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection="3d")

    # Unit sphere wireframe
    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, np.pi, 20)
    xs = np.outer(np.cos(u), np.sin(v))
    ys = np.outer(np.sin(u), np.sin(v))
    zs = np.outer(np.ones_like(u), np.cos(v))
    ax.plot_wireframe(xs, ys, zs, color="lightgray", linewidth=0.4)

    # Axes
    ax.plot([-1, 1], [0, 0], [0, 0], color="black", linewidth=0.5)
    ax.plot([0, 0], [-1, 1], [0, 0], color="black", linewidth=0.5)
    ax.plot([0, 0], [0, 0], [-1, 1], color="black", linewidth=0.5)
    ax.text(0, 0, 1.15, r"|0$\rangle$", ha="center")
    ax.text(0, 0, -1.25, r"|1$\rangle$", ha="center")
    ax.text(1.15, 0, 0, r"|+$\rangle$", ha="center")
    ax.text(-1.25, 0, 0, r"|-$\rangle$", ha="center")

    # State vector
    ax.quiver(0, 0, 0, x, y, z, color="#C44E52", linewidth=2, arrow_length_ratio=0.1)
    ax.scatter([x], [y], [z], color="#C44E52", s=40)

    ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_zlim(-1, 1)
    ax.set_box_aspect((1, 1, 1))
    ax.set_title(f"{title}\n(x,y,z) = ({x:.2f}, {y:.2f}, {z:.2f})")
    ax.set_axis_off()
    fig.tight_layout()
    return fig


# ---------- ASCII circuit diagram ----------

def animate_grover(
    n_qubits: int,
    marked: int,
    out_path: str,
    fps: int = 4,
) -> None:
    """Generate an animated GIF showing Grover amplitude amplification.

    Each frame is the probability distribution after one Grover iteration.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    from .circuit import QuantumCircuit
    from .algorithms.grover import optimal_iterations

    qc = QuantumCircuit(n_qubits)
    for q in range(n_qubits):
        qc.h(q)

    n_iters = optimal_iterations(n_qubits)
    dim = 2**n_qubits
    snapshots = [np.abs(qc.state) ** 2]
    for _ in range(n_iters):
        qc.state[marked] = -qc.state[marked]
        mean = qc.state.mean()
        qc.state = 2 * mean - qc.state
        snapshots.append(np.abs(qc.state) ** 2)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    indices = np.arange(dim)
    bars = ax.bar(indices, snapshots[0], color="#4C72B0")
    bars[marked].set_color("#C44E52")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Basis state index")
    ax.set_ylabel("Probability")
    title = ax.set_title(f"Grover N={n_qubits}, marked={marked:b} — iter 0/{n_iters}")
    if dim <= 32:
        ax.set_xticks(indices)
        ax.set_xticklabels([format(i, f"0{n_qubits}b") for i in indices],
                           rotation=45, ha="right", fontsize=7)

    def update(frame):
        for bar, h in zip(bars, snapshots[frame]):
            bar.set_height(h)
        title.set_text(f"Grover N={n_qubits}, marked={marked:b} — iter {frame}/{n_iters}")
        return list(bars) + [title]

    anim = FuncAnimation(fig, update, frames=len(snapshots), interval=1000 // fps,
                         blit=False, repeat=True)
    writer = PillowWriter(fps=fps)
    anim.save(out_path, writer=writer)
    plt.close(fig)


_FRIENDLY_TAGS = {
    "GroverStep": "[GR]",
    "ReverseQubits": "[REV]",
    "MeasureAll": "[M*]",
    "Measure": "[M]",
}


def _parse_op(op: str) -> tuple[str, list[str]]:
    if "(" not in op:
        return op, []
    name = op.split("(")[0]
    args = op[op.find("(") + 1 : op.rfind(")")]
    parts = [p.strip() for p in args.split(",") if p.strip()]
    return name, parts


def circuit_ascii(qc: QuantumCircuit) -> str:
    """Render the circuit history as an ASCII diagram. One column per op."""
    n = qc.n
    columns: list[list[str]] = []  # columns[i] is a list of length n

    for op in qc.history:
        name, parts = _parse_op(op)
        col = ["-"] * n

        if name in {"H", "X", "Y", "Z", "S", "T"}:
            col[int(parts[0])] = name
        elif name in {"Rx", "Ry", "Rz", "P"}:
            theta = parts[0] if len(parts) >= 2 else ""
            q = int(parts[-1])
            col[q] = f"{name}({theta})"
        elif name == "CNOT":
            c, t = int(parts[0]), int(parts[1])
            col[c] = "*"; col[t] = "X"
            for i in range(min(c, t) + 1, max(c, t)):
                col[i] = "|"
        elif name in {"CP", "CZ"}:
            c, t = int(parts[-2]), int(parts[-1])
            col[c] = "*"
            col[t] = "Z" if name == "CZ" else f"P({parts[0]})"
            for i in range(min(c, t) + 1, max(c, t)):
                col[i] = "|"
        elif name == "SWAP":
            a, b = int(parts[0]), int(parts[1])
            col[a] = "x"; col[b] = "x"
            for i in range(min(a, b) + 1, max(a, b)):
                col[i] = "|"
        elif name.startswith("Measure") and "->" in op:
            outcome = op.split("->")[-1]
            if name == "Measure":  # single qubit
                q = int(parts[0].lstrip("q"))
                col[q] = f"[M={outcome}]"
            else:  # MeasureAll
                for i in range(n):
                    col[i] = f"[M={outcome}]"
        else:
            tag = _FRIENDLY_TAGS.get(name, f"[{name[:4]}]")
            for i in range(n):
                col[i] = tag

        columns.append(col)

    if not columns:
        return "\n".join(f"q{i}: -" for i in range(n))

    # Per-column width.
    widths = [max(len(cell) for cell in col) for col in columns]
    lines = []
    for i in range(n):
        cells = [columns[c][i].center(widths[c], "-") for c in range(len(columns))]
        lines.append(f"q{i}: " + "-".join(cells))
    return "\n".join(lines)
