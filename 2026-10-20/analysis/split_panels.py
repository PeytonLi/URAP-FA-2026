"""Save every chart as its own PNG; graphs are never combined into one image."""
from pathlib import Path


def save_each(panels, folder, stem, size, headline="", dpi=150):
    """`panels` is a list of (name, draw) pairs; `draw(ax)` fills one axis."""
    import matplotlib.pyplot as plt
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    for name, draw in panels:
        fig, ax = plt.subplots(figsize=size)
        draw(ax)
        if headline:
            ax.set_title(headline + "\n" + ax.get_title(), fontsize=11)
        fig.tight_layout()
        fig.savefig(folder / f"{stem}_{name}.png", dpi=dpi, bbox_inches="tight")
        plt.close(fig)
