from typing import Dict, List

import plotext as plt
from rich.ansi import AnsiDecoder
from rich.console import Console, Group
from rich.jupyter import JupyterMixin
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.text import Text

# using plotext and piping into rich see here for more deets" https://github.com/piccolomo/plotext/blob/4d19108b93e34a60ba789681756450ae126a76ed/readme/environments.md


def make_plot(
    width: int,
    height: int,
    series: List[Dict[str, list | str]],
    title: str = "",
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
):
    plt.clf()
    for s in series:
        x, y = s["x"], s["y"]
        kwargs = {}
        for opt in ("label", "color", "marker"):
            if opt in s:
                kwargs[opt] = s[opt]
        plt.scatter(x, y, **kwargs)

    # Set axis‐limits if requested
    if xlim is not None:
        plt.xlim(xlim[0], xlim[1])
    if ylim is not None:
        plt.ylim(ylim[0], ylim[1])

    plt.plotsize(width, height)
    plt.title(title)
    plt.theme("dark")
    return plt.build()


def make_hist_plot(width, height, data, bins, title=""):
    raise NotImplementedError("Histogram plotting not implemented yet")
    plt.clf()
    plt.hist(data, bins=bins)
    plt.plotsize(width, height)
    plt.title(title)
    plt.theme("dark")
    return plt.build()


class plotextMixin(JupyterMixin):
    def __init__(
        self,
        series: List[Dict[str, list | str]],
        xlim: tuple[float, float] | None = None,
        ylim: tuple[float, float] | None = None,
        phase=0,
        title="",
    ):
        if not isinstance(series, list):
            series = [series]
        if not all(isinstance(s, dict) for s in series):
            raise ValueError("Series must be a list of dictionaries.")
        if not all("x" in s and "y" in s for s in series):
            raise ValueError("Each series must contain 'x' and 'y' keys.")
        if not all(len(s["x"]) == len(s["y"]) for s in series):
            raise ValueError("Each series must have the same length for 'x' and 'y'.")
        self.decoder = AnsiDecoder()
        self.phase = phase
        self.title = title
        self.series = series
        self.xlim = xlim
        self.ylim = ylim

    def __rich_console__(self, console, options):
        self.width = options.max_width or console.width
        self.height = options.height or console.height
        canvas = make_plot(
            self.width, self.height, self.series, self.title, self.xlim, self.ylim
        )
        self.rich_canvas = Group(*self.decoder.decode(canvas))
        yield self.rich_canvas


def make_layout():
    layout = Layout(name="root")
    layout.split(
        Layout(name="header", size=1),
        Layout(name="main", ratio=1),
    )
    return layout


if __name__ == "__main__":
    x = list(range(1, 1000))
    y = list(range(1, 1000))
    layout = make_layout()

    header = layout["header"]
    title = "Scatter Plot Example"
    header.update(Text(title, justify="left"))

    static = layout["main"]
    mixin = Panel(plotextMixin(title="Static Plot", x=x, y=y))
    static.update(mixin)

    # Create a console and print the layout
    console = Console()
    console.print(layout)
