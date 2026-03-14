from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from matplotlib import pyplot as plt
from matplotlib.ticker import ScalarFormatter, MaxNLocator, EngFormatter


FIVE_PANEL_FIG_SIZE: tuple[float, float] = (18.0, 12.0)

# Logical geometry contract
LOGICAL_GRID_ROWS = 2
LOGICAL_GRID_COLS = 23
PANEL_GRID_WIDTH = 7
PANEL_GRID_HEIGHT = 1  # one half-row in a 2-row grid
SINGLE_PANEL_SCALE = 1.45

SINGLE_PANEL_FIG_SIZE: tuple[float, float] = (
    FIVE_PANEL_FIG_SIZE[0] * PANEL_GRID_WIDTH / LOGICAL_GRID_COLS * SINGLE_PANEL_SCALE,
    FIVE_PANEL_FIG_SIZE[1] * PANEL_GRID_HEIGHT / LOGICAL_GRID_ROWS * SINGLE_PANEL_SCALE,
)

SUPTITLE_Y = 0.985
SUBTITLE_Y = 0.945
TIGHT_LAYOUT_RECT = [0.03, 0.03, 0.97, 0.88]

HYPERPARAM_AXIS_ORDER: dict[str, str] = {
    "rho": "desc",
    "k": "asc",
    "lambda": "desc",
}

@dataclass(frozen=True)
class PanelLayoutContract:
    five_panel_fig_size: tuple[float, float] = FIVE_PANEL_FIG_SIZE
    single_panel_fig_size: tuple[float, float] = SINGLE_PANEL_FIG_SIZE
    logical_grid: tuple[int, int] = (LOGICAL_GRID_ROWS, LOGICAL_GRID_COLS)
    panel_grid_span: tuple[int, int] = (PANEL_GRID_HEIGHT, PANEL_GRID_WIDTH)
    single_panel_scale: float = SINGLE_PANEL_SCALE

    @property
    def expected_single_size(self) -> tuple[float, float]:
        return (
            self.five_panel_fig_size[0] * self.panel_grid_span[1] / self.logical_grid[1] * self.single_panel_scale,
            self.five_panel_fig_size[1] * self.panel_grid_span[0] / self.logical_grid[0] * self.single_panel_scale,
        )


LAYOUT = PanelLayoutContract()

assert math.isclose(LAYOUT.single_panel_fig_size[0], 18.0 * 7.0 / 23.0 * SINGLE_PANEL_SCALE, rel_tol=0.0, abs_tol=1e-12)
assert math.isclose(LAYOUT.single_panel_fig_size[1], 12.0 / 2.0 * SINGLE_PANEL_SCALE, rel_tol=0.0, abs_tol=1e-12)

def assert_layout_contract() -> None:
    expected = LAYOUT.expected_single_size
    actual = LAYOUT.single_panel_fig_size
    assert math.isclose(actual[0], expected[0], rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(actual[1], expected[1], rel_tol=0.0, abs_tol=1e-12)


def _value_sort_key(value: object) -> tuple[int, object]:
    if isinstance(value, (int, float)):
        return (0, float(value))
    try:
        return (0, float(value))
    except Exception:
        return (1, str(value))


def order_is_descending(param_name: str | None, param_specs: dict[str, dict[str, object]] | None = None) -> bool:
    if not isinstance(param_name, str):
        return False
    normalized = param_name.strip().lower()
    if normalized in HYPERPARAM_AXIS_ORDER:
        return HYPERPARAM_AXIS_ORDER[normalized] == "desc"
    if param_specs is None:
        return False
    spec = param_specs.get(param_name, {})
    return spec.get("print_order") == "high_to_low"


def param_directed_sort_key(
    param_name: str | None,
    value: object,
    param_specs: dict[str, dict[str, object]] | None = None,
) -> tuple[int, object]:
    base_key = _value_sort_key(value)
    if base_key[0] != 0:
        return base_key
    numeric_value = float(base_key[1])
    if order_is_descending(param_name, param_specs):
        return (0, -numeric_value)
    return (0, numeric_value)


def create_panel_axes(
    num_panels: int,
    *,
    sharex: bool,
    sharey: bool,
    row_size: tuple[float, float] = (8.0, 7.0),
    grid_size: tuple[float, float] = (8.0, 6.0),
) -> tuple[Any, list[Any]]:
    """
    Shared geometry contract:
    - 5 panels use the canonical 2 x 23 logical grid
    - 1 panel is exactly one 7-column cell by one half-row of that 5-panel grid
    - no caller may override single-panel sizing
    """
    assert_layout_contract()

    if num_panels == 5:
        fig = plt.figure(figsize=LAYOUT.five_panel_fig_size)
        grid = fig.add_gridspec(LOGICAL_GRID_ROWS, LOGICAL_GRID_COLS)

        first = fig.add_subplot(grid[0, 4:11])
        second = fig.add_subplot(
            grid[0, 12:19],
            sharex=first if sharex else None,
            sharey=first if sharey else None,
        )
        third = fig.add_subplot(
            grid[1, 0:7],
            sharex=first if sharex else None,
            sharey=first if sharey else None,
        )
        fourth = fig.add_subplot(
            grid[1, 8:15],
            sharex=first if sharex else None,
            sharey=first if sharey else None,
        )
        fifth = fig.add_subplot(
            grid[1, 16:23],
            sharex=first if sharex else None,
            sharey=first if sharey else None,
        )
        return fig, [first, second, third, fourth, fifth]

    if num_panels == 1:
        fig, ax = plt.subplots(
            1,
            1,
            figsize=LAYOUT.single_panel_fig_size,
            squeeze=False,
            sharex=sharex,
            sharey=sharey,
        )
        return fig, [ax[0, 0]]

    if num_panels <= 3:
        fig, axes = plt.subplots(
            1,
            num_panels,
            figsize=(row_size[0] * num_panels, row_size[1]),
            squeeze=False,
            sharex=sharex,
            sharey=sharey,
        )
        return fig, list(axes[0])

    ncols = 3
    nrows = math.ceil(num_panels / ncols)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(grid_size[0] * ncols, grid_size[1] * nrows),
        squeeze=False,
        sharex=sharex,
        sharey=sharey,
    )
    flat_axes = list(axes.flat)
    for extra_ax in flat_axes[num_panels:]:
        extra_ax.set_visible(False)
    return fig, flat_axes[:num_panels]

def style_x_axis(
    ax: plt.Axes,
    *,
    rotate_degrees: float = 0.0,
    use_engineering: bool = False,
    use_scientific: bool = False,
    max_ticks: int = 5,
) -> None:
    ax.xaxis.set_major_locator(MaxNLocator(nbins=max_ticks))

    if use_engineering:
        ax.xaxis.set_major_formatter(EngFormatter())
    elif use_scientific:
        formatter = ScalarFormatter(useMathText=True)
        formatter.set_scientific(True)
        formatter.set_powerlimits((-3, 3))
        ax.xaxis.set_major_formatter(formatter)

    if rotate_degrees:
        for label in ax.get_xticklabels():
            label.set_rotation(rotate_degrees)
            label.set_ha("right")
