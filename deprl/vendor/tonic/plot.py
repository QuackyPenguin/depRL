import argparse
import itertools
import os
import pathlib
import time

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import gridspec, lines

from deprl.vendor.tonic import logger


def smooth(vals, window):
    """Smooths values using a sliding window."""
    if window > 1:
        if window > len(vals):
            window = len(vals)
        y = np.ones(window)
        x = vals
        z = np.ones(len(vals))
        mode = "same"
        vals = np.convolve(x, y, mode) / np.convolve(z, y, mode)
    return vals


def stats(xs, means, stds):
    """Extract statistics over the common range of x values."""
    lengths = [len(x) for x in xs]
    min_length = min(lengths)
    xs = [x[:min_length] for x in xs]
    assert len(xs) == 1 or np.all(xs[1:] == xs[0]), xs
    x = xs[0]

    means = np.array([mean[:min_length] for mean in means])
    mean = means.mean(axis=0)
    min_mean = means.min(axis=0)
    max_mean = means.max(axis=0)
    if stds is not None:
        stds = np.array([std[:min_length] for std in stds])
        variances = stds**2
        var_within = variances.mean(axis=0)
        var_between = ((means - mean) ** 2).mean(axis=0)
        std = np.sqrt(var_within + var_between)
    else:
        std = None

    return x, mean, min_mean, max_mean, std


def flip(items, columns):
    """Flips values to fill horizontally the legend instead of vertically."""
    return itertools.chain(*[items[i::columns] for i in range(columns)])


def get_data(
    paths, baselines, baselines_source, x_axis, y_axes, x_min, x_max, window
):
    data = {}

    log_paths = []
    for path in paths:
        if os.path.isdir(path):
            log_paths.extend(pathlib.Path(path).rglob("log.*"))
        elif path[-7:-3] == "log.":
            log_paths.append(path)

    for path in log_paths:
        sub_path, file = os.path.split(path)
        dfs = {}

        if file == "log.csv":
            env, agent, seed = sub_path.split(os.sep)[-3:]
            df_seed = pd.read_csv(path, sep=",", engine="python")
            x = df_seed[x_axis].values
            if not np.all(np.diff(x) > 0):
                logger.warning(f"Skipping unsorted {env} {agent} {seed}")
                continue
            if x_min and x[-1] < x_min:
                logger.warning(
                    f"Skipping {env} {agent} {seed} ({x[-1]} steps)"
                )
                continue
            dfs[seed] = df_seed

        elif file == "log.pkl":
            env, agent = sub_path.split(os.sep)[-2:]
            df = pd.read_pickle(path, compression="zip")
            for seed, df_seed in df.groupby("seed"):
                x = df_seed[x_axis].values
                if not np.all(np.diff(x) > 0):
                    logger.warning(f"Skipping unsorted {env} {agent} {seed}")
                    continue
                if x_min and x[-1] < x_min:
                    logger.warning(
                        f"Skipping {env} {agent} {seed} ({x[-1]} steps)"
                    )
                    continue
                dfs[seed] = df_seed

        for seed, df in dfs.items():
            if env not in data:
                data[env] = {}
            if agent not in data[env]:
                data[env][agent] = {}
            assert seed not in data[env][agent]

            x = df[x_axis].values
            agent_data = {}
            for y_axis in y_axes:
                if y_axis in df:
                    mean = df[y_axis].values
                    if y_axis[-5:] == "/mean" and y_axis[:-5] + "/std" in df:
                        std = df[y_axis[:-5] + "/std"].values
                    else:
                        std = None
                elif y_axis + "/mean" in df:
                    mean = df[y_axis + "/mean"].values
                    if y_axis + "/std" in df:
                        std = df[y_axis + "/std"].values
                    else:
                        std = None
                else:
                    continue

                if x_max:
                    max_index = np.argmax(x > x_max) or len(x)
                else:
                    max_index = len(x)
                x = x[:max_index]
                mean = smooth(mean, window)
                mean = mean[:max_index]
                if std is not None:
                    std = smooth(std, window)
                    std = std[:max_index]

                agent_data[y_axis] = (x, mean, std)

            data[env][agent][seed] = agent_data

    for env, env_data in data.items():
        for agent, agent_data in env_data.items():
            xs, means, stds = {}, {}, {}
            for seed, seed_data in agent_data.items():
                for y_axis in y_axes:
                    x, mean, std = seed_data.get(y_axis, (None, None, None))
                    if x is None:
                        continue
                    if y_axis not in xs:
                        xs[y_axis] = []
                        means[y_axis] = []
                        stds[y_axis] = []
                    xs[y_axis].append(x)
                    means[y_axis].append(mean)
                    stds[y_axis].append(std)

            for y_axis in y_axes:
                if y_axis not in xs:
                    continue
                if stds[y_axis][0] is None:
                    stds[y_axis] = None
                env_data[agent][y_axis] = dict(
                    seeds=(xs[y_axis], means[y_axis]),
                    stats=stats(xs[y_axis], means[y_axis], stds[y_axis])
                )

    return data


def plot(
    paths,
    x_axis,
    y_axes,
    x_label,
    y_label,
    window,
    interval,
    show_seeds,
    columns,
    x_min,
    x_max,
    y_min,
    y_max,
    baselines,
    baselines_source,
    name,
    save_formats,
    cmap,
    legend_columns,
    legend_marker_size,
    dpi,
    title,
    fig,
):
    logger.log("Loading data...")
    data = get_data(
        paths,
        baselines,
        baselines_source,
        x_axis,
        y_axes,
        x_min,
        x_max,
        window,
    )

    envs = sorted(data.keys(), key=str.casefold)
    num_envs = len(envs)
    if num_envs == 0:
        logger.error("No logs found.")
        return

    agents = set()
    for env in data:
        for agent in data[env]:
            agents.add(agent)
    agents = sorted(agents, key=str.casefold)
    num_agents = len(agents)

    if not cmap:
        if num_agents <= 10:
            cmap = "tab10"
        elif num_agents <= 20:
            cmap = "tab20"
        else:
            cmap = "rainbow"
    cmap = plt.get_cmap(cmap)

    if isinstance(cmap, mpl.colors.ListedColormap):
        colors = cmap(range(num_agents))
    else:
        colors = list(cmap(np.linspace(0, 1, num_agents)))

    agent_colors = {a: c for a, c in zip(agents, colors)}
    if columns is None:
        columns = int(np.ceil(np.sqrt(num_envs)))
    else:
        columns = min(columns, num_envs)
    rows = int(np.ceil(num_envs / columns))
    if fig is None:
        plt.ion()
        fig = plt.figure(figsize=(columns * 6, rows * 5))
    else:
        fig.clear()
    grid = gridspec.GridSpec(
        rows + 1,
        1 + columns,
        height_ratios=[1] * rows + [0.1],
        width_ratios=[0] + [1] * columns,
    )
    axes = []
    for i in range(num_envs):
        ax = fig.add_subplot(grid[i // columns, 1 + i % columns])
        axes.append(ax)

    logger.log("Plotting episode scores...")
    for env, ax in zip(envs, axes):
        if interval in ["std", "bounds"]:
            for agent in sorted(data[env], key=str.casefold):
                color = agent_colors[agent]
                for y_axis in y_axes:
                    if y_axis not in data[env][agent]:
                        continue
                    x, mean, min_mean, max_mean, std = data[env][agent][y_axis]["stats"]
                    if interval == "std":
                        if std is None:
                            logger.error("No std found in the data.")
                        else:
                            ax.fill_between(
                                x,
                                mean - std,
                                mean + std,
                                color=color,
                                alpha=0.1,
                                lw=0,
                            )
                    elif interval == "bounds":
                        ax.fill_between(
                            x, min_mean, max_mean, color=color, alpha=0.1, lw=0
                        )

        for agent in sorted(data[env], key=str.casefold):
            color = agent_colors[agent]
            for y_axis in y_axes:
                if y_axis not in data[env][agent]:
                    continue

                xs, means = data[env][agent][y_axis]["seeds"]
                if show_seeds and len(xs) > 1:
                    for x, mean in zip(xs, means):
                        ax.plot(x, mean, c=color, lw=1, alpha=0.5)

                x, mean = data[env][agent][y_axis]["stats"][:2]
                ax.plot(x, mean, c=color, lw=2, alpha=1)

        ax.set_ylim(ymin=y_min, ymax=y_max)
        ax.locator_params(axis="x", nbins=6)
        ax.locator_params(axis="y", tight=True, nbins=6)
        ax.get_yaxis().set_major_formatter(
            mpl.ticker.FuncFormatter(lambda x, p: f"{x:,g}")
        )
        low, high = ax.get_xlim()
        if max(abs(low), abs(high)) >= 1e3:
            ax.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))
        ax.xaxis.grid(linewidth=0.5, alpha=0.5)
        ax.yaxis.grid(linewidth=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="both", which="both", length=0)
        if x_label is None:
            x_label = "Steps" if x_axis == "train/steps" else x_axis
        if y_label is None:
            y_label = "Score" if y_axis == "test/episode_score" else y_axis
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        if title:
            ax.set_title(title)
        else:
            ax.set_title(env)

    legend_ax = fig.add_subplot(grid[-1:, :])
    legend_ax.set_axis_off()
    handles = []
    for color in colors:
        marker = lines.Line2D(
            range(1),
            range(1),
            marker="o",
            markerfacecolor=color,
            markersize=legend_marker_size,
            linewidth=0,
            markeredgewidth=0,
        )
        handles.append(marker)

    if legend_columns is None:
        legend_columns = range(num_agents, 0, -1)
    else:
        legend_columns = [legend_columns]
    for ncol in legend_columns:
        legend = legend_ax.legend(
            flip(handles, ncol),
            flip(agents, ncol),
            loc="center",
            mode="expand",
            borderaxespad=0,
            borderpad=0,
            handlelength=0.9,
            ncol=ncol,
            numpoints=1,
        )
        legend_frame = legend.get_frame()
        legend_frame.set_linewidth(0)
        fig.tight_layout(pad=0, w_pad=0, h_pad=1.0)
        fig.canvas.draw()
        renderer = legend_ax.figure.canvas.get_renderer()
        h_packer = legend.get_children()[0].get_children()[1]
        target_width = h_packer.get_extent(renderer)[0]
        current_width = sum(
            [ch.get_extent(renderer)[0] for ch in h_packer.get_children()]
        )
        if target_width > 1.3 * current_width:
            break

    if save_formats:
        logger.log("Saving...")
        if name is None:
            if len(envs) > 1:
                name = "results"
            else:
                name = envs[0]
        for save_format in save_formats:
            file_name = name + "." + save_format
            fig.savefig(file_name, facecolor=fig.get_facecolor(), dpi=dpi)
            print(file_name)
        print("to", os.getcwd())

    return fig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", nargs="+", default=[])
    parser.add_argument("--x_axis", default="train/steps")
    parser.add_argument("--y_axes", nargs="+", default=["test/episode_score"])
    parser.add_argument("--x_label")
    parser.add_argument("--y_label")
    parser.add_argument("--interval", default="bounds")
    parser.add_argument("--window", type=int, default=1)
    parser.add_argument("--show_seeds", type=bool, default=False)
    parser.add_argument("--columns", type=int)
    parser.add_argument("--x_min", type=int)
    parser.add_argument("--x_max", type=int)
    parser.add_argument("--y_min", type=float)
    parser.add_argument("--y_max", type=float)
    parser.add_argument("--baselines", nargs="+")
    parser.add_argument("--baselines_source", default="tensorflow")
    parser.add_argument("--name")
    parser.add_argument("--save_formats", nargs="*", default=["pdf", "png"])
    parser.add_argument("--seconds", type=int, default=0)
    parser.add_argument("--cmap")
    parser.add_argument("--legend_columns", type=int)
    parser.add_argument("--font_size", type=int, default=12)
    parser.add_argument("--font_family", default="serif")
    parser.add_argument("--legend_font_size", type=int)
    parser.add_argument("--legend_marker_size", type=int, default=10)
    parser.add_argument("--backend", default=None)
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--title")
    args = parser.parse_args()

    has_gui = True
    if args.backend:
        mpl.use(args.backend)
        has_gui = args.backend.lower() != "agg"
    del args.backend

    plt.rc("font", family=args.font_family, size=args.font_size)
    if args.legend_font_size:
        plt.rc("legend", fontsize=args.legend_font_size)
    del args.font_family, args.font_size, args.legend_font_size

    seconds = args.seconds
    del args.seconds

    start_time = time.time()
    fig = plot(**vars(args), fig=None)

    try:
        if seconds == 0:
            if has_gui:
                while plt.get_fignums() != []:
                    plt.pause(0.1)
        else:
            while True:
                if has_gui:
                    while time.time() - start_time < seconds:
                        plt.pause(0.1)
                        assert plt.get_fignums() != []
                else:
                    time.sleep(seconds)
                start_time = time.time()
                plot(**vars(args), fig=fig)

    except Exception:
        pass


if __name__ == "__main__":
    main()
