import matplotlib
import matplotlib as mpl
import matplotlib.colors
import matplotlib.patches as patches
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.path import Path
import textwrap  # Added for multi-line support
rc = {'font.size': 10, 'axes.labelsize': 10, 'legend.fontsize': 10.0,
      'axes.titlesize': 25, 'xtick.labelsize': 18, 'ytick.labelsize': 16}
plt.rcParams.update(**rc)
mpl.rcParams['axes.linewidth'] = .5  # set the value globally


# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"]="0"

def get_canvas(words, x, H=200, W=50):
    """
    Create a canvas with points based on the number of words in a sentence
    Args:
        words:
        x:
        H:
        W:

    Returns:

    """
    ntoks = len(''.join(words))
    W_all = W * ntoks
    delta_even = int(W_all / ntoks)

    X = np.zeros((H, W_all))
    x0 = 0

    x_centers = []
    for i, (w_, b) in enumerate(zip(words, x)):
        delta = int((len(w_) / ntoks) * W_all)

        delta = int((0.85 * delta_even + 0.15 * delta))
        X[:, x0:x0 + delta] = b

        x_centers.append(x0 + int(delta / 2))
        x0 = x0 + delta

    return np.asarray(x_centers)[np.newaxis, :].T


def alpha_blending(relevance, alpha, threshold:float=0.) -> str:
    """
    Alpha blending for relevance value given a threshold.
    Args:
        relevance:
        alpha:
        threshold:

    Returns:

    """
    colors = {'red': [1., 0., 0.],
              'blue': [0., 0, 1.],
              }
    c = 'red' if relevance > threshold else 'blue'

    return matplotlib.colors.to_hex(colors[c] + [alpha], keep_alpha=True)


def plot_bilrp_sentences(decoded,
                         outs,
                         relevance_scores,
                         show_colorbar=False,
                         title=True,
                         yticks_fontsize=20,
                         title_fontsize=24,
                         linewidth=6.,
                         wrap_width=20,
                         name=None): # Added wrap_width parameter
    """
    Plot the BiLRP sequence-to-sequence plots with support for multi-line labels.
    """
    
    # Process 'decoded' to wrap long sentences into multiple lines
    wrapped_decoded = []
    for sentence_list in decoded:
        wrapped_sentence = [textwrap.fill(str(word), width=wrap_width) for word in sentence_list]
        wrapped_decoded.append(wrapped_sentence)
    
    # Use the length of the original strings for canvas calculation 
    # to maintain the original spacing logic
    x_centers0 = get_canvas(decoded[0],
                            outs[0]['Rsen'].squeeze()[0, :] / np.abs(
                                outs[0]['Rsen'].squeeze().std()),
                            H=70,
                            W=52)

    x_centers1 = get_canvas(decoded[1],
                            outs[1]['Rsen'].squeeze()[0, :] / np.abs(
                                outs[0]['Rsen'].squeeze().std()),
                            H=70,
                            W=52)

    x_centers1 = (x_centers1 - x_centers1.min()) / (
        x_centers1.max() - x_centers1.min()) * (
                     x_centers0.max() - x_centers0.min()) + x_centers0.min()

    ys = np.array(np.meshgrid(x_centers0, x_centers1.T),
                  dtype=float).T.reshape(-1, 2)

    ymins = ys.min(axis=0)
    ymaxs = ys.max(axis=0)
    dys = ymaxs - ymins
    ymins -= dys * 0.05
    ymaxs += dys * 0.05

    dys = ymaxs - ymins

    zs = np.zeros_like(ys)
    zs[:, 0] = ys[:, 0]
    zs[:, 1:] = (ys[:, 1:] - ymins[1:]) / dys[1:] * dys[0] + ymins[0]

    fig, host = plt.subplots(figsize=(12, 16)) # Slightly increased height for multi-line text

    axes = [host] + [host.twinx() for i in range(ys.shape[1] - 1)]
    for i, ax in enumerate(axes):
        ax.set_ylim(ymins[i], ymaxs[i])
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.set_yticks(np.unique(ys[:, i]))
        if ax != host:
            ax.spines['left'].set_visible(False)
            ax.yaxis.set_ticks_position('right')
            ax.spines["right"].set_position(("axes", i / (ys.shape[i] - 1)))

        # Use the wrapped_decoded list here for the labels
        ax.set_yticklabels(wrapped_decoded[i], fontsize=yticks_fontsize, va='center')
        ax.set_ylim(ax.get_ylim()[::-1])

    # ... (rest of the plotting logic remains the same) ...
    host.set_xlim(0, ys.shape[1] - 1)
    host.set_xticks(range(ys.shape[1]))
    host.set_xticklabels(["", ""])
    host.tick_params(axis='x', which='major', pad=7)
    host.spines['right'].set_visible(False)
    host.xaxis.tick_top()
    
    if title:
        host.set_title(title, fontsize=title_fontsize, pad=12)

    for j in range(ys.shape[0]):
        verts = list(zip([x for x in np.linspace(0, len(ys) - 1, len(ys) * 3 - 2, endpoint=True)],
                         np.repeat(ys[j, :], 2)))
        codes = [Path.MOVETO] + [Path.CURVE4 for _ in range(len(verts) - 1)]
        path = Path(verts, codes)
        x = np.where(x_centers0.squeeze() == ys[j, 0])
        y = np.where(x_centers1.squeeze() == ys[j, 1])
        raw_score = relevance_scores[x, y].squeeze().tolist()
        # 2. Increase Contrast (The "Suppressor")
        # Raising to power 2 or 3 makes weak signals disappear.
        p = 3.0 
        suppressed_score = np.sign(raw_score) * (np.abs(raw_score) ** p)

        # 3. Scale for visibility & Safety Clip
        # Multiply by a factor (e.g., 2.0) to make the strong ones pop, 
        # then clip to 1.0 so it doesn't crash.
        final_score = np.clip(suppressed_score * 2.5, -1.0, 1.0)

        color = alpha_blending(relevance=final_score, alpha=abs(final_score))
        patch = patches.PathPatch(path, facecolor='none', linewidth=linewidth, edgecolor=color)
        host.add_patch(patch)

    if show_colorbar:
        p = PatchCollection(host.patches, cmap=plt.get_cmap("bwr"), alpha=1.)
        p.set_clim(-1., 1.)
        plt.colorbar(p, orientation='vertical', location="right", pad=0.15)

    plt.tight_layout()
    plt.savefig(name+'.png')

def plot_relevance_conservation(data_in,
                                data_names,
                                plot_title:str="scatter_plot",
                                usetex:bool=False) -> None:
    """
    Plot the relevance conservation scores as a scatter plot of the sum
    of the relevance scores and the sentence embedding features.
    Args:
        data_in: the output of the model
        data_names:
        plot_title: the title of the plot
        usetex: using latex makes the y label pretier
    """
    fig, axis = plt.subplots(2, 1, figsize=(12, 12))

    scatter_colors = "#848586"

    for idx, ax_scatter in enumerate(axis):
        ax_scatter.set_xlabel(data_names[0], fontsize=50, usetex=usetex,
                              labelpad=18)
        ax_scatter.set_ylabel(f"Sentence {idx}", fontsize=50, usetex=usetex,
                              labelpad=18)
        ax_scatter.xaxis.set_tick_params(labelsize=25)
        ax_scatter.yaxis.set_tick_params(labelsize=25)

        _ = ax_scatter.scatter(data_in["sentence_embeddings"][idx]
                                    .detach().cpu().numpy().squeeze(),
                               data_in["Rsen"][idx].sum(axis=2).squeeze(),
                               c=scatter_colors,
                               s=55)

        ax_scatter.axline((0, 0), (1, 1), linewidth=2, color='k', ls='-')
        ax_scatter.set_title(plot_title, fontsize=26)
        ax_scatter.spines['right'].set_visible(False)
        ax_scatter.spines['top'].set_visible(False)
    fig.suptitle('Conservation plots for current sample', fontsize=60)
    fig.supylabel(data_names[1], fontsize=50, usetex=usetex,
                  position=(-0.1, 0.5))
    fig.tight_layout()
    plt.show()
    plt.close(fig)
