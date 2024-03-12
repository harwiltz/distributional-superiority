import chex
import matplotlib
import matplotlib.figure
import matplotlib.pyplot as plt


def plot_decision_dists(decision_dists: chex.Array) -> matplotlib.figure.Figure:
    with plt.style.context("bmh"):
        fig, ax = plt.subplots()
        num_actions, num_atoms = decision_dists.shape
        for a in range(num_actions):
            ax.ecdf(decision_dists[a], label=f"Action {a}")
        ax.legend()
        return fig
