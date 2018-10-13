import matplotlib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def plot_blackjack_value_function(V, title="Value Function"):
    """
    Plots the value function as a surface plot.
    """
    min_x = min(k[0] for k in V.keys())
    max_x = max(k[0] for k in V.keys())
    min_y = min(k[1] for k in V.keys())
    max_y = max(k[1] for k in V.keys())

    x_range = np.arange(min_x, max_x + 1)
    y_range = np.arange(min_y, max_y + 1)
    X, Y = np.meshgrid(x_range, y_range)

    # Find value for all (x, y) coordinates
    Z_noace = np.apply_along_axis(lambda _: V[(_[0], _[1], False)], 2, np.dstack([X, Y]))
    Z_ace = np.apply_along_axis(lambda _: V[(_[0], _[1], True)], 2, np.dstack([X, Y]))

    def plot_surface(X, Y, Z, title):
        fig = plt.figure(figsize=(20, 10))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                               cmap=matplotlib.cm.coolwarm)
        ax.set_xlabel('Player Sum')
        ax.set_ylabel('Dealer Showing')
        ax.set_zlabel('Value')
        ax.set_title(title)
        ax.view_init(ax.elev, -120)
        fig.colorbar(surf)
        plt.show()

    plot_surface(X, Y, Z_noace, "{} (No Usable Ace)".format(title))
    plot_surface(X, Y, Z_ace, "{} (Usable Ace)".format(title))



def plot_blackjack_policy(Q, title="Policy"):
    """
    Plots the value function as a surface plot.
    """
    #x-axis display dealer showing
    #y-axis display player sum
    min_dealer = min(k[1] for k in Q.keys())
    max_dealer = max(k[1] for k in Q.keys())
    min_player = min(k[0] for k in Q.keys())
    max_player= max(k[0] for k in Q.keys())

    x_range = np.arange(min_dealer, max_dealer+1)
    y_range = np.arange(min_player, max_player+1)[::-1]

    noace_value = pd.DataFrame(index=y_range, columns=x_range, data=0)
    ace_value = pd.DataFrame(index=y_range, columns=x_range, data=0)
    for (player_score, dealer_score, usable_ace), q_values in Q.items():
        if usable_ace:
            ace_value.loc[player_score,dealer_score]=np.argmax(q_values)
        else:
            noace_value.loc[player_score,dealer_score]=np.argmax(q_values)

    def plot_heatmap(X, Y, value, title):
        action_space=['C','H']
        fig = plt.figure()
        ax = fig.add_subplot(111)
        surf = ax.imshow(value)
        ax.set_xlabel('Dealer Showing')
        ax.set_ylabel('Player Sum')

        ax.set_title(title)
        # We want to show all ticks...
        ax.set_xticks(np.arange(len(X)))
        ax.set_yticks(np.arange(len(Y)))
        # ... and label them with the respective list entries
        ax.set_xticklabels(X)
        ax.set_yticklabels(Y)
        # Loop over data dimensions and create text annotations.
        for i,x in enumerate(X):
            for j,y in enumerate(Y):
                text = ax.text(i, j, action_space[value.loc[y, x]],
                               ha="center", va="center", color="w")
        fig.tight_layout()
        fig.colorbar(surf)
        plt.show()

    plot_heatmap(x_range, y_range, ace_value, "{} (Usable Ace)".format(title))
    plot_heatmap(x_range, y_range, noace_value, "{} (No Usable Ace)".format(title))
    return noace_value,ace_value