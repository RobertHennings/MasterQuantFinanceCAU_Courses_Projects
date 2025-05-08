import matplotlib.pyplot as plt
import numpy as np

"""
Description:
In what follows is the implementation of a simple binomial tree function that
displays given prices for the asset and the numeraire asset and additional handy
data needed to check manual computations.
"""

def draw_binomial_tree(n: int, prices: list, numeraire: list,
                       space_offset: float, space_offset_numeraire: float,
                       space_offset_disc: float, space_offset_payoff: float):
    coordinates = [[0, 0]]
    node_labels = {}
    pos = {}
    # Next add all the information for plotting the numeraire asset and
    # the discounted prices
    node_labels_numeraire = {}
    node_labels_disc = {}
    node_labels_payoff = {}
    # Sanity Check: Check that enough values were supplied by 2**n rule
    for n_ in range(n):
        num_branches = 2**n_ if n_ != 0 else 0
        print(f"In n: {n_} there are {num_branches} branches")
        for y_ in np.linspace(-n_-space_offset, n_+space_offset, num_branches):
            coordinates.append([n_, y_])

    prices_unnpacked = [s for p in prices for s in p]
    numeraire_unpacked = [s for p in numeraire for s in p]
    disc_prices = np.round(np.array(prices_unnpacked) / np.array(numeraire_unpacked), 2)
    # Compute the given simple options payoff
    option_payoff = [call(price) for price in prices_unnpacked]

    for entry, price_lab in zip(coordinates, prices_unnpacked):
        pos[(entry[0], entry[1])] = (entry[0], entry[1])
        node_labels[(entry[0], entry[1])] = price_lab  # add the prices as labels
    print(node_labels)
    for k,v, d, p in zip(node_labels.keys(), numeraire_unpacked, disc_prices, option_payoff):
        node_labels_numeraire[(k[0], k[1]-space_offset_numeraire)] = v
        node_labels_disc[(k[0], k[1]-space_offset_disc)] = d
        node_labels_payoff[(k[0], k[1]-space_offset_payoff)] = p

    # Display the plot
    fig, ax = plt.subplots()
    # Plot a black separating line above the tree
    y_max = 6.2
    y_min = -(0.8*y_max)
    x = list(range(n))
    y = [y_max for n in range(n)]
    ax.plot(x, y, color="black")
    # Also display the current time step b above the tree and add. information
    for x_,y_ in zip(x, y):
        # Add time step information
        plt.text(x=x_, y=y_-0.3, s=f"n={x_}",
                 # fontdict={"size": 6}
                 )
        # Add price information
        plt.text(x=x_, y=y_-0.6, s=f"S_{x_}_1",
                 # fontdict={"size": 6}
                 )
        # Add numeraire information
        plt.text(x=x_, y=y_-0.9, s=f"S_{x_}_0",
                 # fontdict={"size": 6}
                 )
        # Add discounted prices information
        plt.text(x=x_, y=y_-1.2, s=f"^S_{x_}_1",
                 # fontdict={"size": 6}
                 )
        # Add the option payoff information
        plt.text(x=x_, y=y_-1.2, s=f"^X_{x_}",
                 # fontdict={"size": 6}
                 )
    
    for node, node_numeraire, node_disc, node_payoff in zip(node_labels, node_labels_numeraire, node_labels_disc, node_labels_payoff):
        plt.text(x=node[0], y=node[1], s=str(node_labels[node]))
        plt.text(x=node_numeraire[0], y=node_numeraire[1], s=str(node_labels_numeraire[node_numeraire]))
        plt.text(x=node_disc[0], y=node_disc[1], s=str(node_labels_disc[node_disc]))
        plt.text(x=node_payoff[0], y=node_payoff[1], s=str(node_labels_payoff[node_payoff]))

    ax.set_xlim(left=0, right=3)
    ax.set_ylim(bottom=y_min, top=y_max)
    plt.axis("off")
    plt.title("Binomial Tree")
    # Add diverging lines to the plot
    counter_0 = 0
    counter_1 = 1
    for i in range(7):
        counter_0 += 1 if i==0 else 2
        counter_1 +=1 if i==0 else 2
        x_start = list(node_labels_numeraire.keys())[i][0]
        y_start = list(node_labels_numeraire.keys())[i][1]

        x_end_1 = list(node_labels_numeraire.keys())[counter_0][0]
        y_end_1 = list(node_labels_numeraire.keys())[counter_0][1]
        
        x_end_2 = list(node_labels_numeraire.keys())[counter_1][0]
        y_end_2 = list(node_labels_numeraire.keys())[counter_1][1]
        
        plt.plot([x_start, x_end_1], [y_start, y_end_1], color="black", alpha=0.2)
        plt.plot([x_start, x_end_2], [y_start, y_end_2], color="black", alpha=0.2)

    plt.show()
    return node_labels, node_labels_numeraire

# Example usage
N = 4
r = 0.02
prices = [[100], [90, 110], [70, 85, 115, 130], [70, 85, 115, 130, 70, 85, 115, 130]]
numeraire = [[np.round((1+r)**n, 2)]*(2**n) for n in range(N)] # can also be replaced with custom evolution
space_offset = 1.4
space_offset_numeraire = 0.3
space_offset_disc = 0.6
space_offset_payoff = 0.8
K = 90
call = lambda x: np.max([x-K, 0])
node_labels, node_labels_numeraire = draw_binomial_tree(n=N, prices=prices, numeraire=numeraire,
                                 space_offset=space_offset,
                                 space_offset_numeraire=space_offset_numeraire,
                                 space_offset_disc=space_offset_disc,
                                 space_offset_payoff=space_offset_payoff)
# TODO:
# Add the martingale probabilities
# Add the expected value computations
# Add portfolio processes


# fig, ax = plt.subplots()
# counter_0 = 0
# counter_1 = 1
# for i in range(7):
#     counter_0 += 1 if i==0 else 2
#     counter_1 +=1 if i==0 else 2
#     #print(f"Combine the {i}th coordinate with the {counter_0} and the {counter_1}")
#     print(f"Combine {list(node_labels_numeraire.keys())[i]} with {list(node_labels_numeraire.keys())[counter_0]} and {list(node_labels_numeraire.keys())[counter_1]}")
#     x1, y1 = [list(node_labels_numeraire.keys())[i][0], list(node_labels_numeraire.keys())[counter_0][0]], [list(node_labels_numeraire.keys())[i][1], list(node_labels_numeraire.keys())[counter_1][1]]
#     x2, y2 = [list(node_labels_numeraire.keys())[i][0], list(node_labels_numeraire.keys())[counter_1][0]], [list(node_labels_numeraire.keys())[i][1], list(node_labels_numeraire.keys())[counter_1][1]]
#     x_start = list(node_labels_numeraire.keys())[i][0]
#     y_start = list(node_labels_numeraire.keys())[i][1]

#     x_end_1 = list(node_labels_numeraire.keys())[counter_0][0]
#     y_end_1 = list(node_labels_numeraire.keys())[counter_0][1]
    
#     x_end_2 = list(node_labels_numeraire.keys())[counter_1][0]
#     y_end_2 = list(node_labels_numeraire.keys())[counter_1][1]
    
#     plt.plot([x_start, x_end_1], [y_start, y_end_1], color="black")
#     plt.plot([x_start, x_end_2], [y_start, y_end_2], color="black")
# plt.show()
    
    
    
    


