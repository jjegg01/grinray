import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap

SPHERE_POSITION = (0,0)
SPHERE_RADIUS = 1.0

# Main plotting function
def plot(tracetrees):
    # Create figure with sphere
    fig, ax = plt.subplots(figsize=((8,3.7)))

    # Go through all trace trees
    #cmap = plt.get_cmap("coolwarm")

    cmap = LinearSegmentedColormap.from_list(
        "Custom", [(1,0.5,0.3),(0.3,0.5,1)], N=2)

    for i, tracetree in enumerate(tracetrees):
        # Sample colormap to give this trace tree a unique color
        color = cmap(i / (len(tracetrees) - 1))
        _, _, ray_total, _ = tracetree
        plot_tracetree(tracetree, ray_total, color, ax)
    # Nicer formatting
    ax.set_xlim((-2,3))
    ax.set_ylim((-1.1,1.1))
    ax.set_aspect("equal")

    ax.add_patch(patches.Circle(SPHERE_POSITION, SPHERE_RADIUS, fill=False, zorder=9000))

    ax.xaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_major_locator(plt.NullLocator())

    # Save to file
    fig.tight_layout()
    fig.savefig("debugtrace.png", dpi=300)

def plot_tracetree(tracetree, ray_total, color, ax):
    # Draw line from previous location to the root of this tree using the
    # relative number of rays going down this tree as the opacity
    root_location, _, ray_count, children = tracetree
    for child in children:
        child_location, _, _, _ = child
        ax.plot(
            [root_location[0], child_location[0]],
            [root_location[2], child_location[2]],
            "-", linewidth=1, color=color, alpha=ray_count / ray_total)
    # Process child trees
    for child in children:
        plot_tracetree(child, ray_total, color, ax)

# Pretty printing for trace trees (useful if you want the actual numbers)
def prettyprint_tracetree(tracetree, level=0):
    root_location, root_event, ray_count, children = tracetree
    next_level = level + 1
    if root_event == 0:
        root_event = "Start"
    elif root_event == 1:
        root_event = "Intermediate"
        next_level = level
    elif root_event == 2:
        root_event = "Reflection"
    elif root_event == 3:
        root_event = "Refraction"
    elif root_event == 4:
        root_event = "End"
    print(" "*level + f"{root_event.upper()} at {root_location} (count: {ray_count})")
    for child in children:
        prettyprint_tracetree(child, next_level)