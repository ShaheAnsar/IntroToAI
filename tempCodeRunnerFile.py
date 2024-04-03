 7):
        plt.axhline(i-0.5, color='white', linewidth=2)
        plt.axvline(i-0.5, color='white', linewidth=2)

    text = f"Subgrid the bot should head to: {grid_coor[0], grid_coor[1]}"
    x = 0.5 # horizontally centered
    y = -0.11 # near the bottom of the image
    plt.gca().text(x, y, text, ha='center', va='bottom', \
                   fontsize=10, transform=plt.gca().transAxes)