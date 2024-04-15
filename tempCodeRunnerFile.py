g = Grid2()
b = bot3(g)
a = Alien(g._grid)
MAX_TURNS = 500
turns = 0
for _ in range(MAX_TURNS):
    print(f"Turn {_}")
    b.move()
    if g.grid[a.ind[1]][a.ind[0]].alien_belief == 0:
        print("Alien belief 0 at alien position!!!!")
    #plot_world_state(g, b)
    #plt.show()
    a.move()
    plot_world_state(g, b)
    plt.savefig(f"tmp{_}.png", dpi=200)
    plt.close()
    #plt.show()
    gif_coll.append(Image.open(f"tmp{_}.png"))
    turns += 1
    if g.crew_pos == b.pos:
        print("SUCCES: Crew member reached!")
        break