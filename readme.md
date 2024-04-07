Bot 6:
    - added something to change the probability for when we find one alien. it still needs more work.
    the probability outside the detection square (now will be called 'square') will still probably go to 0
    cause i didn't change that part. this is there in the diffuse function. all i'm doing is scaling the 
    alien probability inside the square so that it'll be higher than the ones in the rest of the grid.
    the logic is something like, 
    m -> belief of the cells in the square
    s -> current sum of _all_ beliefs 
    n -> belief of all the cells in the rest of the grid (basically, s - m)

    α*m + (n / α) = s
    and solving it, we get α = (s / n)
