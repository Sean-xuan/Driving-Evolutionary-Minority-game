# Driving-Evolutionary-Minority-game
1. To see plots of this part, first run the main pragram, then try code similar to the follows:
   a = Evolution(2500, 201, 1000, 501)
   a.game()
   a.plot()
2. For newly replaced player, we have 2 steps of strategy change:
  - All the newly replaced player have probability of error_mut to make oppsite choice, at this time, 
        no changes need to be make in the main program
  - If the newly replaced player inherits Old driver's strategy, then the input state of him will change to a new one.
        For this situation, comment line 156 and 157, then uncomment line 155 and 158.
