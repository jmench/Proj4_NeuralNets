# __Proj4_NeuralNets__
### Author: Jordan Menchen

## __How to Run:__
#### main.py
- run command `python -W ignore main.py` and the program will train the model and test the model, only
outputting progress of the model and finally the accuracy of the model. The `-W ignore` simply keeps
the program from outputting unnecessary warnings to the screen while running.

#### bonus.py
- run command `python -W ignore bonus.py` and the program will begin with a prompt
- select the database to use by following the prompt and entering correct value
- after entering choice, the database will load the correct datasets and perform the training and
testing just like in `main.py`. The `-W ignore` simply keeps
the program from outputting unnecessary warnings to the screen while running.

## __NOTES__
- Both `main.py` and `bonus.py` have parameters that can be edited in their `main()` functions
- These parameters include: bias values, learning rate, number of hidden nodes, and number of epochs
- Changing these values will change the run time and accuracy of the program
- The run time for `main.py` on my machine with 3 epochs and 100 hidden nodes takes about 1.5 hours.
__Please be patient__ when running and use the progress outputs to estimate how long it will take
to build a model. There is a status bar that will show the progress of each epoch.
