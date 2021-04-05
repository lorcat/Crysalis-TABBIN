# Crysalis-TABBIN
Jupyter notebook processing Rigaku Crysalis ".tabbin" files holding information about peak hunting.
Can be used for single crystal data obtained at conventional and high pressure experiments.

Files are selected using (watchdog)[https://github.com/gorakhargosh/watchdog "Python Watchdog"] file system events. 
Processing/filtering of the peaks in the files includes several algorithms, including
filtering reflections present at the same frame region (controlled by a region) over several frames and etc.

Path used by a watchdog can be coded within the first cell of Jupyter notebook.
