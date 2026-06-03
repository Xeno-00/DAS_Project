DAS Project 
Group 19 - Simone Romeo


The main codes are in src/aggregative_tracking
	
	In \aggregative tracking:
		robot_node.py: runs the algorithm for each agent. N of these are started.
		visualizer_node.py: creates the markers for RViz.

	In \launch_folder:
		param_launch.py: the actual file to launch in the terminal. It launches every necessary node and it is possible to choose the parameters here.


After the code has been launched, to plot the cost and gradient norm we can use plot_metrics.py, that takes the metrics from each agent (saved in robot_metrics folder) and merges them.

I also included the file plot_all.py, that I used to plot each graph topology in the same plotting. It uses the folders in robot_metrics (one for each agent, I created those manually). The code is not perfect and is basically a copy of plot_metrics.py bur iterated for each_graph, but it did its work.