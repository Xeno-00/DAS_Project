DAS Project 
Group 19 - Simone Romeo



TASK 1: Multi-Robot Target Localization

BRIEF DESCRIPTION:

In this task, a fleet of robots cooperatively estimates the positions of unknown targets by sharing information through a graph. Each robot has access to a local cost function and implements the Gradient Tracking algorithm to solve a consensus optimization problem.


CODE OVERVIEW:

-T1.1.py:
	Main code. It is possible to choose the number of agents and then there is a cycle over each graph topology.

-T1.1(same graph_different N).py:
	As the name implies, it is possible to choose the graph topology and then there is a cycle over some chosen number gf agents to put in the list N_A. Useful to understand how varying the number of agents impacts the problem.

-T1.2.py
	Implements the previous algorithm in a cooperative localization of N_T targets. 


BEFORE RUNNING - PARAMETERS TO ADJUST (OPTIONAL):

-T1.1.py

	N: number of agents.

	d: dimensionality.

	max_iters: maximum iterations.

	alpha: step-size.

	graph_types: list of graph topologies.

-T1.1(same graph_different N).py:

	N_a: list of number of agents.

	d: dimensionality.

	max_iters: maximum iterations.

	alpha: step-size.

	graph_type: graph topology.

-T1.2.py 

	N: number of agents.

	N_T: number of targets

	d: dimensionality.

	max_iters: maximum iterations.

	alpha: step-size.

	graph_types: list of graph topologies.

	noise_std: noise



TASK 2: Aggregative Optimization for Multi-Robot Systems

BRIEF DESCRIPTION:

This task addresses a distributed control problem where each robot aims to stay close to a private target while maintaining cohesion with the fleet. The robots implement an Aggregative Tracking algorithm to minimize a local cost function that depends both on private goals and the barycenter of the team.


CODE OVERVIEW:

-T2.1.py
	After choosing the parameters, it performs the algorithm and plots cost, gradient norm and other insight. Also, in the end, it shows an animation.


BEFORE RUNNING - PARAMETERS TO ADJUST (OPTIONAL):

-T2.1.py

	N: number of agents.

	d: dimensionality.

	max_iters: maximum iterations.

	alpha: step-size.

	beta: tradeoff parameter

	graph_types: list of graph topologies.




