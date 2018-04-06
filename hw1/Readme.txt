The version of python: 3.6.5
enter the folder code

HW1-1
A. Simulate a Function:
python hw1_1_simulate_function.py

B. Train on Actual Tasks:
python hw1_1_mnist.py
python hw1_1_cifar10.py

HW1-2
A. Visualize the optimization process
B. Observe gradient norm during training
python hw1_2_grad.py
C. What happens when gradient is almost zero?


HW1-3
A. Can network fit random variables?
python hw1_3_shuffle.py
B. Number of parameters v.s. Generalization
	using following program to generate different models with different numbers of parameters
		python hw1_3_2.py
	using following program to plot the final result
		python hw1_3_2_plot.py
C. Flatness v.s. Generalization
	using following program to collected loss and accuracy according to the corresponding alpha value
		python hw1_3_3_part1.py
	using following program to plot the final result
		python hw1_3_3_part1_plot.py
	