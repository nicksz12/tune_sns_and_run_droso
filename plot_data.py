import json
import matplotlib.pyplot as plt
import math


def plot_data(file_name, time_column_key="time"):
    """Load data from SNS+Mujoco model and plot based on user input.

    :param file_name: (str) directory of .json file to load and plot.
    :param time_column_key: (str) key that designates time, the independent variable in the dataset.
    :return: -
    """

    #  Load the .json
    try:
        with open(file_name, 'r') as file:
            data = json.load(file)
    except ValueError:
        print("File name " + str(file_name) + " not recognized as valid .json file.")

    plot_another_graph = True

    while plot_another_graph:
        #  Display the list of data that may be plotted
        i = 0
        for key in data.keys():
            print(str(i) + ":\t" + str(key))
            i = i + 1

        fields_to_plot = [int(x) for x in input("Enter a space-separated sequence of entries to plot: ").split(" ")]

        time_range = [0]
        while len(time_range) != 2:
            time_range = [float(x) for x in input("Enter a space-separated sequence of start time and "
                                                  "end time: ").split(" ")]
            if len(time_range) != 2:
                raise ValueError("Input is the wrong shape; must have exactly 2 elements.")

        #  Cut down time limits to be within the min and max time of the trial.
        time_range[0] = max(time_range[0], min(data[time_column_key]))
        time_range[1] = min(time_range[1], max(data[time_column_key]))

        subplot_size = [int(x) for x in input("Enter a space-separated sequence of rows and columns into "
                                              "which to arrange the subplots (0 0 puts them all on one set "
                                              "of axes): ").split(" ")]
        if len(subplot_size) != 2:
            raise ValueError("Input is the wrong shape; must have exactly 2 elements.")

        num_subplots = len(fields_to_plot)
        if subplot_size == [0, 0]:
            num_rows = 1
            num_cols = 1
            num_subplots = 0  # If they all go on one axis.
        elif math.prod(subplot_size) < num_subplots:
            # User asked for more subplots than their grid allows, so just give a column instead.
            num_rows = num_subplots
            num_cols = 1
        else:
            num_rows = subplot_size[0]
            num_cols = subplot_size[1]

        #  Create the figure
        key_list = list(data.keys())

        if num_subplots > 0:
            plt.figure()
            k = 0
            for i in range(num_rows):
                for j in range(num_cols):
                    if k < num_subplots:
                        data_key = key_list[fields_to_plot[k]]

                        plt.subplot(num_rows, num_cols, k + 1)
                        plt.plot(data["time"], data[data_key])
                        plt.title(data_key)
                        plt.grid()
                        plt.xlim(time_range[0], time_range[1])

                        k = k + 1
                        if i == num_rows - 1:
                            plt.xlabel("time (s)")
                        else:
                            plt.tick_params(axis='x', labelbottom=False)
            plt.suptitle(file_name)
        else:
            fig = plt.figure()
            for i in fields_to_plot:
                data_key = key_list[i]
                plt.plot(data["time"], data[data_key], label=data_key)
            plt.xlim(time_range[0], time_range[1])
            plt.grid()
            plt.legend()
            plt.xlabel("time (s)")
            plt.title(file_name)

        plt.draw()
        plt.show()

        to_quit = input("To quit, enter ''q''. To plot another graph, enter anything but q. ")

        if to_quit == "q" or to_quit == "Q":
            plot_another_graph = False
