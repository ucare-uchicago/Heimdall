'''
    Plot bar graph showing latency reduction across different algorithms at specific percentile/characteristic.

    This file will conduct the following steps:

        1. Read the raw latency output file from several combinations.

        2. For each combination, calculate its characteristics (p99.99, p99.9, p99, p95, p90, median, avg) for each algorithm. 

        3. Average the characteristic value of all the combinations for each characteristic, plot:
            a. bar graph of average latency reduction.
            b. line chart of latencies on thorough percentiles.
'''


import argparse
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import os
import sys
from typing import List, Dict

RESULT_PATH = "../benchmark/results"
ALGORITHMS_TO_BE_COMPARED = ["baseline", "random", "heimdall"]
CHARACTERISTICS = ["p99.99", "p99.9", "p99", "p95", "p90", "p80", "median", "avg"]
BAR_GRAPH_CHARACTERISTIC = ["avg"]
LINE_CHART_X = ["50", "80", "90", "95", "99", "99.9", "99.99"]
ALGO_LINE_CHART_STYLE = {
    "baseline": "s-",
    "random": "o-",
    "heimdall": "D--"
}
ALGO_LINE_CHART_COLOR = {
    "baseline": "magenta",
    "random": "red",
    "heimdall": "lime"
}

legend_elements = [
    Line2D([0], [0], marker=ALGO_LINE_CHART_STYLE[algo][0], color=ALGO_LINE_CHART_COLOR[algo], label=algo, markerfacecolor=ALGO_LINE_CHART_COLOR[algo], markersize=10, linestyle=ALGO_LINE_CHART_STYLE[algo][1:]) for algo in ALGORITHMS_TO_BE_COMPARED
]


def plot_graph() -> None:
    '''
        Plot the bar graph showing latency reduction across different algorithms at specific percentile/characteristic.

        Output:
            Will save the bar graph under `../benchmark/results/eval_figure/<characteristic>_value.png`

        Return:
            None
    '''

    combinations_path = []
    combinations_path = get_combinations_path(RESULT_PATH)

    # for each combination, extra its latency characteristics
    combinations_characteristics = []
    for com_path in combinations_path:

        algos_dict = {}  # to store different algos' characteristics

        # get the characteristic of other algorithms
        for algo in ALGORITHMS_TO_BE_COMPARED:

            characteristics_value_dict = {}   # to store different characteristics

            # read the latency results of this combination replayed with this algo
            lat = read_lat_file(com_path, algo)
            
            # extract the characteristics of this combination replayed with this algo
            for characteristic_name in CHARACTERISTICS:
                value = None
                value = extract_characteristic(characteristic_name, lat)
                assert value != None, "Extract Characteristic <{}> Fail.".format(characteristic_name)
                characteristics_value_dict[characteristic_name] = value / 1000 # from us to ms
            
            algos_dict[algo] = characteristics_value_dict
    
        combinations_characteristics.append(algos_dict)

    # plot and save the bar graph
    plot_and_save(combinations_path, combinations_characteristics)


def plot_and_save(combinations_path: List[str], combinations_characteristics: List[Dict[str, Dict[str, float]]]) -> None:
    '''
        Plot the bar graph for each characteristics, then save it under `RESULT_PATH`/`eval_figure`

        Args:
            @combinations_path: List of path of all combinations that will be included in graph.
            @combinations_characteristics: Characterisitcs reduction of each combination.

        Output:
            Bar graphs for each characteristic.  
    '''
    full_output_dir = os.path.join(RESULT_PATH, "eval_figure")
    os.makedirs(full_output_dir, exist_ok=True)

    # Records all the combinations being included in graphs.
    included_combinations_record = os.path.join(full_output_dir, "included_combinations.txt")
    with open(included_combinations_record, "w") as f:
        for com_path in combinations_path:
            f.write(com_path + "\n")
    print("===== Included Combinations are recorded in: {}".format(included_combinations_record))

    label = ['Baseline', 'Random', 'Heimdall']
    font_colors = ['black', 'black', 'black', 'black']
    colors = ['red', 'magenta', 'limegreen', 'blue']
    # label = ['Baseline', 'Random', 'Heimdall']
    # font_colors = ['black', 'black', 'black']
    # colors = ['red', 'magenta', 'blue']

    # For each characteristic, plot a corresponding bar graph
    for characterisitc_name in BAR_GRAPH_CHARACTERISTIC:
        avg_value_list = []
        for algo in ALGORITHMS_TO_BE_COMPARED:
            combination_value_list = [combinations_characteristics[i][algo][characterisitc_name] for i in range(len(combinations_characteristics))]
            avg_value = np.mean(combination_value_list)
            avg_value_list.append(avg_value)
        
        figure_path = os.path.join(full_output_dir, "{}_value.png".format(characterisitc_name))

        fig, ax = plt.subplots(figsize=(2.4, 3.6))
        ax.bar(ALGORITHMS_TO_BE_COMPARED, avg_value_list, label=label, color=colors)

        ax.set_xlabel("Algorithms")
        ax.set_ylabel("Average Latency (ms)")
        ax.yaxis.set_major_locator(ticker.FixedLocator([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]))
        ax.set_yticklabels(["0", ".1", ".2", ".3", ".4", ".5", ".6", ".7", ".8"])

        xtick_locations = range(len(label)) 
        ax.xaxis.set_major_locator(ticker.FixedLocator(xtick_locations))
        ax.set_xticklabels(label, rotation=90)
        ax.tick_params(axis='x', which='major', pad=-10)
        for ticklabel, tickcolor in zip(fig.gca().get_xticklabels(), font_colors):
            ticklabel.set_color(tickcolor)
            ticklabel.set_horizontalalignment("center")
            ticklabel.set_verticalalignment("bottom")
        plt.savefig(figure_path, dpi=200, bbox_inches='tight', format='png')
        plt.clf()
        print("===== output figure : " + figure_path)

    # Plot extra chart graph
    figure_path = os.path.join(full_output_dir, "line_chart.png")

    _, _ = plt.subplots(figsize = (3.2, 3.2))
    for algo in ALGORITHMS_TO_BE_COMPARED:
        algo_chracteristic_value = []
        for characterisitc_name in LINE_CHART_X:
            if characterisitc_name == "50" or characterisitc_name == "median":
                combination_value_list = [combinations_characteristics[i][algo]["median"] for i in range(len(combinations_characteristics))]
            else:
                characterisitc_name = "p{}".format(characterisitc_name)
                combination_value_list = [combinations_characteristics[i][algo][characterisitc_name] for i in range(len(combinations_characteristics))]
            avg_value = np.mean(combination_value_list)
            algo_chracteristic_value.append(avg_value)

        plt.plot(LINE_CHART_X, algo_chracteristic_value, ALGO_LINE_CHART_STYLE[algo], label=algo, color=ALGO_LINE_CHART_COLOR[algo], linewidth=1, markersize=4)
    plt.xlabel('Percentile')
    plt.ylabel('Read Lat. (ms)')
    plt.ylim(0, 3)
    plt.legend(handles=legend_elements, loc='upper left', frameon=False, labelspacing=0.25, columnspacing=0.5, borderpad=-0.2, handletextpad=0.2, handlelength=2)
    plt.savefig(figure_path, dpi=200, bbox_inches='tight', format='png')
    plt.clf()
    print("===== output figure : " + figure_path)
        

def read_lat_file(combination_path: str, algo: str) -> List[int]:
    '''
        Read latency results of this combination replayed with this algo

        Args:
            @combination_path: path of directory that store the combination replayed result
            @algo: specify the algorithm name you wanna read the replayed latency results out

        Return:
            @lat: list of latency value from this combination's replayed result using this algorithm. 
    '''
    algo_lat_file_path = "{}/{}/2ssds_{}.data".format(combination_path, algo, algo)
    algo_lat_file = open(algo_lat_file_path)
    lat_line = algo_lat_file.readline()
    lat = []
    while lat_line:
        if int(lat_line.split(",")[2]) == 1:    # only count read IO, 1 is the indicator of read IO
            lat.append(int(lat_line.split(",")[1]))
        lat_line = algo_lat_file.readline()

    return lat


def extract_characteristic(characteristic_name: str, list_of_value: List[str]) -> int:
    '''
        Specify a name of characteristic and a list of value, 
        calculate the corresponding characterisitcs of this value list.

        Args:
            @characteristic_name: name of the characteristic, e.g. p99.9
            @list_of_value
        
        Return:
            @characteristic_value: the corresponding characteristic value
    '''
    characteristic_value = None
    if characteristic_name == "p99.99":
        characteristic_value = np.percentile(list_of_value, 99.99)
    elif characteristic_name == "p99.9":
        characteristic_value = np.percentile(list_of_value, 99.9)
    elif characteristic_name == "p99":
        characteristic_value = np.percentile(list_of_value, 99)
    elif characteristic_name == "p95":
        characteristic_value = np.percentile(list_of_value, 95)
    elif characteristic_name == "p90":
        characteristic_value = np.percentile(list_of_value, 90)
    elif characteristic_name == "p80":
        characteristic_value = np.percentile(list_of_value, 80)
    elif characteristic_name == "median":
        characteristic_value = np.percentile(list_of_value, 50)
    elif characteristic_name == "avg":
        characteristic_value = np.mean(list_of_value)
    else:
        print("[ERROR] characteristic <{}> not support.".format(characteristic_name))
        exit(1)
    return characteristic_value


def get_combinations_path(result_path: str) -> List[str]:
    '''
        Given path of directory storing results,
        Extract the path of each combinations.
        
        Return:
            @combinations_path: List of all combinations' paths under the result directory.
    '''
    combinations_path = []
    with os.scandir(result_path) as entries:
        for entry in entries:
            if (entry.is_dir()) and ('eval_figure' not in entry.path):
                # Append the path of the directory to the list
                combinations_path.append(entry.path)

    return combinations_path


if __name__ == '__main__':

    plot_graph()