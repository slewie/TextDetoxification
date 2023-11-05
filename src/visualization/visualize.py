import seaborn as sns
import matplotlib.pyplot as plt


def draw_histograms(data, *columns):
    """
    Function draws several histograms together
    :param data: dataframe with values
    :param columns: names of columns to be plotted
    :return:
    """
    fig, axs = plt.subplots(1, len(columns), figsize=(15, 4))
    for i in range(len(columns)):
        sns.histplot(data[columns[i]], ax=axs[i])
