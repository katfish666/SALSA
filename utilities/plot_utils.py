import matplotlib.pyplot as plt
from IPython.display import clear_output

def live_plot(data_dict, bs, e, figsize=(10,3), save=False, path=None):
    clear_output(wait=True)
    plt.figure(figsize=figsize)
    for label,data in data_dict.items():
        plt.plot(data, label=label)
    title=f'Batch size: {bs}, Epochs: {e}'
    plt.title(title)
    plt.grid(True)
    plt.xlabel('Run')
    plt.legend(loc='upper right') # the plot evolves to the right
    if save:
        plt.savefig(path, bbox_inches='tight')
    plt.show();