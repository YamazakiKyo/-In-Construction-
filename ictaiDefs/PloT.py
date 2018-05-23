import matplotlib.pyplot as plt

def plot_anime(iter_set, loss_set, train_acc_set, val_acc_set):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.set_xlabel('Iterations')
    ax.set_ylabel('% Accuracy')
    ax.set_ylim(0, 1)
    ax2 = ax.twinx()
    ax2.set_ylabel('Loss')
    plt.ion()
    plt.show()
    try:
        ax.lines.remove(line1)
        ax.lines.remove(line2)
        ax2.lines.remove(line3)
    except Exception:
        pass
    line1, = ax.plot(iter_set, train_acc_set, '-r', label='Training_acc')
    line2, = ax.plot(iter_set, val_acc_set, '-b', label='Validation_acc')
    line3, = ax2.plot(iter_set, loss_set, '-k', label='Loss')
    ax.legend(loc=3)
    ax2.legend(loc=8)
    plt.pause(0.1)
