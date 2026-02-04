import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib.axes
except ImportError:
    _has_plot = False
else:
    _has_plot = True

def PR_contour(ax:[matplotlib.axes.Axes|None]=None):
    """Generate a nice-looking contour plot for Precision vs. Recall

    REQUIRES optional [plot] dependencies!
    """
    if not _has_plot:
        raise ImportError("Optional contingiency[plot] dependencies required.")

    if ax is None: 
        ax = plt.gca()
    thres = np.linspace(0.2, 0.8, num=4)
    lines, labels = [], []
    for t in thres:
        # recall = np.linspace(0.00001, 1., num=100)
        # recall = np.logspace(-5, 0., num=100)
        recall_f1 = np.linspace(t/(2-t), 1.)
        recall_fm = np.linspace(t**2,1.)
        prec_f1 = t * recall_f1 / (2 * recall_f1 - t)
        # prec_f1 = 1/(2/t - 1/recall)
        # f1_bound = (recall>t/2)&(1.1>=prec_f1)# (0<=prec_f1)&(1>=prec_f1)
        prec_fm = t**2/recall_fm
        # fm_bound = (0<=y_fm)&(1>=y_fm)

        # (l,) = ax.plot(recall[f1_bound], prec_f1[f1_bound], color="0.8")
        # (l,) = ax.plot(x[fm_bound], y_fm[fm_bound], color="0.9")
        (l,) = ax.plot(recall_f1, prec_f1, color="0.8")
        (l,) = ax.plot(recall_fm, prec_fm, color="0.95")
        # midpt = y_fm[25]-0.03
        ax.annotate(f"{t:0.1f}", xy=(t-.02, t-0.02), color='0.8', bbox=dict(facecolor='white', linewidth=0, alpha=0.5))
        # print(y_f1[24])

        # plt.annotate("f1={0:0.1f}".format(f_score), xy=(1.1, y_f1[48]-0.01), color='xkcd:orange')
        # return plt.gca()
    # ax.legend()
    ax.annotate(r"$F_1$", xy=(1.01, 0.2/(2-0.2)-0.01), color='0.8')
    ax.annotate(r"F-M", xy=(1.01, 0.2**2-0.01), color='0.9')
    ax.set(
        ylim=(0,1.1),
        xlim=(0,1.1),
        ylabel='Precision',
        xlabel='Recall'
    )
