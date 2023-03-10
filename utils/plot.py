import matplotlib.pyplot as plt
import seaborn as sns


def display_confusion_matrix(
        cmat,
        accuracy,
        score,
        precision,
        recall,
        filename,
        CLASS_MAP):
    plt.figure(figsize=(8, 6))
    ax = plt.gca()
    sns.heatmap(cmat, annot=True, fmt='.2g', ax=ax, cmap='Blues')
    ax.set_xticks(range(len(CLASS_MAP)))
    ax.set_xticklabels(CLASS_MAP, fontdict={'fontsize': 7})
    plt.setp(
        ax.get_xticklabels(),
        rotation=45,
        ha="left",
        rotation_mode="anchor")
    ax.set_yticks(range(len(CLASS_MAP)))
    ax.set_yticklabels(CLASS_MAP, fontdict={'fontsize': 7})
    plt.setp(
        ax.get_yticklabels(),
        rotation=45,
        ha="right",
        rotation_mode="anchor")
    titlestring = ""
    if score is not None:
        titlestring += 'F1 = {:.3f} '.format(score)
    if accuracy is not None:
        titlestring += 'Accuracy = {:.3f} '.format(accuracy)
    if precision is not None:
        titlestring += '\nPrecision = {:.3f} '.format(precision)
    if recall is not None:
        titlestring += 'Recall = {:.3f} '.format(recall)
    if len(titlestring) > 0:
        plt.title(titlestring)
    plt.savefig(filename)
    #plt.show()