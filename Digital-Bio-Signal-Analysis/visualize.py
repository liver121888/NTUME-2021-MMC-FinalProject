import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.base import clone
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score, classification_report

def plot_signal1d(x, fs=None, title='Signal'):
    """
    Plot the single-channel signal
    """
    plt.figure(figsize=(25, 5))
    if fs:
        t = np.arange(0, len(x) / fs, 1. / fs)
        plt.plot(t, x)
    else:
        plt.plot(x)
    plt.autoscale(tight=True)
    if fs:
        plt.xlabel('Time')
    else:
        plt.xlabel('Sample')
    plt.ylabel('Amplitude (mV)')
    plt.title(title)
    plt.show()


def plot_signalnd(x, fs=None, title='Signal'):
    """
    Plot the n-channel signal
    """
    t = None
    if fs:
        t = np.arange(0, len(x) / fs, 1. / fs)
    num_channels = len(x[0])
    fig, axs = plt.subplots(num_channels, 1, figsize=(25, 25))
    for i in range(num_channels):
        if fs:
            axs[i].plot(t, x[:, i])
            axs[i].set_xlabel('Time')
        else:
            axs[i].plot(x[:, i])
            axs[i].set_xlabel('Sample')
        axs[i].set_ylabel('Amplitude (mV)')
        axs[i].set_title(title + ' channel {}'.format(i + 1))
    plt.show()


def get_metrics(tgts, preds):
    print('Accuracy: {:.4f}'.format(accuracy_score(tgts, preds)))
    print('Precision: {:.4f}'.format(precision_score(tgts, preds, average='weighted')))
    print('Recall: {:.4f}'.format(recall_score(tgts, preds, average='weighted')))
    print('F1 Score: {:.4f}'.format(f1_score(tgts, preds, average='weighted')))


def get_classification_report(tgts, preds, classes):
    report = classification_report(tgts, preds, labels=classes)
    print(report)


def display_model_performance(tgts, preds, classes):
    print('Model metrics:')
    print('-' * 30)
    get_metrics(tgts, preds)
    print('\nModel classification report:')
    print('-' * 30)
    get_classification_report(tgts, preds, classes)


def plot_confusion_matrix(tgts, preds, labels, save_root='plots', save_name='image.png'):
    if not os.path.exists(save_root):
        os.mkdir(save_root)
    data = {'Actual': tgts,
            'Predicted': preds}

    df = pd.DataFrame(data, columns=['Actual', 'Predicted'])
    confusion_matrix = pd.crosstab(df['Actual'], df['Predicted'],
                                   rownames=['Actual'],
                                   colnames=['Predicted'])
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix, annot=True, xticklabels=labels, yticklabels=labels,  fmt='g')

    plt.savefig(os.path.join(save_root, save_name))
    plt.show()


def plot_model_roc_curve(clf, features, true_labels, label_encoder=None, class_names=None, save_name='ROC CURVE'):
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    if hasattr(clf, 'classes_'):
        class_labels = clf.classes_
    elif label_encoder:
        class_labels = label_encoder.classes_
    elif class_names:
        class_labels = class_names
    else:
        raise ValueError('Unable to derive prediction classes, please specify class_names!')
    n_classes = len(class_labels)
    y_test = label_binarize(true_labels, classes=class_labels)
    if n_classes == 2:
        if hasattr(clf, 'predict_proba'):
            prob = clf.predict_proba(features)
            y_score = prob[:, prob.shape[1] - 1]
        elif hasattr(clf, 'decision_function'):
            prob = clf.decision_function(features)
            y_score = prob[:, prob.shape[1] - 1]
        else:
            raise AttributeError('Estimator doesn\'t have a probability or confidence scoring system!')
        fpr, tpr, _ = roc_curve(y_test, y_score)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label='ROC curve (area = {0:0.2f})'.format(roc_auc), linewidth=2.5)
    elif n_classes > 2:
        if hasattr(clf, 'predict_proba'):
            y_score = clf.predict_proba(features)
        elif hasattr(clf, 'decision_function'):
            y_score = clf.decision_function(features)
        else:
            raise AttributeError('Estimator doesn\'t have a probability or confidence scoring system!')
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        # Compute micro-average ROC curve and ROC area
        fpr['micro'], tpr['micro'], _ = roc_curve(y_test.ravel(), y_score.ravel())
        roc_auc['micro'] = auc(fpr['micro'], tpr['micro'])
        # Compute macro-average ROC curve and ROC area
        # First aggregate all false positive rates
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
        # Then interpolate all ROC curves at this points
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
        # Finally average it and compute AUC
        mean_tpr /= n_classes
        fpr['macro'] = all_fpr
        tpr['macro'] = mean_tpr
        roc_auc['macro'] = auc(fpr['macro'], tpr['macro'])
        # Plot ROC curves
        plt.figure(figsize=(8, 6))
        plt.plot(fpr['micro'], tpr['micro'],
                 label='micro-average ROC curve (area = {0:0.2f})'.format(roc_auc['micro']), linewidth=3)
        plt.plot(fpr['macro'], tpr['macro'],
                 label='macro-average ROC curve (area = {0:0.2f})'.format(roc_auc['macro']), linewidth=3)
        for i, label in enumerate(class_labels):
            plt.plot(fpr[i], tpr[i],
                     label='ROC curve of class {0} (area = {1:0.2f})'.format(label, roc_auc[i]),
                     linewidth=2, linestyle=':')
    else:
        raise ValueError('Number of classes should be at least 2 or more')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('{}'.format(save_name))
    plt.legend(loc='lower right')
    plt.savefig('{}.png'.format(save_name))
    plt.show()


def plot_model_decision_surface(clf, train_features, train_labels, plot_step=0.02, cmap=plt.cm.RdYlBu,
                                markers=None, alphas=None, colors=None, title='DECISION SURFACE'):
    if train_features.shape[1] != 2:
        raise ValueError('X_train should have exactly 2 columns!')
    x_min, x_max = train_features[:, 0].min() - plot_step, train_features[:, 0].max() + plot_step
    y_min, y_max = train_features[:, 1].min() - plot_step, train_features[:, 1].max() + plot_step
    xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step), np.arange(y_min, y_max, plot_step))
    clf_est = clone(clf)
    clf_est.fit(train_features, train_labels)
    if hasattr(clf_est, 'predict_proba'):
        Z = clf_est.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
    else:
        Z = clf_est.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # plot contour
    plt.contourf(xx, yy, Z, cmap=cmap)
    le = LabelEncoder()
    y_enc = le.fit_transform(train_labels)
    n_classes = len(le.classes_)
    plot_colors = ''.join(colors) if colors else [None] * n_classes
    label_names = le.classes_
    markers = markers if markers else [None] * n_classes
    alphas = alphas if alphas else [None] * n_classes
    for i, color in zip(range(n_classes), plot_colors):
        idx = np.where(y_enc == i)
        plt.scatter(train_features[idx, 0], train_features[idx, 1], c=color,
                    label=label_names[i], cmap=cmap, edgecolors='black',
                    marker=markers[i], alpha=alphas[i])
    plt.legend()
    plt.savefig('{}.png'.format(title))
    plt.show()


def plot_loss_accuracy(train_loss, val_loss, train_acc, val_acc, colors,
                       loss_legend_loc='upper center', acc_legend_loc='upper left',
                       fig_size=(20, 10), sub_plot1=(1, 2, 1), sub_plot2=(1, 2, 2),
                       save_root='plots', save_name='image.png'):
    if not os.path.exists(save_root):
        os.mkdir(save_root)
    plt.rcParams['figure.figsize'] = fig_size
    fig = plt.figure()

    plt.subplot(sub_plot1[0], sub_plot1[1], sub_plot1[2])

    for i in range(len(train_loss)):
        x_train = range(len(train_loss[i]))
        x_val = range(len(val_loss[i]))

        min_train_loss = train_loss[i].min()

        min_val_loss = val_loss[i].min()

        plt.plot(x_train, train_loss[i], linestyle='-', color='tab:{}'.format(colors[0]),
                 label='TRAIN LOSS ({0:.4})'.format(min_train_loss))
        plt.plot(x_val, val_loss[i], linestyle='--', color='tab:{}'.format(colors[1]),
                 label='VALID LOSS ({0:.4})'.format(min_val_loss))

    plt.xlabel('epoch no.')
    plt.ylabel('loss')
    plt.legend(loc=loss_legend_loc)
    plt.title('Training and Validation Loss')

    plt.subplot(sub_plot2[0], sub_plot2[1], sub_plot2[2])

    for i in range(len(train_acc)):
        x_train = range(len(train_acc[i]))
        x_val = range(len(val_acc[i]))

        max_train_acc = train_acc[i].max()

        max_val_acc = val_acc[i].max()

        plt.plot(x_train, train_acc[i], linestyle='-', color='tab:{}'.format(colors[0]),
                 label='TRAIN ACC ({0:.4})'.format(max_train_acc))
        plt.plot(x_val, val_acc[i], linestyle='--', color='tab:{}'.format(colors[1]),
                 label='VALID ACC ({0:.4})'.format(max_val_acc))

    plt.xlabel('epoch no.')
    plt.ylabel('accuracy')
    plt.legend(loc=acc_legend_loc)
    plt.title('Training and Validation Accuracy')

    fig.savefig(os.path.join(save_root, save_name))
    plt.show()
