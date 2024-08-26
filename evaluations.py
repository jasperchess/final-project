import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
import gradcam

def plot_confusion_matrix(matrix, title, labels=['Normal', 'Tuberculosis']):
    plt.title(title)
    sns.heatmap(matrix, annot=True, fmt='d', cmap=plt.cm.Blues, xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted Class', fontsize=10)
    plt.ylabel('Actual Class', fontsize=10)

def visualize_confusion_matrices(predicted_y, true_y, title):
    plt.figure(figsize=(20, 15))
    plt.suptitle(title, fontsize=18)

    mean_matrix = np.array([[0, 0], [0, 0]])
    for i, _ in enumerate(predicted_y):
        pred = predicted_y[i]
        pred = (pred > 0.5).astype(int)
        true = true_y[i]
        matrix = confusion_matrix(true, pred)
        mean_matrix += np.array(matrix)
        plt.subplot(3, 3, i + 1)
        plot_confusion_matrix(matrix, 'Model %d' % (i + 1))

    mean_matrix = mean_matrix / len(predicted_y)
    mean_matrix = np.around(mean_matrix).astype(int)
    plt.subplot(3, 3, len(predicted_y) + 1)
    plot_confusion_matrix(mean_matrix, 'Mean Confusion Matrix')
    plt.show()

def plot_loss_curve(loss, val_loss):
    plt.plot(loss, label='loss')
    plt.plot(val_loss, label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

def visualize_history_val_loss(loss, val_loss, title):
    plt.figure(figsize=(20, 15))
    plt.suptitle(title, fontsize=18)

    for i in range(len(loss)):
        plt.subplot(3, 3, i + 1)
        plot_loss_curve(loss[i], val_loss[i])

    plt.show()

def visualize_grad_cam(img_path, model, layer_idx, model_idx):
    img_array = gradcam.get_img_array(img_path, size=(256, 256), color_mode='grayscale')
    model.layers[-1].activation = None
    plt.figure(figsize=(20, 15))

    mpl.colormaps["jet"]
    heatmap = gradcam.make_gradcam_heatmap(img_array, model, layer_idx=layer_idx)

    plt.matshow(heatmap)
    plt.title("Heatmap for Model %d" % (model_idx + 1))
    plt.show(gradcam.overlayed_gradcam(img_path, heatmap))
    plt.title("Superimposed for Model %d" % (model_idx + 1))

