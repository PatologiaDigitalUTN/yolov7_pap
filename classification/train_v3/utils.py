import torch
import matplotlib
import matplotlib.pyplot as plt
from pandas import DataFrame
import os
import seaborn as sn
from sklearn.metrics import confusion_matrix, classification_report
from torch.utils.tensorboard import SummaryWriter

matplotlib.style.use('ggplot')
def save_model(epochs, model, optimizer, criterion, dest_path):
    """
    Function to save the trained model to disk.
    """
    torch.save({
                'epoch': epochs,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': criterion,
                }, f"{dest_path}/{os.path.basename(dest_path)}.pth")


def save_plots(train_acc, valid_acc, train_loss, valid_loss, pretrained):
    """
    Function to save the loss and accuracy plots to disk.
    """
    # accuracy plots
    plt.figure(figsize=(10, 7))
    plt.plot(
        train_acc, color='green', linestyle='-', 
        label='train accuracy'
    )
    plt.plot(
        valid_acc, color='blue', linestyle='-', 
        label='validataion accuracy'
    )
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(f"../outputs/accuracy_pretrained_{pretrained}.png")
    
    # loss plots
    plt.figure(figsize=(10, 7))
    plt.plot(
        train_loss, color='orange', linestyle='-', 
        label='train loss'
    )
    plt.plot(
        valid_loss, color='red', linestyle='-', 
        label='validataion loss'
    )
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f"../outputs/loss_pretrained_{pretrained}.png")


def conf_matrix_report(predictions, targets, writer, train_dataset, dest_path=None):
    "Creates a confussion matrix and a report of the classification. Adds it to Tensorboard"
    predictions = torch.cat(predictions, dim=0)
    targets = torch.cat(targets, dim=0)
    conf_matrix = confusion_matrix(targets.cpu().numpy(), predictions.cpu().numpy())

    df_cm = DataFrame(conf_matrix , index=train_dataset.class_to_idx, columns=train_dataset.class_to_idx)
    class_report = classification_report(targets.cpu().numpy(), predictions.cpu().numpy(), target_names=train_dataset.class_to_idx)
    print(f'Confusion matrix:\n{conf_matrix}')
    print(f'Classification report:\n{class_report}')

    fig = plt.figure()
    sn.set(font_scale=1.4)
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 16}, cmap='Blues', fmt='g')
    plt.xlabel('Predicted')
    plt.ylabel('True')

    # Add conf matrix and report to tensorboard
    writer.add_figure('confussion_matrix', fig)
    writer.add_text('classification_report', class_report)

    if dest_path is not None:
        # save conf matrix
        plt.savefig(os.path.join(dest_path, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
        # save class report
        with open(os.path.join(dest_path, 'classification_report.txt'), 'w') as f:
            f.write(class_report)
