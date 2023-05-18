import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import FormatStrFormatter
import os
import pandas as pd

def parse_logtrain(path_to_log_train: str) -> pd.DataFrame:
    epochs = []
    losses = []
    mean_squared_relative_mses = []
    val_epochs = []
    val_losses = []
    val_mean_squared_relative_mses = []

    with open(path_to_log_train) as fin:
        for line in fin:
            if "trainer" in line:
                if ("Saving" in line) or ("stops" in line):
                    continue
                if "epoch" in line and "val_epoch" not in line:
                    epoch = int(line.split(":")[-1])
                    epochs.append(epoch)
                elif "loss" in line and "val_loss" not in line:
                    loss = float(line.split(":")[-1])
                    losses.append(loss)
                elif "mean_squared_relative_mse" in line and "val_mean_squared_relative_mse" not in line:
                    mean_squared_relative_mse = float(line.split(":")[-1])
                    mean_squared_relative_mses.append(mean_squared_relative_mse)
                elif "val_epoch" in line:
                    val_epoch = float(line.split(":")[-1])
                    val_epochs.append(val_epoch)
                elif "val_loss" in line:
                    val_loss = float(line.split(":")[-1])
                    val_losses.append(val_loss)
                elif "val_mean_squared_relative_mse" in line:
                    val_mean_squared_relative_mse = float(line.split(":")[-1])
                    val_mean_squared_relative_mses.append(val_mean_squared_relative_mse)

    N = min(len(epochs), len(losses), len(mean_squared_relative_mses), len(val_losses), len(val_mean_squared_relative_mses))

    return pd.DataFrame({"epoch": epochs[:N], "loss": losses[:N], "mean_squared_relative_mse": mean_squared_relative_mses[:N],
                         "val_epoch": epochs[:N], "val_loss": val_losses[:N],
                         "val_mean_squared_relative_mse": val_mean_squared_relative_mses[:N],
                         })

def plot_train_result(ax, epochs, data_on_epochs, title, color, ymax):
    ax.plot(epochs, data_on_epochs, color=color, linewidth=3, alpha=0.7)
    ax.set_title(title, fontsize=22, fontweight="bold")
    ax.tick_params(labelsize=16)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, ymax + 0.05 * ymax)
    ax.set_xlabel("epoch", fontsize=6)

    return ax

def plot_valid_train(train_df, axs, color, loss_max, mse_max):
    plot_train_result(axs[0][0], train_df["epoch"], train_df["loss"], title="Loss / Train", color=color, ymax=loss_max)
    plot_train_result(axs[0][1], train_df["epoch"], train_df["val_loss"], title="Loss / Validation", color=color,
                      ymax=loss_max)
    plot_train_result(axs[1][0], train_df["epoch"], train_df["mean_squared_relative_mse"], title="Relative MSE / Train",
                      color=color, ymax=mse_max)
    plot_train_result(axs[1][1], train_df["epoch"], train_df["val_mean_squared_relative_mse"], title="Relative MSE / Validation",
                      color=color, ymax=mse_max)


if __name__ == "__main__":

    paths_to_log_dirs = [
        "../electrostatic/experiments/WormLikeChainGraphDataset_Net/log/EdgeConvNodeGATModel_example",
        "../gbsa/experiments/GbsaDgGraphDataset_Net/log/EdgeConvNodeGATModel_example",
        "../kd/EdgeConvNodeGAT/experiments/GbsaKdGraphDataset_Net/log/EdgeConvNodeGATModel_embeddings_example",
        "../kd/ProteinMPNN/experiments/GbsaKdGraphDataset_Net/log/ProteinMPNN_embeddings_example"]

    for ind_path, path in enumerate(paths_to_log_dirs):
        path_to_log = os.path.join(path, "log.log")

        loss_max = 0
        mse_max = 0

        df = parse_logtrain(path_to_log)

        loss_val = df["loss"].max()
        if loss_val > loss_max:
            loss_max = loss_val
        mse_val = max(df["mean_squared_relative_mse"].max(),
                      df["val_mean_squared_relative_mse"].max(),
                      )
        if mse_val > mse_max:
            mse_max = mse_val

        with PdfPages(path + "/TrainResults.pdf") as out_pdf:
            fig, axs = plt.subplots(figsize=(12, 7.5), nrows=2, ncols=2)
            plot_valid_train(df, axs, color='red', loss_max=loss_max, mse_max=mse_max)

            fig.tight_layout(rect=[0.03, 0.03, 0.97, 0.9])
            plt.suptitle(r"Train/Validation results", y=0.95, fontsize=25)

            axs[0][0].set_xlabel("")
            axs[0][1].set_xlabel("")
            axs[1][0].set_xlabel("epoch", fontsize=20, labelpad=5)
            axs[1][1].set_xlabel("epoch", fontsize=20, labelpad=5)

            for i in [0, 1]:
                for j in [0, 1]:
                    axs[i][j].set_box_aspect(0.6)
                    axs[i][j].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
                    axs[i][j].grid(alpha=0.3, lw=1.3)

            out_pdf.savefig(fig)
            plt.close()
