import argparse
import os
import os.path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def run_parse(path_to_log_dir: str):
    for dirpath, dirnames, filenames in os.walk(path_to_log_dir):
        for filename in [f for f in filenames if f.endswith(".log")]:
            file = os.path.join(dirpath, filename)

            print(file)

            with open(file, 'r') as f:
                for index, line in enumerate(f):
                    # search string
                    if 'Trainable parameters' in line:
                        skip = index+1
                        break

            logfile = pd.read_csv(file, skiprows=skip, sep=" - ")

            logfile = logfile.drop(columns=logfile.columns[:4])

            lims_x_max = max(logfile['predicted'].values)
            lims_y_max = max(logfile['target'].values)
            lims_max = max(lims_y_max, lims_x_max)
            lims_x_min = min(logfile['predicted'].values)
            lims_y_min = min(logfile['target'].values)
            lims_min = min(lims_y_min, lims_x_min)
            fig = plt.figure(figsize=(6, 6))
            ax = fig.add_subplot(111)
            ax.set_xlim(lims_min, lims_max)
            ax.set_ylim(lims_min, lims_max)

            font_size = 25

            ax.plot(logfile['predicted'].values, logfile['target'].values, marker="o", ls="", color="red", alpha=0.2)
            ax.plot([lims_min, lims_max], [lims_min, lims_max], color='k', lw=1.5)
            ax.tick_params(labelsize=font_size, pad=15)
            ax.set_title('R = {:.02f}'.format(np.corrcoef(logfile['predicted'].values, logfile['target'].values)[0][1]),
                         fontsize=font_size, pad=8)

            ax.set_ylabel("target", fontsize=font_size, labelpad=15)
            ax.set_xlabel("predicted", fontsize=font_size, labelpad=20)

            fig.tight_layout()
            plt.savefig(os.path.join(dirpath, 'corr_test.png'), dpi=800)
            plt.close()
            
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.hist(logfile['target'], density=True, bins=100, color='blue')
            ax.set_title("Target distribution", fontsize=22)
            fig.tight_layout()
            plt.savefig(os.path.join(dirpath, 'target_distribution.png'))
            plt.close()

            fig = plt.figure()
            ax = fig.add_subplot(111)

            ax.hist(logfile['predicted'], density=True, color="red", bins=100)
            ax.set_title("Predicted distribution", fontsize=22)
            fig.tight_layout()
            plt.savefig(os.path.join(dirpath, 'predicted_distribution.png'))
            plt.close()

            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.hist(logfile['target'], range=[lims_min, lims_max], density=True, bins=100, color="blue", alpha=0.5)
            ax.set_title("Distributions", fontsize=22)
            ax.hist(logfile['predicted'], range=[lims_min, lims_max], density=True, bins=100, color="red", alpha=0.5)
            ax.tick_params(labelsize=25, pad=15)
            fig.tight_layout()
            plt.savefig(os.path.join(dirpath, 'both_distributions.png'), bbox_inches='tight')
            plt.close()

            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.hist(logfile['target'] - logfile['predicted'], density=True, bins=50, color='green')
            ax.set_title("Error distribution", fontsize=22)
            fig.tight_layout()
            plt.savefig(os.path.join(dirpath, 'error_distribution.png'))
            plt.close()


if __name__ == '__main__':

    paths_to_log_dirs = [
        "../electrostatic/experiments/WormLikeChainGraphDataset_Net/test_log/EdgeConvNodeGATModel_example",
        "../gbsa/experiments/GbsaDgGraphDataset_Net/test_log/EdgeConvNodeGATModel_example",
        "../kd/EdgeConvNodeGAT/experiments/GbsaKdGraphDataset_Net/test_log/EdgeConvNodeGATModel_embeddings_example",
        "../kd/ProteinMPNN/experiments/GbsaKdGraphDataset_Net/test_log/ProteinMPNN_embeddings_example"]

    for path_to_log_dir in paths_to_log_dirs:
        run_parse(path_to_log_dir)
