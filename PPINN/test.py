from utils.setup import SetupRun, SetupLogger
from utils.util import read_json
import dataset as module_dataset
import utils.metric as module_metric
import model.model as module_arch
from utils.visualization import TensorboardWriter
from utils.util import MetricTracker

from torch_geometric.loader import DataLoader
import torch
import torch.nn.functional as F
from tqdm import tqdm
import os


def run_testing(run_setup: SetupRun,
                logger_setup: SetupLogger,
                path_to_the_best_checkpoint: str,
                device: str):
    # setup logger
    logger = logger_setup("test")

    # setup dataset
    dataset = run_setup.init_obj('dataset_test', module_dataset, transform=None)

    logger.info(dataset.root)

    # setup data_loader instances
    test_loader = DataLoader(dataset=dataset, batch_size=1)

    # setup model architecture, then print to console
    model = run_setup.init_obj('arch', module_arch)
    logger.info(model)

    # loading the best checkpoint from training
    model.load_state_dict(torch.load(path_to_the_best_checkpoint, map_location=torch.device('cpu'))['state_dict'])

    # setup function handles of metrics
    criterion = run_setup.init_funct("loss", F)
    metrics = [run_setup.init_funct(metric, module_metric) for metric in run_setup['metrics']]

    # run testing process with saving metrics in logs
    model.eval()
    cfg_trainer = run_setup['trainer']['args']
    writer = TensorboardWriter(logger_setup["log_dir"], logger_setup, cfg_trainer['tensorboard'])
    test_metrics = MetricTracker('loss', 'output', *[m.__name__ for m in metrics], writer=writer)

    message = ["number", "predicted", "target", "loss"]
    for met in metrics:
        message.append(met.__name__)

    logger.info(" - ".join(message))

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(tqdm(test_loader)):

            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)

            writer.set_step(batch_idx, mode="test")
            test_metrics.update('loss', loss.item())
            test_metrics.update('output', output)

            message = [batch_idx + 1, output.item(), target.item(), loss.item()]

            for met in metrics:
                test_metrics.update(met.__name__, met(output, target))
                message.append(met(output, target))

            logger.info(" - ".join(str(x) for x in message))

    return test_metrics.result()


if __name__ == '__main__':
    paths_to_experiments = ["../electrostatic/experiments/WormLikeChainGraphDataset_Net/checkpoint/EdgeConvNodeGATModel_example",
                            "../gbsa/experiments/GbsaDgGraphDataset_Net/checkpoint/EdgeConvNodeGATModel_example",
                            "../kd/EdgeConvNodeGAT/experiments/GbsaKdGraphDataset_Net/checkpoint/EdgeConvNodeGATModel_embeddings_example",
                            "../kd/ProteinMPNN/experiments/GbsaKdGraphDataset_Net/checkpoint/ProteinMPNN_embeddings_example"]

    log_dirs = [
        "../electrostatic/experiments/WormLikeChainGraphDataset_Net/test_log/EdgeConvNodeGATModel_example",
        "../gbsa/experiments/GbsaDgGraphDataset_Net/test_log/EdgeConvNodeGATModel_example",
        "../kd/EdgeConvNodeGAT/experiments/GbsaKdGraphDataset_Net/test_log/EdgeConvNodeGATModel_embeddings_example",
        "../kd/ProteinMPNN/experiments/GbsaKdGraphDataset_Net/test_log/ProteinMPNN_embeddings_example"]

    device = "cpu"
    path_to_log_config = "logger_config.json"

    for path_to_experiment, log_dir in zip(paths_to_experiments, log_dirs):
        config = read_json(os.path.join(path_to_experiment, "config.json"))

        log_config = read_json(path_to_log_config)

        run_setup = SetupRun(config=config, checkpoint_dir=path_to_experiment)

        log_setup = SetupLogger(config=log_config, log_dir=log_dir)

        if os.path.exists(os.path.join(path_to_experiment, "model_best.pth")):
            run_testing(run_setup=run_setup,
                        logger_setup=log_setup,
                        path_to_the_best_checkpoint=os.path.join(path_to_experiment, "model_best.pth"),
                        device=device,
                        )
