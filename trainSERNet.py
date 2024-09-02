#!/usr/bin/python
# -*- coding: utf-8 -*-

import argparse
import datetime
import socket
import warnings
import zipfile
from contextlib import closing
from pathlib import Path
from shutil import rmtree

import torch.multiprocessing as mp

import yaml
from DatasetLoader import *
from ser_dataset_loader import *
from SERNet import *
from tuneThreshold import *

## ===== ===== ===== ===== ===== ===== ===== =====
## Parse arguments
## ===== ===== ===== ===== ===== ===== ===== =====
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

parser = argparse.ArgumentParser(description="SERNet")

parser.add_argument("--config", type=str, default=None, help="Config YAML file")

## Data loader
parser.add_argument(
    "--max_frames",
    type=int,
    default=200,
    help="Input length to the network for training",
)
parser.add_argument(
    "--eval_frames",
    type=int,
    default=300,
    help="Input length to the network for testing; 0 uses the whole files",
)
parser.add_argument(
    "--batch_size",
    type=int,
    default=32,
    help="Batch size, number of speakers per batch",
)
parser.add_argument(
    "--batch_length",
    type=int,
    default=80000,  # # 16000 * 5
    help="Sample length in sample size for train samples",
)
parser.add_argument(
    "--val_batch_length",
    type=int,
    default=176000,
    help="Sample length in sample size for validation samples",
)
parser.add_argument(
    "--max_seg_per_spk",
    type=int,
    default=500,
    help="Maximum number of utterances per speaker per epoch",
)
parser.add_argument(
    "--nDataLoaderThread", type=int, default=10, help="Number of loader threads"
)
parser.add_argument("--augment", type=bool, default=True, help="Augment input")
parser.add_argument(
    "--seed", type=int, default=100, help="Seed for the random number generator"
)

## Training details
parser.add_argument(
    "--test_interval",
    type=int,
    default=1,
    help="Test and save every [test_interval] epochs",
)
parser.add_argument(
    "--max_epoch", type=int, default=50, help="Maximum number of epochs"
)
parser.add_argument("--trainfunc", type=str, default="aamsoftmax", help="Loss function")

## Optimizer
parser.add_argument("--optimizer", type=str, default="adamw", help="sgd or adam")
parser.add_argument(
    "--scheduler", type=str, default="steplr", help="Learning rate scheduler"
)
parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
parser.add_argument(
    "--lr_decay",
    type=float,
    default=0.9,
    help="Learning rate decay every [test_interval] epochs",
)

## Pre-trained Transformer Model
parser.add_argument(
    "--pretrained_model_path",
    type=str,
    default="None",
    help="Absolute path to the pre-trained model",
)
parser.add_argument(
    "--weight_finetuning_reg",
    type=float,
    default=0.001,
    help="L2 regularization towards the initial pre-trained model",
)
parser.add_argument(
    "--LLRD_factor",
    type=float,
    default=1.0,
    help="Layer-wise Learning Rate Decay (LLRD) factor",
)
parser.add_argument(
    "--LR_Transformer",
    type=float,
    default=2e-5,
    help="Learning rate of pre-trained model",
)
parser.add_argument(
    "--LR_MHFA",
    type=float,
    default=5e-3,
    help="Learning rate of back-end attentive pooling model",
)

## Loss functions
parser.add_argument(
    "--hard_prob",
    type=float,
    default=0.5,
    help="Hard negative mining probability, otherwise random, only for some loss functions",
)
parser.add_argument(
    "--hard_rank",
    type=int,
    default=10,
    help="Hard negative mining rank in the batch, only for some loss functions",
)
parser.add_argument(
    "--margin",
    type=float,
    default=0.2,
    help="Loss margin, only for some loss functions",
)
parser.add_argument(
    "--scale", type=float, default=30, help="Loss scale, only for some loss functions"
)

parser.add_argument(
    "--nClasses",
    type=int,
    default=4,
    help="Number of speakers in the softmax layer, only for softmax-based losses",
)

## Evaluation parameters
parser.add_argument(
    "--dcf_p_target",
    type=float,
    default=0.05,
    help="A priori probability of the specified target speaker",
)
parser.add_argument(
    "--dcf_c_miss", type=float, default=1, help="Cost of a missed detection"
)
parser.add_argument(
    "--dcf_c_fa", type=float, default=1, help="Cost of a spurious detection"
)

## Load and save
parser.add_argument(
    "--initial_model", type=str, default="", help="Initial model weights"
)
parser.add_argument(
    "--save_path",
    type=str,
    default="/app/data2/mhfa_training_logs",
    help="Path for model and logs",
)

## Training and test data

# SER
parser.add_argument(
    "--train_path",
    type=str,
    default="",
    help="Path to the root dir with torch tensors.",
)
parser.add_argument(
    "--train_list",
    type=str,
    default="",
    help="Path to csv file with labels.",
)
parser.add_argument(
    "--eval_session",
    type=str,
    default="Ses05",
    help="Session that will not be used in training.",
)
parser.add_argument(
    "--test_gender",
    type=str,
    default="M",
    help="Speaker of the eval session that will be used for testing, other for evaluation (male - M or female - F)",
)

parser.add_argument(
    "--ds_size",
    default=None,
    help="Size of the dataset used for training and evaluation",
)
parser.add_argument(
    "--head_nb",
    type=int,
    default=16,
    help="Number of heads in MHFA",
)

## Model definition
parser.add_argument("--n_mels", type=int, default=80, help="Number of mel filterbanks")
parser.add_argument(
    "--num_workers", type=int, default=1, help="Number of workers for dataset"
)
parser.add_argument("--log_input", type=bool, default=False, help="Log input features")
parser.add_argument(
    "--save_ckpts", type=bool, default=False, help="if delete ckpts after test"
)
parser.add_argument("--model", type=str, default="", help="Name of model definition")
parser.add_argument("--encoder_type", type=str, default="SAP", help="Type of encoder")
parser.add_argument(
    "--classes", type=list, default=[1, 2, 3, 4], help="Sequence of classes"
)
parser.add_argument("--random_crop", type=bool)

## For test only
parser.add_argument("--eval", dest="eval", action="store_true", help="Eval only")

## Distributed and mixed precision training
parser.add_argument(
    "--port",
    type=str,
    default="7889",
    help="Port for distributed training, input as text",
)
parser.add_argument(
    "--distributed",
    dest="distributed",
    action="store_true",
    help="Enable distributed training",
)
parser.add_argument(
    "--mixedprec",
    dest="mixedprec",
    action="store_true",
    help="Enable mixed precision training",
)

args = parser.parse_args()


## Parse YAML
def find_option_type(key, parser):
    for opt in parser._get_optional_actions():
        if ("--" + key) in opt.option_strings:
            return opt.type
    raise ValueError


if args.config is not None:
    with open(args.config, "r") as f:
        yml_config = yaml.load(f, Loader=yaml.FullLoader)
    for k, v in yml_config.items():
        if k in args.__dict__:
            typ = find_option_type(k, parser)
            if typ is not None:
                args.__dict__[k] = typ(v)
        else:
            sys.stderr.write("Ignored unknown parameter {} in yaml.\n".format(k))

## Try to import NSML
try:
    import nsml
    from nsml import (DATASET_PATH, HAS_DATASET, MY_RANK, NSML_NFS_OUTPUT,
                      PARALLEL_PORTS, PARALLEL_WORLD, SESSION_NAME)
except:
    pass

warnings.simplefilter("ignore")


## ===== ===== ===== ===== ===== ===== ===== =====
## Trainer script
## ===== ===== ===== ===== ===== ===== ===== =====


def main_worker(gpu, ngpus_per_node, dist_url, args):
    args.gpu = gpu

    ## Load models
    s = SERNet(**vars(args))

    if args.distributed:
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = args.port
        dist.init_process_group(
            backend="nccl",
            init_method=dist_url,
            world_size=ngpus_per_node,
            rank=args.gpu,
        )
        torch.cuda.set_device(args.gpu)
        s.cuda(args.gpu)
        s = torch.nn.parallel.DistributedDataParallel(
            s, device_ids=[args.gpu], find_unused_parameters=True
        )
        print("Loaded the model on GPU {:d}".format(args.gpu))
    else:
        s = WrappedModel(s).cuda(args.gpu)

    it = 1
    best_balanced_accuracy = 0.0
    best_ckpt_path = None

    if args.gpu == 0:
        ## Write args to scorefile
        scorefile = open(args.result_save_path + "/scores.txt", "a+")

    ## Initialise trainer and data loader
    dataloader_factory = DataloaderFactory(args)
    train_loader = dataloader_factory.build(
        size=args.ds_size, state="train", bs=args.batch_size // ngpus_per_node
    )
    val_loader = dataloader_factory.build(
        size=args.ds_size, state="val", bs=args.batch_size // ngpus_per_node
    )
    test_loader = dataloader_factory.build(
        size=args.ds_size, state="test", bs=args.batch_size // ngpus_per_node
    )

    print(f"Train loader: {len(train_loader)}")
    print(f"Val loader: {len(val_loader)}")
    print(f"Test loader: {len(test_loader)}")

    print(f"Initialized dataloaders")
    # trainLoader = get_data_loader(args.train_list, **vars(args));
    trainer = ModelTrainer(s, **vars(args))

    ## Load model weights
    modelfiles = glob.glob("%s/model0*.model" % args.model_save_path)
    modelfiles.sort()

    if args.initial_model != "":
        trainer.loadParameters(args.initial_model)
        print("Model {} loaded!".format(args.initial_model))
    elif len(modelfiles) >= 1:
        print("Model {} loaded from previous state!".format(modelfiles[-1]))
        trainer.loadParameters(modelfiles[-1])
        it = int(os.path.splitext(os.path.basename(modelfiles[-1]))[0][5:]) + 1

    for ii in range(1, it):
        trainer.__scheduler__.step()

    pytorch_total_params = sum(p.numel() for p in s.module.__S__.parameters())

    print("Total parameters: ", pytorch_total_params)
    ## Evaluation code - must run on single GPU
    # if args.eval == True:
    #
    #     print("Test list", args.test_list)
    #
    #     sc, lab, _, sc1, sc2 = trainer.evaluateFromList(**vars(args))
    #
    #     if args.gpu == 0:
    #
    #         result = tuneThresholdfromScore(sc, lab, [1, 0.1])
    #         result_s1 = tuneThresholdfromScore(sc1, lab, [1, 0.1])
    #         result_s2 = tuneThresholdfromScore(sc2, lab, [1, 0.1])
    #
    #         fnrs, fprs, thresholds = ComputeErrorRates(sc, lab)
    #         mindcf, threshold = ComputeMinDcf(
    #             fnrs,
    #             fprs,
    #             thresholds,
    #             args.dcf_p_target,
    #             args.dcf_c_miss,
    #             args.dcf_c_fa,
    #         )
    #
    #         print(
    #             "\n",
    #             time.strftime("%Y-%m-%d %H:%M:%S"),
    #             "VEER {:2.4f}".format(result[1]),
    #             "VEER_s1 {:2.4f}".format(result_s1[1]),
    #             "VEER_s2 {:2.4f}".format(result_s2[1]),
    #             "MinDCF {:2.5f}".format(mindcf),
    #         )
    #
    #         if ("nsml" in sys.modules) and args.gpu == 0:
    #             training_report = {}
    #             training_report["summary"] = True
    #             training_report["epoch"] = it
    #             training_report["step"] = it
    #             training_report["val_eer"] = result[1]
    #             training_report["val_dcf"] = mindcf
    #
    #             nsml.report(**training_report)
    #
    #     return

    ## Save training code and params
    if args.gpu == 0:
        pyfiles = glob.glob("./*.py")
        strtime = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

        zipf = zipfile.ZipFile(
            args.result_save_path + "/run%s.zip" % strtime, "w", zipfile.ZIP_DEFLATED
        )
        for file in pyfiles:
            zipf.write(file)
        zipf.close()

        with open(args.result_save_path + "/run%s.cmd" % strtime, "w") as f:
            f.write("%s" % args)

    ## Core training script
    for it in range(it, args.max_epoch + 1):
        if args.distributed:
            train_loader.set_epoch(it)

        clr = [x["lr"] for x in trainer.__optimizer__.param_groups]

        loss = trainer.train_network(train_loader, verbose=(args.gpu == 0))

        if args.gpu == 0:
            (
                balanced_accuracy,
                accuracy,
                f1_weighted,
                f1_macro,
                precision_macro,
                precision_weighted,
            ) = trainer.evaluate(val_loader, **vars(args))

            print(
                "\n",
                time.strftime("%Y-%m-%d %H:%M:%S"),
                "Epoch {:d}, TLOSS {:f}, LR {:f}, balanced_accuracy {:2.3f} "
                "accuracy {:2.3f} f1_weighted {:2.3f} f1_macro {:2.3f} "
                "precision_macro {:2.3f} precision_weighted {:2.3f}".format(
                    it,
                    loss,
                    max(clr),
                    balanced_accuracy,
                    accuracy,
                    f1_weighted,
                    f1_macro,
                    precision_macro,
                    precision_weighted,
                ),
            )
            scorefile.write(
                "Epoch {:d}, TLOSS {:f}, LR {:f} balanced_accuracy {:2.3f} "
                "accuracy {:2.3f} f1_weighted {:2.3f} f1_macro {:2.3f} "
                "precision_macro {:2.3f} precision_weighted {:2.3f}".format(
                    it,
                    loss,
                    max(clr),
                    balanced_accuracy,
                    accuracy,
                    f1_weighted,
                    f1_macro,
                    precision_macro,
                    precision_weighted,
                )
            )

            if balanced_accuracy > best_balanced_accuracy:
                best_balanced_accuracy = balanced_accuracy
                best_ckpt_path = args.model_save_path + "/model%09d.model" % it

            trainer.saveParameters(args.model_save_path + "/model%09d.model" % it)

            scorefile.flush()

        if ("nsml" in sys.modules) and args.gpu == 0:
            training_report = {
                "summary": True,
                "epoch": it,
                "step": it,
                "train_loss": loss,
            }
            nsml.report(**training_report)

    # Testing and closing file
    if args.gpu == 0:
        trainer.loadParameters(best_ckpt_path)
        (
            balanced_accuracy,
            accuracy,
            f1_weighted,
            f1_macro,
            precision_macro,
            precision_weighted,
        ) = trainer.evaluate(test_loader, **vars(args))

        print(
            "\n",
            time.strftime("%Y-%m-%d %H:%M:%S"),
            "Ckpt path {:s}, test_balanced_accuracy {:2.3f} "
            "test_accuracy {:2.3f} test_f1_weighted {:2.3f} test_f1_macro {:2.3f} "
            "test_precision_macro {:2.3f} test_precision_weighted {:2.3f}".format(
                best_ckpt_path,
                balanced_accuracy,
                accuracy,
                f1_weighted,
                f1_macro,
                precision_macro,
                precision_weighted,
            ),
        )
        scorefile.write(
            "Ckpt path {:s}, test_balanced_accuracy {:2.3f} "
            "test_accuracy {:2.3f} test_f1_weighted {:2.3f} test_f1_macro {:2.3f} "
            "test_precision_macro {:2.3f} test_precision_weighted {:2.3f}\n".format(
                best_ckpt_path,
                balanced_accuracy,
                accuracy,
                f1_weighted,
                f1_macro,
                precision_macro,
                precision_weighted,
            )
        )

        scorefile.close()
        if not args.save_ckpts:
            rmtree(args.model_save_path)


## ===== ===== ===== ===== ===== ===== ===== =====
## Main function
## ===== ===== ===== ===== ===== ===== ===== =====


def find_free_port() -> int:
    """
    Find a free port for dist url
    :return:
    """
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("", 0))
        port = s.getsockname()[1]

    return port


def main():
    if ("nsml" in sys.modules) and not args.eval:
        args.save_path = os.path.join(args.save_path, SESSION_NAME.replace("/", "_"))
    args.save_path = args.save_path + (
        f"weight_finetuning_reg_{args.weight_finetuning_reg}_"
        f"LLRD_factor_{args.LLRD_factor}_"
        f"LR_Transformer_{args.LR_Transformer}_"
        f"LR_MHFA_{args.LR_MHFA}_"
        f"head_nb_{args.head_nb}_"
        f"batch_size_{args.batch_size}_"
        f"eval_session_{args.eval_session}_"
        f"test_gender_{args.test_gender}"
    )

    if not Path(str(args.save_path)).exists():
        args.model_save_path = args.save_path + "/model"
        args.result_save_path = args.save_path + "/result"
        args.feat_save_path = ""

        os.makedirs(args.model_save_path, exist_ok=True)
        os.makedirs(args.result_save_path, exist_ok=True)

        n_gpus = torch.cuda.device_count()

        print("Python Version:", sys.version)
        print("PyTorch Version:", torch.__version__)
        print("Number of GPUs:", torch.cuda.device_count())
        print("Save path:", args.save_path)

        if args.distributed:
            print(f"Using distributed training.")

            free_port = find_free_port()
            dist_url = f"tcp://127.0.0.1:{free_port}"

            # Create processes
            processes = []
            for rank in range(torch.cuda.device_count()):
                p = mp.Process(target=main_worker, args=(rank, n_gpus, dist_url, args))
                p.start()
                processes.append(p)

            # Join processes
            for p in processes:
                p.join()
            # mp.spawn(fn=main_worker, nprocs=n_gpus, args=(n_gpus, args))
        else:
            main_worker(0, 1, None, args)
    else:
        print(f"{args.save_path} exist!")


if __name__ == "__main__":
    main()
