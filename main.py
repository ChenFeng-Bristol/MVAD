import argparse, os, datetime, glob
import numpy as np
import torch
import torchvision
import pytorch_lightning
from omegaconf import OmegaConf
from data import DataModuleFromConfig

from pytorch_lightning import seed_everything
from pytorch_lightning.trainer import Trainer

from utils import instantiate_from_config


def get_parser(**parser_kwargs):
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument(
        "-n",
        "--name",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="postfix for logdir",
    )
    parser.add_argument(
        "-r",
        "--resume",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="resume from checkpoint",
    )
    parser.add_argument(
        "-b",
        "--base",
        nargs="*",
        metavar="base_config.yaml",
        help="paths to base configs. Loaded from left-to-right. "
             "Parameters can be overwritten or added with command-line options of the form `--key value`.",
        default=list(),
    )
    parser.add_argument(
        "--test_only",
        type=str2bool,
        const=True,
        default=False,
        nargs="?",
        help="perform only test",
    )
    parser.add_argument(
        "--test_name",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="postfix for test dir",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=23,
        help="seed for seed_everything",
    )
    parser.add_argument(
        "-l",
        "--logdir",
        type=str,
        default="logs",
        help="directory for logging",
    )
    parser.add_argument(
        "-d",
        "--debug",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="enable post-mortem debugging",
    )
    return parser


def parse_unknown_args(unknown_args):
    """
    Parse unknown arguments into a nested dictionary.
    Assumes unknown arguments are of the form '--key.subkey=value'.
    """
    def is_float(value):
        try:
            float(value)
            return True
        except ValueError:
            return False
    unknown_args_dict = {}
    for arg in unknown_args:
        # Split argument into nested keys and value based on '=' delimiter
        keys, value = arg.split('=', 1)
        keys = keys.lstrip('-').split('.')  # Split nested keys
        current_level = unknown_args_dict
        # Traverse the nested dictionary and create missing levels if necessary
        for key in keys[:-1]:
            current_level = current_level.setdefault(key, {})
        # Convert numerical values to integers or floats where appropriate
        if value.isdigit():
            value = int(value)
        elif is_float(value):
            value = float(value)
        # Assign the value to the innermost key
        current_level[keys[-1]] = value
    return unknown_args_dict


if __name__ == "__main__":

    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")

    parser = get_parser()

    # opt = parser.parse_args()
    opt, unknown = parser.parse_known_args()

    config_cli = OmegaConf.create(parse_unknown_args(unknown))

    # setup log dirs, and determine ckpt if resume
    if opt.resume:
        if not os.path.exists(opt.resume):
            raise ValueError("Cannot find {}".format(opt.resume))
        if os.path.isfile(opt.resume):
            paths = opt.resume.split("/")
            # idx = len(paths)-paths[::-1].index("logs")+1
            # logdir = "/".join(paths[:idx])
            logdir = "/".join(paths[:-2])
            ckpt = opt.resume
        else:
            assert os.path.isdir(opt.resume), opt.resume
            logdir = opt.resume.rstrip("/")
            ckpt = os.path.join(logdir, "checkpoints", "last.ckpt")

        opt.resume_from_checkpoint = ckpt
        base_configs = sorted(glob.glob(os.path.join(logdir, "configs/*.yaml")))
        opt.base = base_configs + opt.base
        _tmp = logdir.split("/")
        nowname = _tmp[-1]
    else:
        opt.resume_from_checkpoint = None
        if opt.name:
            name = "_" + opt.name
        elif opt.base:
            cfg_fname = os.path.split(opt.base[0])[-1]
            cfg_name = os.path.splitext(cfg_fname)[0]
            name = "_" + cfg_name
        else:
            name = ""
        nowname = now + name
        logdir = os.path.join(opt.logdir, nowname)

    ckptdir = os.path.join(logdir, "checkpoints")
    cfgdir = os.path.join(logdir, "configs")
    seed_everything(opt.seed)

    # init and save configs
    configs = [OmegaConf.load(cfg) for cfg in opt.base] + [config_cli,] # append cli config to overwrite
    config = OmegaConf.merge(*configs)
    lightning_config = config.pop("lightning", OmegaConf.create())
    trainer_config = lightning_config.get("trainer", OmegaConf.create()) # ddp, gpus, nodes, epochs, etc.
    # trainer_opt = argparse.Namespace(**trainer_config)

    if opt.test_only:
        # use a single device if test only
        trainer_config.update({
            'devices': 1,
            'num_nodes': 1,
            'strategy': 'auto'
        })
        # use test batch size of 1 to allow multiple test clips
        config.data.params.update({'batch_size': 1})

    # get world size, will be handy later
    world_size = trainer_config['devices'] * trainer_config['num_nodes']
    if config.model['params']['contrastive_loss_config'] is not None:
        config.model['params']['contrastive_loss_config']['params'].update({'world_size': world_size})

    # model
    model = instantiate_from_config(config.model)

    # start populating trainer params
    trainer_kwargs = dict()

    # set up logger
    os.makedirs(logdir, exist_ok=True) # this is for wandb
    default_logger_cfgs = {
        "wandb": {
            "target": "pytorch_lightning.loggers.WandbLogger",
            "params": {
                "project": 'MVAD',
                "name": nowname,
                "save_dir": logdir,
                "id": glob.glob(os.path.join(logdir, "wandb", 'run-*'))[0].split('-')[-1] if opt.resume else None,
                "offline": opt.debug,
                "resume": "must" if opt.resume else None
            }
        },
        "testtube": {
            "target": "pytorch_lightning.loggers.TensorBoardLogger",
            "params": {
                "name": "testtube",
                "save_dir": logdir,
            }
        },
    }
    default_logger_cfg = default_logger_cfgs["wandb"]
    if "logger" in lightning_config:
        logger_cfg = lightning_config.logger
    else:
        logger_cfg = OmegaConf.create()
    logger_cfg = OmegaConf.merge(default_logger_cfg, logger_cfg) # logger in lightning_config will override default one
    trainer_kwargs["logger"] = instantiate_from_config(logger_cfg)
    
    # set up checkpointing callback and other callbacks
    default_modelckpt_cfg = {
        "target": "pytorch_lightning.callbacks.ModelCheckpoint",
        "params": {
            "dirpath": ckptdir,
            "filename": "{epoch:06}",
            "verbose": True,
            "save_last": True,
        }
    }
    if hasattr(model, "monitor"):
        print(f"Monitoring {model.monitor} as checkpoint metric.")
        default_modelckpt_cfg["params"]["monitor"] = model.monitor
        default_modelckpt_cfg["params"]["save_top_k"] = 3
        default_modelckpt_cfg["params"]["mode"] = 'max' if 'acc' in model.monitor else 'min'

    if "modelcheckpoint" in lightning_config:
        modelckpt_cfg = lightning_config.modelcheckpoint
    else:
        modelckpt_cfg =  OmegaConf.create()
    modelckpt_cfg = OmegaConf.merge(default_modelckpt_cfg, modelckpt_cfg) # ckptcallback in lightning_config will override default one

    default_callbacks_cfg = {
        "checkpoint_callback": modelckpt_cfg,
        "setup_callback": {
            "target": "utils.SetupCallback",
            "params": {
                "resume": opt.resume,
                "now": now,
                "logdir": logdir,
                "ckptdir": ckptdir,
                "cfgdir": cfgdir,
                "config": config,
                "lightning_config": lightning_config,
            }
        },
        "learning_rate_logger": {
            "target": "pytorch_lightning.callbacks.LearningRateMonitor",
            "params": {
                "logging_interval": "step",
                # "log_momentum": True
            }
        },
        "cuda_callback": {
            "target": "utils.CUDACallback"
        },
        "progress_bar": {
            "target": 'pytorch_lightning.callbacks.TQDMProgressBar',
            "params": {
                'refresh_rate': 100,
            }
        },
    }
    if "callbacks" in lightning_config:
        callbacks_cfg = lightning_config.callbacks
    else:
        callbacks_cfg = OmegaConf.create()
    callbacks_cfg = OmegaConf.merge(default_callbacks_cfg, callbacks_cfg) # anything also specified in lightning_config will override default one

    trainer_kwargs["callbacks"] = [instantiate_from_config(callbacks_cfg[k]) for k in callbacks_cfg]

    trainer = Trainer(**trainer_config, **trainer_kwargs)


    # data
    data = instantiate_from_config(config.data)

    # # allow checkpointing via USR1
    # def melk(*args, **kwargs):
    #     # run all checkpoint hooks
    #     if trainer.global_rank == 0:
    #         print("Summoning checkpoint.")
    #         ckpt_path = os.path.join(ckptdir, "last.ckpt")
    #         trainer.save_checkpoint(ckpt_path)


    # def divein(*args, **kwargs):
    #     if trainer.global_rank == 0:
    #         import pudb;
    #         pudb.set_trace()


    # import signal
    # import platform
    # # see https://github.com/rinongal/textual_inversion/issues/44
    # if platform.system() == 'Windows':
    #     os.environ["PL_TORCH_DISTRIBUTED_BACKEND"] = "gloo"
    #     signal.signal(signal.SIGTERM, melk)
    #     signal.signal(signal.SIGTERM, divein)
    # else:
    #     signal.signal(signal.SIGUSR1, melk)
    #     signal.signal(signal.SIGUSR2, divein)

    # run
    if opt.test_only:
        assert opt.resume_from_checkpoint is not None, 'Must specify resume dir or ckpt path using -r if test only!'
        setattr(model, 'test_name', opt.test_name)

        # set test fragment size
        test_sampler_config = config.data.params.test.params.sampler_config.params
        test_sampling_config = config.data.params.test.params.sampling_config
        test_t = test_sampler_config['fsize_t'] * test_sampler_config['fragments_t']
        test_h = test_sampling_config['fsize_h'] * test_sampling_config['fragments_h']
        test_w = test_sampling_config['fsize_w'] * test_sampling_config['fragments_w']
        setattr(model.model.feat_extractor, 'base_x_size', (test_t, test_h, test_w))
        
        print(OmegaConf.to_yaml(config))

        trainer.test(model, datamodule=data, ckpt_path=opt.resume_from_checkpoint)
    else:
        trainer.fit(model, datamodule=data, ckpt_path=opt.resume_from_checkpoint)