"""by lyuwenyu
"""

import os 
import sys 
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import argparse

import src.misc.dist as dist 
from src.core import YAMLConfig 
from src.solver import TASKS


def main(args, ) -> None:
    '''main
    '''
    dist.init_distributed()

    assert not all([args.tuning, args.resume]), \
        'Only support from_scrach or resume or tuning at one time'

    cfg = YAMLConfig(
        args.config,
        resume=args.resume, 
        use_amp=args.amp,
        tuning=args.tuning, 
        use_wb=args.wb
    )

    # init weight and bias
    is_wb = False
    if args.wb: 
        import wandb
        wandb.init(
            # set the wandb project where this run will be logged
            project="rt-detr-training",

            # track hyperparameters and run metadata
            config={
                "learning_rate": str(cfg.lr_scheduler),
                "architecture": str(cfg.model),
                "dataset": str(cfg.train_dataset),
                "epochs": cfg.epoches,
            }
        )
        is_wb = True

    solver = TASKS[cfg.yaml_cfg['task']](cfg)
    
    if args.test_only:
        solver.val()
    else:
        solver.fit()
    
    if is_wb:
        wandb.finish()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, )
    parser.add_argument('--resume', '-r', type=str, )
    parser.add_argument('--tuning', '-t', type=str, )
    parser.add_argument('--test-only', action='store_true', default=False,)
    parser.add_argument('--amp', action='store_true', default=False,)
    parser.add_argument('--wb', action='store_true', default=False)

    args = parser.parse_args()

    main(args)
