#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import json
import os
import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import default_argument_parser, launch

from config import add_spegc_config
from engine.trainer import SPEGCTrainer, BaselineTrainer

# hacky way to register
from modeling.meta_arch.rcnn import DAobjTwoStagePseudoLabGeneralizedRCNN
from modeling.meta_arch.vgg import build_vgg_backbone  # noqa
from modeling.proposal_generator.rpn import PseudoLabRPN
from modeling.roi_heads.roi_heads import StandardROIHeadsPseudoLab
from modeling.meta_arch.ts_ensemble import EnsembleTSModel
import data.datasets.builtin

def custom_setup(cfg, args):
    """
    Perform some basic setups, suppressing environment printouts, config exports, and verbose warning logs.
    """
    import os
    import logging
    from detectron2.utils import comm
    from detectron2.utils.env import seed_all_rng
    from detectron2.utils.logger import setup_logger

    output_dir = cfg.OUTPUT_DIR
    if comm.is_main_process() and output_dir:
        os.makedirs(output_dir, exist_ok=True)

    rank = comm.get_rank()
    # Initialize log handlers without file writing to prevent creating empty log.txt files
    setup_logger(None, distributed_rank=rank, name="fvcore")
    setup_logger(None, distributed_rank=rank, name="detectron2")

    # Mute all INFO/WARNING logs from detectron2, fvcore, and fvcore.common.checkpoint
    logging.getLogger("detectron2").setLevel(logging.WARNING)
    logging.getLogger("fvcore").setLevel(logging.WARNING)
    logging.getLogger("fvcore.common.checkpoint").setLevel(logging.ERROR)

    seed = cfg.SEED if hasattr(cfg, "SEED") else -1
    seed_all_rng(None if seed < 0 else seed + rank)


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_spegc_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    # Convert paths to absolute paths to prevent resolution issues in detectron2/fvcore checkpointer
    if cfg.MODEL.WEIGHTS and not cfg.MODEL.WEIGHTS.startswith("detectron2://"):
        cfg.MODEL.WEIGHTS = os.path.abspath(cfg.MODEL.WEIGHTS)
    if cfg.OUTPUT_DIR:
        cfg.OUTPUT_DIR = os.path.abspath(cfg.OUTPUT_DIR)

    cfg.freeze()

    custom_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)
    if cfg.SEMISUPNET.Trainer == "spegc":
        Trainer = SPEGCTrainer
    elif cfg.SEMISUPNET.Trainer == "baseline":
        Trainer = BaselineTrainer
    else:
        raise ValueError("Trainer Name is not found.")

    if args.eval_only:
        if cfg.SEMISUPNET.Trainer == "spegc":
            model = Trainer.build_model(cfg)
            model_teacher = Trainer.build_model(cfg)
            ensem_ts_model = EnsembleTSModel(model_teacher, model)

            DetectionCheckpointer(
                ensem_ts_model, save_dir=cfg.OUTPUT_DIR
            ).resume_or_load(cfg.MODEL.WEIGHTS, resume=args.resume)
            if cfg.TEST.EVAL_STU:
                res = Trainer.test(cfg, ensem_ts_model.modelStudent)
            else:
                res = Trainer.test(cfg, ensem_ts_model.modelTeacher)
            
        else:
            model = Trainer.build_model(cfg)
            # total_params = sum(p.numel() for p in model.parameters())
            # print(f"Total Parameters: {total_params}")
            DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
                cfg.MODEL.WEIGHTS, resume=args.resume
            )
            res = Trainer.test(cfg, model, Trainer.build_optimizer(cfg, model))

        # Pretty format evaluation results
        pretty_results = []
        pretty_results.append("=" * 90)
        pretty_results.append("Evaluation Results")
        pretty_results.append(f"Model Weights: {cfg.MODEL.WEIGHTS}")
        pretty_results.append("=" * 90)
        pretty_results.append(f"{'Dataset / Split':<35} | {'Dice Coeff':<12} | {'Enhanced Align':<16} | {'Struct Similarity':<18}")
        pretty_results.append("-" * 90)
        for dataset_name, metrics in res.items():
            if isinstance(metrics, dict):
                dice = metrics.get('Dice Coefficient', 0.0)
                ea = metrics.get('Enhanced Alignment Metric', 0.0)
                sm = metrics.get('Structural Similarity Metric', 0.0)
                pretty_results.append(f"{dataset_name:<35} | {dice:>10.4f}%  | {ea:>14.4f}%  | {sm:>16.4f}%")
        pretty_results.append("=" * 90)
        pretty_results.append("\n")
        formatted_text = "\n".join(pretty_results)

        print("\n" + formatted_text)
        
        # Save formatted results to result.txt
        with open(os.path.join(cfg.OUTPUT_DIR, 'result.txt'), 'a') as file:
            file.write(formatted_text)
        
        return res

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)

    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    args.resume = True
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
