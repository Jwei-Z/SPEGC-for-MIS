# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# import sys
# import os
# project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
# sys.path.append(project_root)
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from detectron2.modeling.meta_arch.rcnn import GeneralizedRCNN
from detectron2.config import configurable
# from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
# from detectron2.modeling.meta_arch.rcnn import GeneralizedRCNN
import logging
from typing import Dict, Tuple, List, Optional
from collections import OrderedDict
from detectron2.modeling.proposal_generator import build_proposal_generator
from detectron2.modeling.backbone import build_backbone, Backbone
from detectron2.modeling.roi_heads import build_roi_heads
from detectron2.utils.events import get_event_storage
from detectron2.structures import ImageList

from modeling.GModule.build_graph import PrototypeComputation
from modeling.GModule.spegc import SPEGC

############### Image discriminator ##############
class FCDiscriminator_img(nn.Module):
    def __init__(self, num_classes, ndf1=256, ndf2=128):
        super(FCDiscriminator_img, self).__init__()

        self.conv1 = nn.Conv2d(num_classes, ndf1, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(ndf1, ndf2, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(ndf2, ndf2, kernel_size=3, padding=1)
        self.classifier = nn.Conv2d(ndf2, 1, kernel_size=3, padding=1)

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.leaky_relu(x)
        x = self.conv2(x)
        x = self.leaky_relu(x)
        x = self.conv3(x)
        x = self.leaky_relu(x)
        x = self.classifier(x)
        return x
#################################

################ Gradient reverse function
class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg()

def grad_reverse(x):
    return GradReverse.apply(x)

#######################

class TTTGraphPool:
    """TTT, MCP"""
    def __init__(self, pool_size, min_pool_size):
        self.pool_size = pool_size
        self.min_pool_size = min_pool_size
        self.graph_pool = []  # (nodes, labels)

    def update_pool(self, nodes, labels):
        """FIFO"""
        if len(self.graph_pool) >= self.pool_size:
            self.graph_pool.pop(0)
        # detach(),
        self.graph_pool.append((nodes.detach().clone(), labels.detach().clone()))

    def create_pseudo_batch(self, current_nodes, current_labels):
        pass
        if len(self.graph_pool) == 0:
            # :
            return [current_nodes], [current_labels]

        # :  +
        # detach,
        pool_nodes = [item[0] for item in self.graph_pool]
        pool_labels = [item[1] for item in self.graph_pool]

        nodes_batch = [current_nodes] + pool_nodes
        labels_batch = [current_labels] + pool_labels

        return nodes_batch, labels_batch

    def should_do_matching(self):
        pass
        return len(self.graph_pool) >= self.min_pool_size

    def get_pool_size(self):
        pass
        return len(self.graph_pool)

@META_ARCH_REGISTRY.register()
class DAobjTwoStagePseudoLabGeneralizedRCNN(GeneralizedRCNN):
    """
    DARCNN
    MCP (Multi-graph Consistency Prompting) 

    MCPUPCA: 
    1. V_signal(32, )
    2. V_noise(32, )
    3. μ_U, MCP: 
       - K_C = V_signal^T × μ_U()
       - K_IC = V_noise^T × μ_U()
    """

    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        proposal_generator: nn.Module,
        roi_heads: nn.Module,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        input_format: Optional[str] = None,
        vis_period: int = 0,
        dis_type: str,
        num_centroids: int = 32,
        sample_dist: int = 10,
        ttt_pool_size: int = 3,
        ttt_min_pool_size: int = 1,
        spegc_enable: bool = True,
        spegc_z: int = 48,
        spegc_m: int = 8,
        spegc_t: int = 4,
        spegc_lambda: float = 0.2,
        spegc_p: float = 0.5,
        ttt_on: bool = False,
    ):
        """
        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            proposal_generator: a module that generates proposals using backbone features
            roi_heads: a ROI head that performs per-region computation
            pixel_mean, pixel_std: list or tuple with #channels element, representing
                the per-channel mean and std to be used to normalize the input image
            input_format: describe the meaning of channels of input. Needed by visualization
            vis_period: the period to run visualization. Set to 0 to disable.
        """
        super(GeneralizedRCNN, self).__init__()
        self.backbone = backbone
        self.proposal_generator = proposal_generator
        self.roi_heads = roi_heads

        self.input_format = input_format
        self.vis_period = vis_period
        if vis_period > 0:
            assert input_format is not None, "input_format is required for visualization!"

        self.register_buffer("pixel_mean", torch.tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.tensor(pixel_std).view(-1, 1, 1), False)
        assert (
            self.pixel_mean.shape == self.pixel_std.shape
        ), f"{self.pixel_mean} and {self.pixel_std} have different shapes!"

        self.dis_type = dis_type

        self.D_img = FCDiscriminator_img(self.backbone._out_feature_channels[self.dis_type])

        # Initialize centroids
        self.graph_generator = PrototypeComputation(self.roi_heads.num_classes, sample_dist)
        self.centroids = nn.Parameter(torch.randn(num_centroids, 256) + 1 / num_centroids)

        # Initialize TTT graph pool for pseudo-batch processing
        self.ttt_graph_pool = TTTGraphPool(ttt_pool_size, ttt_min_pool_size)

        # SPEGC initialization
        self.spegc_enable = spegc_enable
        if self.spegc_enable:
            self.spegc_z = spegc_z
            self.spegc_m = spegc_m
            self.spegc_t = spegc_t
            self.spegc_lambda = spegc_lambda
            self.spegc_p = spegc_p
            self.spegc_module = SPEGC(dim=256, Z=spegc_z, M=spegc_m, lambda_c=spegc_lambda)

    def build_discriminator(self):
        self.D_img = FCDiscriminator_img([self.dis_type]).to(self.device)

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        return {
            "backbone": backbone,
            "proposal_generator": build_proposal_generator(cfg, backbone.output_shape()),
            "roi_heads": build_roi_heads(cfg, backbone.output_shape()),
            "input_format": cfg.INPUT.FORMAT,
            "vis_period": cfg.VIS_PERIOD,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            "dis_type": cfg.SEMISUPNET.DIS_TYPE,
            "num_centroids": cfg.SEMISUPNET.NUM_CENTROIDS,
            "sample_dist": cfg.SEMISUPNET.SAMPLE_DIST,
            "ttt_pool_size": cfg.SEMISUPNET.TTT_POOL_SIZE,
            "ttt_min_pool_size": cfg.SEMISUPNET.TTT_MIN_POOL_SIZE,
            "spegc_enable": cfg.SEMISUPNET.SPEGC_ENABLE,
            "spegc_z": cfg.SEMISUPNET.SPEGC_Z,
            "spegc_m": cfg.SEMISUPNET.SPEGC_M,
            "spegc_t": cfg.SEMISUPNET.SPEGC_T,
            "spegc_lambda": cfg.SEMISUPNET.SPEGC_LAMBDA,
            "spegc_p": cfg.SEMISUPNET.SPEGC_P,
            "ttt_on": cfg.TEST.TTT,
        }



    def preprocess_image_train(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        """
        Normalize, pad and batch the input images.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.backbone._out_feature_channelself.backbone.size_divisibility)

        images_t = [x["image_unlabeled"].to(self.device) for x in batched_inputs]
        images_t = [(x - self.pixel_mean) / self.pixel_std for x in images_t]
        images_t = ImageList.from_tensors(images_t, self.backbone.size_divisibility)

        return images, images_t

    def forward(
        self, batched_inputs, branch=None, given_proposals=None, val_mode=False
    ):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances (optional): groundtruth :class:`Instances`
                * proposals (optional): :class:`Instances`, precomputed proposals.

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.

        Returns:
            list[dict]:
                Each dict is the output for one input image.
                The dict contains one key "instances" whose value is a :class:`Instances`.
                The :class:`Instances` object has the following keys:
                "pred_boxes", "pred_classes", "scores", "pred_masks", "pred_keypoints"
        """
        if self.D_img == None:
            self.build_discriminator()
        if (not self.training) and (not val_mode):  # only conduct when testing mode
            return self.inference(batched_inputs)

        source_label = 0
        target_label = 1

        if branch == "domain":
            # self.D_img.train()
            # source_label = 0
            # target_label = 1
            # images = self.preprocess_image(batched_inputs)
            images_s, images_t = self.preprocess_image_train(batched_inputs)

            features = self.backbone(images_s.tensor)

            # import pdb
            # pdb.set_trace()
           
            features_s = grad_reverse(features[self.dis_type])
            D_img_out_s = self.D_img(features_s)
            loss_D_img_s = F.binary_cross_entropy_with_logits(D_img_out_s, torch.FloatTensor(D_img_out_s.data.size()).fill_(source_label).to(self.device))

            features_t = self.backbone(images_t.tensor)
            
            features_t = grad_reverse(features_t[self.dis_type])
            # features_t = grad_reverse(features_t['p2'])
            D_img_out_t = self.D_img(features_t)
            loss_D_img_t = F.binary_cross_entropy_with_logits(D_img_out_t, torch.FloatTensor(D_img_out_t.data.size()).fill_(target_label).to(self.device))

            # import pdb
            # pdb.set_trace()

            losses = {}
            losses["loss_D_img_s"] = loss_D_img_s
            losses["loss_D_img_t"] = loss_D_img_t
            return losses, [], [], None, None

        # self.D_img.eval()
        images = self.preprocess_image(batched_inputs)

        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None

        features = self.backbone(images.tensor)
        if branch == "TTT":
            features = {k: v.detach() for k, v in features.items()}

        # TODO: remove the usage of if else here. This needs to be re-organized
        if branch.startswith("supervised_source"):
            # Region proposal network
            proposals_rpn, proposal_losses = self.proposal_generator(
                images, features, gt_instances
            )

            # roi_head lower branch
            _, detector_losses = self.roi_heads(
                images,
                features,
                proposals_rpn,
                compute_loss=True,
                targets=gt_instances,
                branch=branch,
            )

            # visualization
            if self.vis_period > 0:
                storage = get_event_storage()
                if storage.iter % self.vis_period == 0:
                    self.visualize_training(batched_inputs, proposals_rpn, branch)

            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            features = [feat[1] for feat in features.items()]
            return losses, [], [], features

        elif branch.startswith("supervised_target"):
            # Region proposal network
            proposals_rpn, proposal_losses = self.proposal_generator(
                images, features, gt_instances
            )

            # roi_head lower branch
            _, detector_losses = self.roi_heads(
                images,
                features,
                proposals_rpn,
                compute_loss=True,
                targets=gt_instances,
                branch=branch,
            )

            # visualization
            if self.vis_period > 0:
                storage = get_event_storage()
                if storage.iter % self.vis_period == 0:
                    self.visualize_training(batched_inputs, proposals_rpn, branch)

            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            return losses, [], [], None, features

        elif branch == "unsup_data_weak":
            # Unsupervised weak branch: input image without ground-truth labels
            # Region proposal network
            proposals_rpn, _ = self.proposal_generator(
                images, features, None, compute_loss=False
            )

            # ROI head inference without targets
            proposals_roih, ROI_predictions = self.roi_heads(
                images,
                features,
                proposals_rpn,
                targets=None,
                compute_loss=False,
                branch=branch,
            )

            return {}, proposals_rpn, proposals_roih, ROI_predictions, features
        elif branch == "TTT":
            if self.spegc_enable:
                # Run RPN and ROI heads on clean features to get stable proposals
                proposals_rpn, _ = self.proposal_generator(
                    images, features, None, compute_loss=False
                )

                proposals_roih, ROI_predictions = self.roi_heads(
                    images,
                    features,
                    proposals_rpn,
                    targets=None,
                    compute_loss=False,
                    branch=branch,
                )

                # Run MC Dropout passes on features to compute uncertainty_maps
                F_stochastic_passes = []
                for _ in range(self.spegc_t):
                    f_stochastic = {k: F.dropout2d(v, p=0.1, training=True) for k, v in features.items()}
                    F_stochastic_passes.append(f_stochastic)

                uncertainty_maps = {}
                for k in features.keys():
                    stacked = torch.stack([pass_feats[k] for pass_feats in F_stochastic_passes], dim=0) # (t, B, C, H, W)
                    var_k = torch.var(stacked, dim=0)
                    uncertainty_maps[k] = torch.mean(var_k, dim=1) # (B, H, W)

                features_list = [feat[1] for feat in features.items()]
                uncertainty_maps_list = [feat[1] for feat in uncertainty_maps.items()]

                # Filter foreground nodes from clean features using uncertainty maps
                current_nodes, current_labels = self.graph_generator(
                    features_list, proposals_roih, uncertainty_maps=uncertainty_maps_list, spegc_p=self.spegc_p
                )

                current_node_count = len(current_nodes[0]) if current_nodes and current_nodes[0] is not None else 0

                if current_nodes is None or current_labels is None or current_node_count == 0:
                    return None, [], [], features_list

                nodes_batch, labels_batch = self.ttt_graph_pool.create_pseudo_batch(
                    current_nodes[0],
                    current_labels[0]
                )

                centroids = self.centroids

                if self.ttt_graph_pool.should_do_matching() and len(nodes_batch) > 1:
                    loss = self.spegc_module(nodes_batch, centroids)
                else:
                    loss = None

                self.ttt_graph_pool.update_pool(current_nodes[0], current_labels[0])
                return loss, [], [], features_list
            else:
                raise ValueError("SPEGC is disabled, but TTT branch was selected.")
        elif branch == "val_loss":
            raise NotImplementedError()
        
    def process_pseudo_label(
        self, proposals_rpn_unsup_k, cur_threshold, proposal_type, psedo_label_method=""
    ):
        list_instances = []
        num_proposal_output = 0.0
        for proposal_bbox_inst in proposals_rpn_unsup_k:
            # thresholding
            if psedo_label_method == "thresholding":
                proposal_bbox_inst = self.threshold_bbox(
                    proposal_bbox_inst, thres=cur_threshold, proposal_type=proposal_type
                )
            else:
                raise ValueError("Unkown pseudo label boxes methods")
            num_proposal_output += len(proposal_bbox_inst)
            list_instances.append(proposal_bbox_inst)
        num_proposal_output = num_proposal_output / len(proposals_rpn_unsup_k)
        return list_instances, num_proposal_output

    def visualize_training(self, batched_inputs, proposals, branch=""):
        """
        This function different from the original one:
        - it adds "branch" to the `vis_name`.

        A function used to visualize images and proposals. It shows ground truth
        bounding boxes on the original image and up to 20 predicted object
        proposals on the original image. Users can implement different
        visualization functions for different models.

        Args:
            batched_inputs (list): a list that contains input to the model.
            proposals (list): a list that contains predicted proposals. Both
                batched_inputs and proposals should have the same length.
        """
        from detectron2.utils.visualizer import Visualizer

        storage = get_event_storage()
        max_vis_prop = 20

        for input, prop in zip(batched_inputs, proposals):
            img = input["image"]
            img = convert_image_to_rgb(img.permute(1, 2, 0), self.input_format)
            v_gt = Visualizer(img, None)
            v_gt = v_gt.overlay_instances(boxes=input["instances"].gt_boxes)
            anno_img = v_gt.get_image()
            box_size = min(len(prop.proposal_boxes), max_vis_prop)
            v_pred = Visualizer(img, None)
            v_pred = v_pred.overlay_instances(
                boxes=prop.proposal_boxes[0:box_size].tensor.cpu().numpy()
            )
            prop_img = v_pred.get_image()
            vis_img = np.concatenate((anno_img, prop_img), axis=1)
            vis_img = vis_img.transpose(2, 0, 1)
            vis_name = (
                "Left: GT bounding boxes "
                + branch
                + ";  Right: Predicted proposals "
                + branch
            )
            storage.put_image(vis_name, vis_img)
            break  # only visualize one image in a batch
