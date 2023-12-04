import re
from typing import Literal

import torch

from .hawp import HAWPv1, HAWPv2
from .lcnn import LCNN
from .letr import LETR
from .fclip import FClip


MODELS = ["hawp-v1", "hawp-v2", "lcnn", "letr-r101", "letr-r50", "fclip", "fclip-lb"]


def build_hawp(version: Literal[1, 2]):
    if version == 1:
        state_dict = torch.load("weights/model-hawp-hg-5d31f70.pth",
                                map_location="cpu")
        model = HAWPv1()
    else:
        checkpoint = torch.load("weights/hawpv2-edb9b23f.pth",
                                map_location="cpu")
        state_dict = checkpoint["model"]
        model = HAWPv2()

    new_state_dict = {}
    compat = (
        (r"^regional_head.*$",                  r""),
        (r"^line_mlp.*$",                       r""),
        (r"(?<=^backbone\.)hg\.(\d+)\.hg\.",    r"hourglasses.\1.0.layers."),
        (r"(?<=^backbone\.)res\.(\d+)\.",       r"hourglasses.\1.1."),
        (r"(?<=^backbone\.)fc\.(\d+)\.0\.",     r"hourglasses.\1.2."),
        (r"(?<=^backbone\.)fc\.(\d+)\.1\.",     r"hourglasses.\1.3."),
        (r"(?<=^backbone\.)score\.",            r"heads."),
        (r"(?<=^backbone\.)fc_\.",              r"refinements."),
        (r"(?<=^backbone\.)score_\.",           r"remap_convs."),
    )
    for name, param in state_dict.items():
        for name_from, name_to in compat:
            name = re.sub(name_from, name_to, name)
        if name != "":
            new_state_dict[name] = param
    state_dict = new_state_dict

    model.load_state_dict(state_dict)
    return model


def build_lcnn():
    checkpoint = torch.load("weights/190418-201834-f8934c6-lr4d10-312k.pth.tar",
                            map_location="cpu")
    state_dict = checkpoint["model_state_dict"]
    new_state_dict = {}
    compat = (
        (r"^backbone\.",                        r""),
        (r"(?<=^backbone\.)hg\.(\d+)\.hg\.",    r"hourglasses.\1.0.layers."),
        (r"(?<=^backbone\.)res\.(\d+)\.",       r"hourglasses.\1.1."),
        (r"(?<=^backbone\.)fc\.(\d+)\.0\.",     r"hourglasses.\1.2."),
        (r"(?<=^backbone\.)fc\.(\d+)\.1\.",     r"hourglasses.\1.3."),
        (r"(?<=^backbone\.)score\.",            r"heads."),
        (r"(?<=^backbone\.)fc_\.",              r"refinements."),
        (r"(?<=^backbone\.)score_\.",           r"remap_convs."),
    )
    for name, param in state_dict.items():
        for name_from, name_to in compat:
            name = re.sub(name_from, name_to, name)
        new_state_dict[name] = param
    state_dict = new_state_dict
    model = LCNN()
    model.load_state_dict(state_dict)
    return model


def build_letr(backbone: Literal["resnet50", "resnet101"]):
    if backbone == "resnet50":
        checkpoint = torch.load("weights/res50_stage2_focal.pth",
                                map_location="cpu")
    else:
        checkpoint = torch.load("weights/res101_stage2_focal.pth",
                                map_location="cpu")
    state_dict = checkpoint["model"]
    new_state_dict = {}
    compat = (
        (r"^letr\.backbone\..*$",       r""),
        (r"^letr\.class_embed\..*$",    r""),
        (r"^letr\.lines_embed\..*$",    r""),
        (r"^backbone\.0\.body\.",       r"backbone."),
        (r"^letr\.query_embed\.",       r"query_embed."),
        (r"^letr\.input_proj\.",        r"input_proj1."),
        (r"^letr\.transformer\.",       r"transformer1."),
        (r"^input_proj\.",              r"input_proj2."),
        (r"^transformer\.",             r"transformer2."),
        (r"^lines_embed\.layers\.0",    r"lines_embed.0"),
        (r"^lines_embed\.layers\.1",    r"lines_embed.2"),
        (r"^lines_embed\.layers\.2",    r"lines_embed.4")
    )
    for name, param in state_dict.items():
        for name_from, name_to in compat:
            name = re.sub(name_from, name_to, name)
        if name != "":
            new_state_dict[name] = param
    state_dict = new_state_dict
    model = LETR(backbone=backbone)
    model.load_state_dict(state_dict)
    return model


def build_fclip(line_block: bool):
    if line_block:
        checkpoint = torch.load("weights/HG2_LB.pth.tar",
                                map_location="cpu")
    else:
        checkpoint = torch.load("weights/HG2.pth.tar",
                                map_location="cpu")
    state_dict = checkpoint["model_state_dict"]
    new_state_dict = {}
    compat = (
        (r"(?<=^backbone\.)hg\.(\d+)\.hg\.",        r"hourglasses.\1.0.layers."),
        (r"(?<=^backbone\.)res\.(\d+)\.",           r"hourglasses.\1.1."),
        (r"(?<=^backbone\.)fc\.(\d+)\.0\.",         r"hourglasses.\1.2."),
        (r"(?<=^backbone\.)fc\.(\d+)\.1\.",         r"hourglasses.\1.3."),
        (r"(?<=^backbone\.)score\.",                r"heads."),
        (r"(?<=^backbone\.)fc_\.",                  r"refinements."),
        (r"(?<=^backbone\.)score_\.",               r"remap_convs."),
        (r"(?<=^backbone\.)merge_fc\.(\d+)\.0\.",   r"hourglasses.\1.5."),
        (r"(?<=^backbone\.)merge_fc\.(\d+)\.1\.",   r"hourglasses.\1.6."),
        (r"(?<=conv2\.)0.",                         r"conv_h."),
        (r"(?<=conv2\.)1.",                         r"conv_v."),
    )
    if line_block:
        compat = ((r"(^backbone\.(hg|res|fc)\.\d+\.)0\.", r"\1"),) + compat
    for name, param in state_dict.items():
        for name_from, name_to in compat:
            name = re.sub(name_from, name_to, name)
        new_state_dict[name] = param
    state_dict = new_state_dict
    model = FClip(line_block)
    model.load_state_dict(state_dict)
    return model


def build_model(model: str):
    if model == "lcnn":
        return build_lcnn()
    if model == "letr-r50":
        return build_letr(backbone="resnet50")
    if model == "letr-r101":
        return build_letr(backbone="resnet101")
    if model == "fclip":
        return build_fclip(line_block=False)
    if model == "fclip-lb":
        return build_fclip(line_block=True)
    if model == "hawp-v1":
        return build_hawp(version=1)
    if model == "hawp-v2":
        return build_hawp(version=2)

    raise NotImplementedError()
