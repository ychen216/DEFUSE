import argparse
import pathlib
from copy import deepcopy

import tensorflow as tf
import numpy as np

from pretrain import run
from stream_train_test import stream_run, stream_run_our


def run_params(args):
    params = deepcopy(vars(args))
    params["model"] = "MLP_SIG"
    params["optimizer"] = "Adam"
    if args.data_cache_path != "None":
        pathlib.Path(args.data_cache_path).mkdir(parents=True, exist_ok=True)
    if args.mode == "pretrain":
        if args.method == "Pretrain":
            params["loss"] = "cross_entropy_loss"
            params["dataset"] = "baseline_prtrain"
        elif args.method == "Pretrain_1d":
            params["loss"] = "cross_entropy_loss"
            params["dataset"] = "baseline_pretrain_1d_cut_hour_"+str(args.C)
        elif args.method == "DFM":
            params["loss"] = "delayed_feedback_loss"
            params["dataset"] = "dfm_prtrain"
            params["model"] = "MLP_EXP_DELAY"
        elif args.method == "ES-DFM":
            params["loss"] = "tn_dp_pretraining_loss"
            params["dataset"] = "tn_dp_mask30d_pretrain_cut_hour_"+str(args.C)
            params["model"] = "MLP_tn_dp"
        elif args.method == "ES-DFM_1d":
            params["loss"] = "tn_dp_pretraining_loss"
            params["dataset"] = "tn_dp_mask30d_pretrain_1d_cut_hour_"+str(args.C)
            params["model"] = "MLP_tn_dp"
        elif args.method == "Bi-DEFUSE_MLP":
            params["loss"] = "inw_outw_cross_entropy_loss"
            params["dataset"] = "bidefuse_pretrain_cut_hour_"+str(args.C)
            params["model"] = "Bi-DEFUSE_MLP"
        elif args.method == "Bi-DEFUSE_MLP_1d":
            params["loss"] = "inw_outw_cross_entropy_loss"
            params["dataset"] = "bidefuse_pretrain_1d_cut_hour_"+str(args.C)
            params["model"] = "Bi-DEFUSE_MLP"
        elif args.method == "Bi-DEFUSE":
            params["loss"] = "inw_outw_cross_entropy_loss"
            params["dataset"] = "bidefuse_pretrain_cut_hour_"+str(args.C)
            params["model"] = "Bi-DEFUSE_inoutw"
        elif args.method == "Bi-DEFUSE_1d":
            params["loss"] = "inw_outw_cross_entropy_loss"
            params["dataset"] = "bidefuse_pretrain_1d_cut_hour_"+str(args.C)
            params["model"] = "Bi-DEFUSE_inoutw"
        # elif args.method == "DEFUSE_inoutw_ind":
        #     params["loss"] = "inw_outw_cross_entropy_loss"
        #     params["dataset"] = "bidefuse_pretrain_cut_hour_"+str(args.C)
        #     params["model"] = "MLP_DEFUSE_inoutw_ind"
        elif args.method == "dp_1d":
            params["loss"] = "dp_loss"
            params["dataset"] = "dp_v2_1d_pretrain_cut_hour_"+str(args.C)
            params["model"] = "MLP_dp"
        elif args.method == "DEFER":
            params["loss"] = "dp_loss"
            params["dataset"] = "dp_pretrain_cut_hour_"+str(args.C)
            params["model"] = "MLP_dp"
        elif args.method == "DEFER_1d":
            params["loss"] = "dp_loss"
            params["dataset"] = "dp_pretrain_1d_cut_hour_"+str(args.C)
            params["model"] = "MLP_dp"
        else:
            raise ValueError(
                "{} method do not need pretraining other than Pretrain".format(args.method))
    else:
        if args.method == "Pretrain":
            params["loss"] = "none_loss"
            params["dataset"] = "last_30_train_test_oracle"
        elif args.method == "Pretrain_1d":
            params["loss"] = "none_loss"
            params["dataset"] = "last_30_1d_train_test_oracle"
        elif args.method == "Oracle":
            params["loss"] = "cross_entropy_loss"
            params["dataset"] = "last_30_train_test_oracle"
        elif args.method == "Oracle_1d":
            params["loss"] = "cross_entropy_loss"
            params["dataset"] = "last_30_1d_train_test_oracle"
        elif args.method == "DFM":
            params["loss"] = "delayed_feedback_loss"
            params["dataset"] = "last_30_train_test_dfm"
        elif args.method == "ES-DFM":
            params["loss"] = "esdfm_loss"
            params["dataset"] = "last_30_train_test_esdfm_oracle_v2_cut_hour_" + \
                str(args.C)
        elif args.method == "ES-DFM_1d":
            params["loss"] = "esdfm_loss"
            params["dataset"] = "last_30_train_test_esdfm_oracle_v2_1d_cut_hour_" + \
                str(args.C)
        elif args.method == "DEFUSE":
            params["loss"] = "defuse_loss"
            params["dataset"] = "last_30_train_test_esdfm_oracle_v2_cut_hour_" + \
                str(args.C)
        elif args.method == "DEFUSE_1d":
            params["loss"] = "defuse_loss"
            params["dataset"] = "last_30_train_test_esdfm_oracle_v2_1d_cut_hour_" + \
                str(args.C)
        elif args.method == "DEFUSE_3d":
            params["loss"] = "defuse_loss"
            params["dataset"] = "last_30_train_test_esdfm_oracle_v2_3d_cut_hour_" + \
                str(args.C)
        elif args.method == "DEFUSE_7d":
            params["loss"] = "defuse_loss"
            params["dataset"] = "last_30_train_test_esdfm_oracle_v2_7d_cut_hour_" + \
                str(args.C)
        elif args.method == "DEFUSE_14d":
            params["loss"] = "defuse_loss"
            params["dataset"] = "last_30_train_test_esdfm_oracle_v2_14d_cut_hour_" + \
                str(args.C)
        elif args.method == "Bi-DEFUSE_MLP":
            params["loss"] = "bidefuse_loss"
            params["dataset"] = "last_30_train_test_bidefuse_cut_hour_"+str(args.C)
        elif args.method == "Bi-DEFUSE":
            params["loss"] = "bidefuse_loss"
            params["dataset"] = "last_30_train_test_bidefuse_cut_hour_"+str(args.C)
        elif args.method == "Bi-DEFUSE_1d":
            params["loss"] = "bidefuse_loss"
            params["dataset"] = "last_30_train_test_bidefuse_1d_cut_hour_"+str(args.C)
        # elif args.method == "DEFUSE_inoutw_1d":
        #     params["loss"] = "bidefuse_loss"
        #     params["dataset"] = "last_30_train_test_bidefuse_1d_cut_hour_"+str(args.C)
        # elif args.method == "DEFUSE_inoutw_MTL_1d":
        #     params["loss"] = "bidefuse_loss"
        #     params["dataset"] = "last_30_train_test_bidefuse_1d_cut_hour_"+str(args.C)
        elif args.method == "DEFER":
            params["loss"] = "defer_loss"
            params["dataset"] = "last_30_train_test_defer_cut_hour_{}_attr_day_{}".format(args.C, args.W)
            # params["dataset"] = "last_30_train_test_defer_oracle_cut_hour_{}_attr_day_{}".format(args.C, args.W)
        elif args.method == "DEFER_unbiased":
            params["loss"] = "unbiased_defer_loss"
            params["dataset"] = "last_30_train_test_defer_cut_hour_{}_attr_day_{}".format(args.C, args.W)
            # params["dataset"] = "last_30_train_test_defer_oracle_cut_hour_{}_attr_day_{}".format(args.C, args.W)
        elif args.method == "Vanilla":
            params["loss"] = "cross_entropy_loss"
            params["dataset"] = "last_30_train_test_vanilla_cut_hour_" + \
                str(args.C)
        elif args.method == "Vanilla_1d":
            params["loss"] = "cross_entropy_loss"
            params["dataset"] = "last_30_train_test_vanilla_1d_cut_hour_" + \
                str(args.C)
        elif args.method == "Vanilla-win":
            params["loss"] = "cross_entropy_loss"
            params["dataset"] = "last_30_train_test_esdfm_cut_hour_" + \
                str(args.C)
        elif args.method == "Vanilla-win_1d":
            params["loss"] = "cross_entropy_loss"
            params["dataset"] = "last_30_train_test_vanilla_1d_cut_hour_" + \
                str(args.C)
        elif args.method == "FNW":
            params["loss"] = "fake_negative_weighted_loss"
            params["dataset"] = "last_30_train_test_fnw"
        elif args.method == "FNW_1d":
            params["loss"] = "fake_negative_weighted_loss"
            params["dataset"] = "last_30_train_test_fnw_1d"
        elif args.method == "FNW_unbiased":
            params["loss"] = "unbiased_fake_negative_weighted_loss"
            params["dataset"] = "last_30_train_test_fnw"
        elif args.method == "FNW_unbiased_1d":
            params["loss"] = "unbiased_fake_negative_weighted_loss"
            params["dataset"] = "last_30_train_test_fnw_1d"
        elif args.method == "FNC":
            params["loss"] = "cross_entropy_loss"
            params["dataset"] = "last_30_train_test_fnw"
        elif args.method == "FNC_1d":
            params["loss"] = "cross_entropy_loss"
            params["dataset"] = "last_30_train_test_fnw_1d"
    return params


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", help="delayed feedback method", default='Bi-DEFUSE', 
                        choices=[
                                 "DFM",
                                 "ES-DFM",
                                 "ES-DFM_1d",
                                 "dp",
                                 "DEFER",
                                 "DEFER_1d",
                                 "DEFER_unbiased",
                                 "DEFER_unbiased_1d",
                                 "DEFER_1d",
                                 "DEFUSE",
                                 "DEFUSE_1d",
                                 "Bi-DEFUSE",
                                 "Bi-DEFUSE_1d",
                                 "Bi-DEFUSE_MLP",
                                 "Bi-DEFUSE_MLP_1d",
                                 "FNW",
                                 "FNW_unbiased",
                                 "FNW_1d",
                                 "FNW_unbiased_1d",
                                 "FNC",
                                 "FNC_1d",
                                 "Pretrain",
                                 "Pretrain_1d",
                                 "Oracle",
                                 "Oracle_1d",
                                 "Vanilla",
                                 "Vanilla_1d",
                                 "Vanilla-win",
                                 "Vanilla-win_1d"],
                        type=str)
    parser.add_argument(
        "--mode", default="stream", type=str, choices=["pretrain", "stream"], help="training mode")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--C", type=float, default=1.0,
                        help="elapsed time in ES-DFM/DEFER")
    parser.add_argument("--W", type=int, default=1,
                        help="attribution time in DEFER")
    parser.add_argument("--loss_op", type=str, default='em',
                        help="loss op")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--data_path", type=str, default='../data/criteo/data.txt',
                        help="path of the data.txt in criteo dataset, e.g. /home/xxx/data.txt")
    parser.add_argument("--data_cache_path", type=str, default="./delayed_feedback_release/data_cache")
    parser.add_argument("--model_ckpt_path", type=str,
                        help="path to save pretrained model")
    parser.add_argument("--pretrain_baseline_model_ckpt_path", type=str, default='./delayed_feedback_release/pretrain/pretrain',
                        help="path to the checkpoint of pretrained baseline model(Pretrain),  \
                        necessary for the streaming evaluation of \
                         ES-DFM, FNW, FNC, Pretrain, Oracle, Vanilla method")
    parser.add_argument("--pretrain_dfm_model_ckpt_path", type=str, default='./pretrain_ckpts/dfm/dfm',
                        help="path to the checkpoint of pretrained DFM model,  \
                        necessary for the streaming evaluation of \
                            DFM method")
    parser.add_argument("--pretrain_esdfm_model_ckpt_path", type=str, default="./delayed_feedback_release/ckpts/esdfm_oracle/esdfm_oracle",
                        help="path to the checkpoint of pretrained ES-DFM model,  \
                        necessary for the streaming evaluation of \
                        ES-DFM method")
    parser.add_argument("--pretrain_defuse_model_ckpt_path", type=str, default="./delayed_feedback_release/ckpts/pretrain/pretrain",
                        help="path to the checkpoint of our's pretrained model,  \
                        necessary for the streaming evaluation for our method")
    parser.add_argument("--pretrain_dp_model_ckpt_path", type=str, default="",
                        help="path to the checkpoint of pretrained dp model,  \
                        necessary for the streaming evaluation of \
                        dp method")
    parser.add_argument("--pretrain_defer_model_ckpt_path", type=str, default='',
                        help="path to the checkpoint of pretrained DEFER model,  \
                        necessary for the streaming evaluation of \
                        DEFER method")
    parser.add_argument("--batch_size", type=int,
                        default=1024)
    parser.add_argument("--epoch", type=int, default=5,
                        help="training epoch of pretraining")
    parser.add_argument("--l2_reg", type=float, default=1e-6,
                        help="l2 regularizer strength")

    args = parser.parse_args()
    params = run_params(args)
    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)
    print("params {}".format(params))
    if args.mode == "pretrain":
        run(params)
    else:
        if args.method in ["Bi-DEFUSE", "Bi-DEFUSE_1d", "Bi-DEFUSE_MLP", "Bi-DEFUSE_MLP_1d"]:
            stream_run_our(params)
        else:
            stream_run(params)
        # stream_run_our(params)
            # stream_run_ori(params)