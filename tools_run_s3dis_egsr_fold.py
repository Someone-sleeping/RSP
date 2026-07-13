import argparse
import os
import shlex
import subprocess
from pathlib import Path


AREAS = ["Area_1", "Area_2", "Area_3", "Area_4", "Area_5", "Area_6"]


def parse_args():
    parser = argparse.ArgumentParser("Run or print EGSR fold commands for S3DIS")
    parser.add_argument("--area", default="Area_5", choices=AREAS + ["all"], help="held-out area")
    parser.add_argument(
        "--stage",
        default="all",
        choices=["reference", "refiner", "eval", "projection", "all"],
        help="pipeline stage to run or print",
    )
    parser.add_argument("--root", default="ckpt/S3DIS/egsr_6fold", help="output root for fold artifacts")
    parser.add_argument("--teacher_ckpt_dir", default="", help="override fold reference ckpt dir for refiner/eval")
    parser.add_argument("--teacher_epoch", type=int, default=-1)
    parser.add_argument("--reference_resume", default="", help="optional train_S3DIS.py resume checkpoint for the reference stage")
    parser.add_argument(
        "--reference_max_epoch",
        type=int,
        nargs=2,
        default=None,
        metavar=("STAGE1", "STAGE2"),
        help="optional train_S3DIS.py --max_epoch override for bounded reference runs, e.g. 80 0",
    )
    parser.add_argument("--seed", type=int, default=2022)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--conda_env", default="cm_growsp")
    parser.add_argument("--execute", action="store_true", help="execute commands; otherwise print only")
    return parser.parse_args()


def fold_slug(area):
    return area.lower().replace("_", "")


def quote_cmd(cmd):
    return " ".join(shlex.quote(str(part)) for part in cmd)


def python_cmd(args, script, extra):
    return [
        "conda",
        "run",
        "-n",
        args.conda_env,
        "python",
        script,
        *[str(item) for item in extra],
    ]


def fold_paths(args, area):
    root = Path(args.root)
    slug = fold_slug(area)
    reference = root / f"{slug}_reference"
    refiner = root / f"{slug}_egsr_seed{args.seed}_e8"
    teacher_ckpt_dir = Path(args.teacher_ckpt_dir) if args.teacher_ckpt_dir else reference / "ckpts"
    return reference, refiner, teacher_ckpt_dir


def commands_for_area(args, area):
    reference, refiner, teacher_ckpt_dir = fold_paths(args, area)
    reference_extra = [
        "--save_path",
        reference,
        "--test_area",
        area,
    ]
    if args.reference_resume:
        reference_extra.extend(["--resume", args.reference_resume])
    if args.reference_max_epoch is not None:
        reference_extra.extend(["--max_epoch", *args.reference_max_epoch])
    reference_cmd = python_cmd(args, "train_S3DIS.py", reference_extra)
    refiner_cmd = python_cmd(
        args,
        "train_refiner_S3DIS.py",
        [
            "--save_path",
            refiner,
            "--pseudo_label_path",
            refiner / "pseudo_labels",
            "--teacher_ckpt_dir",
            teacher_ckpt_dir,
            "--teacher_epoch",
            args.teacher_epoch,
            "--test_area",
            area,
            "--max_epoch",
            8,
            "--eval_interval",
            1,
            "--save_interval",
            1,
            "--log_interval",
            20,
            "--batch_size",
            6,
            "--workers",
            8,
            "--cluster_workers",
            4,
            "--seed",
            args.seed,
            "--refine_lr",
            0.00015,
            "--refine_project_ce_lambda",
            0.2,
            "--refine_residual_scale",
            0.7,
        ],
    )
    eval_cmd = python_cmd(
        args,
        "eval_S3DIS.py",
        [
            "--save_path",
            refiner,
            "--eval_epoch",
            "best",
            "--test_area",
            area,
            "--refine_enable",
            "--refine_split_enable",
        ],
    )
    projection_cmd = python_cmd(
        args,
        "tools_eval_region_projection.py",
        [
            "--save_path",
            refiner,
            "--test_area",
            area,
        ],
    )
    return {
        "reference": reference_cmd,
        "refiner": refiner_cmd,
        "eval": eval_cmd,
        "projection": projection_cmd,
    }


def selected_stages(stage):
    if stage == "all":
        return ["reference", "refiner", "eval", "projection"]
    return [stage]


def run_cmd(args, cmd):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    print("$ CUDA_VISIBLE_DEVICES={} {}".format(args.gpu, quote_cmd(cmd)), flush=True)
    if args.execute:
        subprocess.run(cmd, check=True, env=env)


def main():
    args = parse_args()
    areas = AREAS if args.area == "all" else [args.area]
    for area in areas:
        print("\n# Held-out {}".format(area), flush=True)
        cmds = commands_for_area(args, area)
        for stage in selected_stages(args.stage):
            run_cmd(args, cmds[stage])


if __name__ == "__main__":
    main()
