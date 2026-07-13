import argparse
import json
from pathlib import Path


AREAS = ["Area_1", "Area_2", "Area_3", "Area_4", "Area_5", "Area_6"]


def parse_args():
    parser = argparse.ArgumentParser("Collect EGSR fold best_refiner.json results")
    parser.add_argument("--root", default="ckpt/S3DIS/egsr_6fold")
    parser.add_argument("--seed", type=int, default=2022)
    parser.add_argument("--out", default="")
    return parser.parse_args()


def fold_slug(area):
    return area.lower().replace("_", "")


def fmt(value):
    if value is None:
        return "NA"
    return "{:.2f}".format(float(value))


def collect(args):
    rows = []
    for area in AREAS:
        path = Path(args.root) / f"{fold_slug(area)}_egsr_seed{args.seed}_e8" / "best_refiner.json"
        if not path.exists():
            rows.append({"area": area, "path": str(path)})
            continue
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        rows.append(
            {
                "area": area,
                "path": str(path),
                "best_epoch": data.get("best_epoch"),
                "baseline_mIoU": data.get("baseline_mIoU"),
                "no_op_refined_mIoU": data.get("no_op_refined_mIoU"),
                "refined_mIoU": data.get("refined_mIoU"),
                "delta_mIoU": data.get("delta_mIoU"),
                "delta_vs_no_op_projection": data.get("delta_vs_no_op_projection"),
                "changed_ratio": data.get("changed_ratio"),
                "queries": data.get("queries"),
                "split_regions": data.get("split_regions"),
            }
        )
    return rows


def markdown(rows):
    lines = [
        "| Area | Best epoch | Frozen mIoU | Split/proj mIoU | EGSR mIoU | Delta vs frozen | Delta vs split/proj | Changed | Queries | Split regions |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    complete = []
    for row in rows:
        if "refined_mIoU" in row:
            complete.append(row)
            changed = "NA" if row.get("changed_ratio") is None else "{:.2f}%".format(100 * float(row["changed_ratio"]))
            lines.append(
                "| {area} | {epoch} | {base} | {noop} | {refined} | {delta} | {delta_noop} | {changed} | {queries} | {split_regions} |".format(
                    area=row["area"],
                    epoch=row.get("best_epoch", "NA"),
                    base=fmt(row.get("baseline_mIoU")),
                    noop=fmt(row.get("no_op_refined_mIoU")),
                    refined=fmt(row.get("refined_mIoU")),
                    delta=fmt(row.get("delta_mIoU")),
                    delta_noop=fmt(row.get("delta_vs_no_op_projection")),
                    changed=changed,
                    queries=row.get("queries", "NA"),
                    split_regions=row.get("split_regions", "NA"),
                )
            )
        else:
            lines.append(f"| {row['area']} | NA | NA | NA | NA | NA | NA | NA | NA | NA |")
    if complete:
        mean = sum(float(row["refined_mIoU"]) for row in complete) / len(complete)
        delta = sum(float(row["delta_mIoU"]) for row in complete) / len(complete)
        lines.append("")
        lines.append("Completed folds: {} / 6".format(len(complete)))
        lines.append("Mean EGSR mIoU over completed folds: {:.2f}".format(mean))
        lines.append("Mean delta vs frozen over completed folds: {:+.2f}".format(delta))
    else:
        lines.append("")
        lines.append("Completed folds: 0 / 6")
    return "\n".join(lines) + "\n"


def main():
    args = parse_args()
    text = markdown(collect(args))
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
