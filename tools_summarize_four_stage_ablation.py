import argparse
import json


MODULE_PREFIXES = {
    "split": ("split_noop", "split_multi_noop", "split_merge_"),
    "refiner": (
        "split_refiner",
        "split_multi_refiner",
        "refiner_scale_",
        "refiner_local_",
        "scale_select_",
        "typed_refine_",
        "scale_meta_",
    ),
    "verifier": (
        "point_verified_",
        "region_verified_",
        "point_region_union_",
        "refiner_verified_union_",
        "meta_local_",
        "meta_region_",
    ),
    "meta": (
        "meta_scene_poe",
        "meta_init_poe_",
        "meta_adapt_",
        "selected_joint",
    ),
}


def parse_args():
    parser = argparse.ArgumentParser("Summarize four-stage upper-bound ablations")
    parser.add_argument("json_files", nargs="+")
    parser.add_argument("--output", default="")
    return parser.parse_args()


def matches(name, prefixes):
    return any(name == prefix or name.startswith(prefix) for prefix in prefixes)


def summarize(path):
    with open(path, "r", encoding="utf-8") as input_file:
        payload = json.load(input_file)
    results = payload["results"]
    rows = []
    for module, prefixes in MODULE_PREFIXES.items():
        candidates = {
            name: result
            for name, result in results.items()
            if matches(name, prefixes) and "oracle_diagnostic" not in name
        }
        if not candidates:
            continue
        best_name, best = max(candidates.items(), key=lambda item: item[1]["mIoU"])
        point = results.get(f"{module}_point_oracle_diagnostic", {})
        region = results.get(f"{module}_region_oracle_diagnostic", {})
        rows.append(
            {
                "module": module,
                "best": best_name,
                "mIoU": best["mIoU"],
                "delta": best["delta_mIoU"],
                "point_oracle": point.get("mIoU"),
                "region_oracle": region.get("mIoU"),
            }
        )
    return payload.get("test_area", path), rows


def render(summaries):
    lines = []
    for area, rows in summaries:
        lines.extend(
            [
                f"## {area}",
                "",
                "| Module | Best actual strategy | mIoU | Delta | Region oracle | Point oracle |",
                "|---|---|---:|---:|---:|---:|",
            ]
        )
        for row in rows:
            region = "-" if row["region_oracle"] is None else f'{row["region_oracle"]:.4f}'
            point = "-" if row["point_oracle"] is None else f'{row["point_oracle"]:.4f}'
            lines.append(
                f'| {row["module"]} | `{row["best"]}` | {row["mIoU"]:.4f} '
                f'| {row["delta"]:+.4f} | {region} | {point} |'
            )
        lines.append("")
    return "\n".join(lines)


def main():
    args = parse_args()
    text = render([summarize(path) for path in args.json_files])
    print(text)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as output_file:
            output_file.write(text + "\n")


if __name__ == "__main__":
    main()
