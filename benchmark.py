import argparse
import multiprocessing as mp
from pathlib import Path

from metricnet.benchmark.evaluator import HabitatEvaluator
from metricnet.benchmark.models_utils import load_config
from metricnet.benchmark.plot import load_data, plot_results

if __name__ == "__main__":
    # enable multiprocessing
    mp.set_start_method("spawn", force=True)

    # set up paths and parse arguments
    parser = argparse.ArgumentParser(description="Visual Navigation Transformer")
    parser.add_argument(
        "--config",
        "-c",
        default="metricnet/config/benchmark.yaml",
        type=str,
        help="Path to the config file",
    )
    args = parser.parse_args()

    # load config
    eval_config = load_config(args.config)
    device = eval_config["device"]

    # run evaluation or plotting based on config
    if not eval_config.get("plot_only", False):
        for model_type, model_info in eval_config["models"].items():
            evaluator = HabitatEvaluator(
                benchmark_dir=eval_config["benchmark_dir"],
                model_type=model_type,
                model_cfg=model_info["model_cfg"],
                model_ckpt=model_info["model_ckpt"],
                result_dir=eval_config["result_dir"],
                num_workers=eval_config["num_workers"],
                max_n_agents=eval_config["max_agent_per_worker"],
                eval_type=eval_config["eval_type"],
                device=device,
                controller=eval_config["controller"],
                repetitions_per_trajectory=eval_config["repetitions_per_trajectory"],
                waypoint_idx=eval_config["waypoint_idx"],
                scale=eval_config.get("scale", 0.25),
            )

    # load data and plot results
    result_df = load_data(eval_config["result_dir"], eval_type=eval_config["eval_type"])
    algo_names = eval_config["models"].keys()
    result_df = result_df[result_df["algorithm"].isin(algo_names)]
    if result_df.empty:
        raise ValueError(
            "No data found. Check your directory structure and JSON files."
        )
    plot_results(
        result_df,
        save_dir=Path(f"{eval_config['result_dir']}"),
        eval_type=eval_config["eval_type"],
    )
