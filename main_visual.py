import argparse
import yaml
from pathlib import Path

# --- Our own modules ---
from baoflamingo.data_structure_visual import Config, Paths, Plotting
from baoflamingo.plotting import correlation_plotter


# -------------------------------------------------------------------
# Helper: build the full file path
# -------------------------------------------------------------------
def make_path(results_dir, simulation,  base, redshift, cosmology):
    """
    Construct the full path to the HDF5 correlation file.

    Example:
        /base/results_dir/simulation/redshift/cosmology/correlation_results.hdf5
    """
    filename = f"{cosmology}.hdf5"  # Change if your naming differs
    path=Path(base) / results_dir / simulation / str(redshift) / filename
    
    return path


# -------------------------------------------------------------------
# Helper: load YAML config and convert to dataclasses
# -------------------------------------------------------------------
def load_config(config_path: str) -> Config:
    """
    Load a YAML configuration file and unpack into dataclasses.
    """
    with open(config_path, "r") as f:
        cfg_dict = yaml.safe_load(f)

    paths_cfg = Paths(**cfg_dict["paths"])
    plotting_cfg = Plotting(**cfg_dict["plotting"])
    return Config(paths=paths_cfg, plotting=plotting_cfg)


# -------------------------------------------------------------------
# Main execution
# -------------------------------------------------------------------
def main():
    # --- Argument parser ---
    parser = argparse.ArgumentParser(description="Plot BAOflamingo PyCorr correlation outputs")
    parser.add_argument(
        "--config",
        type=str,
        default="configurations/config_visual.yaml",
        help="Path to the plotting YAML configuration file"
    )
    args = parser.parse_args()

    # --- Load configuration file ---
    cfg = load_config(args.config)

    # --- Build full HDF5 path ---
    for redshift in cfg.paths.redshift if isinstance(cfg.paths.redshift, list) else [cfg.paths.redshift]:
        for cosmology in cfg.paths.cosmology if isinstance(cfg.paths.cosmology, list) else [cfg.paths.cosmology]:
            
            data_filename = make_path(
                results_dir=cfg.paths.results_dir,
                simulation=cfg.paths.simulation,
                base=cfg.paths.base,
                redshift=redshift,
                cosmology=cosmology
            )

            print(f"Loading correlation data from: {data_filename}")

            # --- Run plotting pipeline ---
            data_plot = correlation_plotter(
                filename=data_filename,
                cfg=cfg,
                mu_rebin=cfg.plotting.mu_rebin,
                s_rebin=cfg.plotting.s_rebin
            )

            print("Plotting completed successfully.")


# -------------------------------------------------------------------
if __name__ == "__main__":
    main()
