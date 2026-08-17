
def resolve_dynamic_model_path(base_path: str, run_name: str, exp_tag: str = "") -> str:
    """
    Dynamically injects an experiment tag into a base LoRA path.

    Args:
        base_path (str): The default save path from the YAML config.
        run_name (str): The configuration run name (e.g., 'cloud_r8_lr2e-4').
        exp_tag (str): The optional experiment identifier (e.g., 'exp1').

    Returns:
        str: The fully resolved dynamic path.
    """
    if not exp_tag:
        return base_path

    if run_name in base_path:
        return base_path.replace(run_name, f"{exp_tag}_{run_name}")

    return f"{base_path}_{exp_tag}_{run_name}"