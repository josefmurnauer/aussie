import os
import sys
from omegaconf import OmegaConf


def get_prev_config(prev_exp_dir):
    return OmegaConf.load(os.path.join(prev_exp_dir, ".hydra/config.yaml"))


def get_prev_overrides(prev_exp_dir):
    return OmegaConf.load(os.path.join(prev_exp_dir, ".hydra/overrides.yaml"))


def _strip_hydra_override_prefix(override_str):
    """Strip Hydra's own override-syntax prefixes ('+' = add new key,
    '++' = force add/override, '~' = delete) from a raw CLI override
    string before feeding it to OmegaConf.from_dotlist.

    Context: when composing a fresh Hydra config from the defaults tree,
    '+key=value' is REQUIRED to add a key that doesn't exist yet in the
    base config (otherwise Hydra's struct-checking rejects it outright,
    before main() even runs). But once inside main(), update_config_from_prev
    discards that composed cfg and rebuilds one by merging `prev_cfg`
    (a plain, non-struct OmegaConf config loaded from disk) with the RAW
    override strings via OmegaConf.from_dotlist -- which has no notion of
    '+'/'++'/'~' and would otherwise take the prefix as a literal
    character in the key name (e.g. "+tunfold.enabled=true" becomes a
    bogus top-level key "+tunfold", NOT "tunfold"). Since OmegaConf.merge
    onto a non-struct config can add new keys unconditionally, the prefix
    serves no purpose at this second stage and must be removed.
    """
    if override_str.startswith("++"):
        return override_str[2:]
    if override_str.startswith("+"):
        return override_str[1:]
    if override_str.startswith("~"):
        # deletions aren't meaningfully supported by this simple merge;
        # strip the marker so at least the key name parses, rather than
        # crashing -- deleting keys from prev_cfg this way isn't handled
        # here and would need separate logic if ever required.
        return override_str[1:]
    return override_str


def update_config_from_prev(cfg, hydra_cfg, prev_exp_dir):
    prev_cfg = get_prev_config(prev_exp_dir)
    raw_overrides = OmegaConf.to_object(hydra_cfg.overrides.task)
    cleaned_overrides = [_strip_hydra_override_prefix(s) for s in raw_overrides]
    overrides = OmegaConf.from_dotlist(cleaned_overrides)
    return OmegaConf.merge(prev_cfg, overrides)


def check_cfg(cfg, log):

    # exit
    if (cfg.prev_exp_dir and cfg.train) and not cfg.training.warm_start:
        log.error(
            "Rerunning experiment with train=True but warm_start=False. Exiting to avoid overwrite."
        )
        sys.exit()