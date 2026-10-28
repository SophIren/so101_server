#!/usr/bin/env python3
"""Add opt-in local LoRA optimizer/RNG checkpointing to the author's trainer."""
import argparse
from pathlib import Path


def patch(path):
    path = Path(path)
    source = path.read_text()
    validate_marker = '    cfg.validate()\n'
    resume_base = '''
    # Adapter-only checkpoints contain no full model.safetensors. LeRobot's
    # resume validation points pretrained_path at the adapter, so recover the
    # original training base before make_policy; load_lora_adapters runs later.
    if cfg.resume and cfg.lora.enable:
        import json as _json_resume
        _resume_config_path = Path(cfg.checkpoint_path) / PRETRAINED_MODEL_DIR / "train_config.json"
        _resume_config = _json_resume.loads(_resume_config_path.read_text())
        _resume_base = Path(_resume_config["policy"]["pretrained_path"])
        if not (_resume_base / "model.safetensors").is_file():
            raise FileNotFoundError(f"Original LoRA training base missing: {_resume_base}")
        cfg.policy.pretrained_path = _resume_base
        print(f"[SO101 resume] Loading original base {_resume_base} before restoring adapters/optimizer")
'''
    if resume_base not in source:
        if source.count(validate_marker) != 1:
            raise ValueError('Unexpected trainer config validation')
        source = source.replace(validate_marker, validate_marker + resume_base)
    old_load = '            load_lora_adapters(policy, cfg.checkpoint_path)\n'
    new_load = ('            if not load_lora_adapters(policy, cfg.checkpoint_path):\n'
                '                raise RuntimeError("Resume adapter weights were not loaded")\n')
    if new_load not in source:
        if source.count(old_load) != 1:
            raise ValueError('Unexpected adapter resume handling')
        source = source.replace(old_load, new_load)
    old = '                    if is_resume_step:\n'
    new = '                    if is_resume_step or os.environ.get("VLASH_SAVE_LOCAL_RESUME") == "1":\n'
    if new not in source:
        if source.count(old) != 1:
            raise ValueError('Unexpected trainer checkpoint implementation')
        source = source.replace(old, new)
    marker = '                update_last_checkpoint(checkpoint_dir)\n'
    addition = '''
                # SO101 local resume: preserve every adapter, retain only the
                # newest optimizer/RNG state on the limited persistent volume.
                if cfg.lora.enable and os.environ.get("VLASH_SAVE_LOCAL_RESUME") == "1":
                    for _old_checkpoint in checkpoint_dir.parent.iterdir():
                        if _old_checkpoint.is_symlink() or _old_checkpoint == checkpoint_dir:
                            continue
                        _old_state = _old_checkpoint / "training_state"
                        if _old_state.is_dir():
                            shutil.rmtree(_old_state)
'''
    if addition not in source:
        if source.count(marker) != 1:
            raise ValueError('Unexpected last-checkpoint handling')
        source = source.replace(marker, marker + addition)
    compile(source, str(path), 'exec')
    path.write_text(source)
    print('Local LoRA resume patch applied:', path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('trainer', type=Path)
    args = parser.parse_args()
    patch(args.trainer)
