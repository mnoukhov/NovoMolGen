# AGENTS Notes

## Finetune Configs

- `run_finetune()` uses `cfg.finetune`, not the top-level `trainer` batch-size block.
- For AugmentedHC/REINVENT finetune, `finetune.per_device_train_batch_size` is the relevant batch size.
- The top-level BPE finetune config at [configs/finetune_PMO_ZINC_1B_bpe_smiles_llama-32M.yaml](/weka/oe-adapt-default/michaeln/NovoMolGen/configs/finetune_PMO_ZINC_1B_bpe_smiles_llama-32M.yaml) was overriding `finetune.checkpoint` to `32M`; that had to be removed so `--finetune=augment_hc_trainer_pmo_157M_SMILES_BPE` or `300M` can bring their own checkpoint.
- The size-specific AugmentedHC configs now point at HF checkpoints directly:
  - `32M`: `chandar-lab/NovoMolGen_32M_SMILES_BPE` / `..._AtomWise`
  - `157M`: `chandar-lab/NovoMolGen_157M_SMILES_BPE` / `..._AtomWise`
  - `300M`: `chandar-lab/NovoMolGen_300M_SMILES_BPE` / `..._AtomWise`
- The duplicated `.yaml.yaml` BPE finetune files were renamed to `.yaml`.

## Verified Launch Patterns

- There are no dedicated top-level `157M` or `300M` finetune entry configs. Use the `32M` top-level finetune config and override `--model` plus `--finetune`.
- Example shape:
  - `uv run src/main.py finetune --config_name=finetune_PMO_ZINC_1B_bpe_smiles_llama-32M --model=llama-157M --finetune=augment_hc_trainer_pmo_157M_SMILES_BPE`
- `smolgen` BPE checkpoints can be used by overriding `--finetune.checkpoint=ddidacus/...`.

## Runtime Compatibility Fixes

- Flash-attn generation was incompatible with newer Transformers output containers:
  - failure: `flash_attn/utils/generation.py` tried `output.scores = None`
  - fix in [src/models/modeling_novomolgen.py](/weka/oe-adapt-default/michaeln/NovoMolGen/src/models/modeling_novomolgen.py): call `self.generate(..., return_dict_in_generate=False, output_scores=True)`
- This fix was validated locally on the exact `run_gantry.sh`-style 157M BPE finetune command. It loaded `chandar-lab/NovoMolGen_157M_SMILES_BPE` and completed a short run successfully.

## Import-Time Fragility

- Finetune jobs should not import the dataset pipeline at process startup.
- [src/main.py](/weka/oe-adapt-default/michaeln/NovoMolGen/src/main.py) was changed so `MolDataModule` is imported only inside `run_train()` and `run_tokenize()`.
- This avoids unrelated finetune crashes from `molvs` / RDKit InChI import failures.

## Evaluator Import Fixes

- [src/eval/components/moses.py](/weka/oe-adapt-default/michaeln/NovoMolGen/src/eval/components/moses.py) no longer imports `rdkit.Chem.Descriptors.MolWt`.
- It now uses `rdMolDescriptors.CalcExactMolWt`, which avoids an RDKit `Fragments` import path that was crashing startup with:
  - `ImportError: Smarts '[CX3]=[OX1]' could not be parsed`
- `from src.eval import MoleculeEvaluator` was verified to import successfully after this change.

## Known Remaining Runtime Hazards

- `tadf` import still warns if `XTBHOME` is unset. This is non-fatal for unrelated tasks.
- Some helper imports are still sensitive to cwd-relative data paths under `./data/...`; direct utility scripts can fail if run from the wrong working directory.
- A short Sitagliptin local smoke run hit a later flash-attn Triton layer-norm device error during preload:
  - `RuntimeError: invalid argument to exchangeDevice`
  - This happened after correct `157M` checkpoint load and is separate from the earlier startup/import issues.

## Gantry Wrapper Behavior

- [scripts/run_gantry.sh](/weka/oe-adapt-default/michaeln/NovoMolGen/scripts/run_gantry.sh):
  - supports `GANTRY_SHOW_LOGS=1` to add `--show-logs`
  - appends `GANTRY_GROUP` to `WANDB_TAGS` when present
- `launch_all_tasks.sh` just loops over tasks and forwards extra finetune args.
- Passing a brand-new `GANTRY_GROUP` causes Gantry to prompt for group creation; in non-interactive runs this aborts submission. Use an existing group or omit `GANTRY_GROUP`.

## Naming Notes

- Finetune run names come from `creat_unique_experiment_name_for_finetune()`.
- The `AugmentedHC_SMILES-<hash>-...` prefix hash is derived from the non-finetune config after removing `finetune`; it is not directly user-settable from the CLI today.
