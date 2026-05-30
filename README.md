# protoplanetary-disk-model-visibilities
This is the first version of a repository that create a protoplanetary disk emission and then calculate the visibilitites in ASCII format ready to use in MPoL and ALMA-DIP.

The table of visibilities is create using GALARIO.

## Run from any directory

`emission_pp_model.py` reads the model configuration from a directory. The
directory must contain one of these files:

- `model_config.py`
- `config.py`
- `model_config.json`
- `config.json`

For Python configs, define `MODEL_CONFIG` or `model_config`. You can also define
`RUN_CONFIG` or `run_config` for normalization, observation, and output options.

Example:

```bash
python /path/to/protoplanetary-disk-model-visibilities/emission_pp_model.py \
  --config-dir /path/to/my/config_directory
```

There is a ready-to-run example in `configs/example`:

```bash
python emission_pp_model.py --config-dir configs/example
```

To create a starter config in your current directory:

```bash
python /path/to/emission_pp_model.py --config-dir . --write-default-config
```
