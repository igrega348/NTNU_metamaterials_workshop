# Workshop data

Generated inputs and training datasets live here. Submodules (`neural_xray`, `fem_lattice_simulator`) are for **code only**.

```
data/
└── kelvin_indentation/              # Kelvin lattice indentation experiment
    ├── yaml/            # FEM scene YAML per timestep
    ├── renders/         # raw X-ray projections
    └── …                # staged transforms + images (after stage_kelvin_for_nerf.py)

outputs/
└── kelvin_indentation/              # neural_xray training checkpoints
```

See [`kelvin_indentation/README.md`](kelvin_indentation/README.md) for the full pipeline.
