# AutoResearch Generated Sweep Specs

AutoResearch proposal outputs are written here by:

```bash
python3.11 App/run_autoresearch_pipeline.py --generate-next-sweep
```

The generated JSON files are normal `mode: "training_sweep"` specs, so you can run them through the standard pipeline entry point after reviewing the proposed grid refinements.

If you want AutoResearch to generate and launch the next round in one step, use:

```bash
python3.11 App/run_autoresearch_pipeline.py --generate-next-sweep --execute-generated-sweep
```
