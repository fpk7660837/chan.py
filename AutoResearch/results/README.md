# AutoResearch Results

`AutoResearch/results/` is the default storage root for experiment runs.

Generated run artifacts, recommendation tables, training model files, and leaderboard files are intentionally ignored here so the repo only tracks the scaffold, not local research output.

Training runs store model artifacts under `AutoResearch/results/experiments/<experiment>/runs/<run_id>/models/` by default.

If a run explicitly enables publish/promote, the chosen model artifact may also be copied into the shared global `./models` directory, but that is an opt-in secondary publish target rather than the default storage location.
