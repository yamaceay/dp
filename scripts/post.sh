#!/bin/bash

python parse_runtime.py && \
python merge_logs.py && \
python transform_logs.py && \
python single_metric_plots.py && \
python pu_tradeoff.py --all && \
python plot_summary.py --all && \
python plot_drift.py