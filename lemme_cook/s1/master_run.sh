#!/bin/bash
#
# This is your "master file".
# It provides the 36 commands needed to run the full benchmark.
#
# HOW TO USE:
# 1. Create a shared output directory:
#    mkdir benchmark_results
#
# 2. On each computer, copy one or more of these commands and run them
#    in a terminal.
#
# 3. When all 36 jobs are done, run the analysis script ONCE:
#    python analysis.py --input-dir benchmark_results
#

OUTPUT_DIR="benchmark_results"
echo "--- Starting Benchmark ---"
echo "Output will be saved to: $OUTPUT_DIR"
echo "Run these 36 commands on your worker machines."
echo "------------------------------------------------"

# Define Detectors and Pipelines
DETECTORS=(
    "SSIM+MultiThreshold"
    "MultiScale-SSIM"
    "EdgeDelta-Blue"
    "ColorHist-32"
    "OpticalFlow"
    "GradientMag"
    "PatchCore-Lite"
    "BGS-MOG2"
    "Autoencoder-Recon"
    "TemplateMatch-ZS"
    "StatOutlier-ZS"
    "Frequency-ZS"
)

PIPELINES=(
    "Raw"
    "Median+SIFT"
    "Median+SIFT+Blue"
)

# Generate all commands
# The '&' at the end would run them in parallel *on one machine*
# For multiple machines, copy/paste the commands manually without the '&'.

for pipe in "${PIPELINES[@]}"; do
    for det in "${DETECTORS[@]}"; do
        echo "python run_benchmark_worker.py --detector \"$det\" --pipeline \"$pipe\" --output-dir \"$OUTPUT_DIR\""
    done
done

echo "------------------------------------------------"
echo "After all commands are finished, run:"
echo "python analysis.py --input-dir $OUTPUT_DIR"
