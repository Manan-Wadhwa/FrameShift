# Visual Detection Benchmark Framework

This project contains a modular, parallel-friendly benchmark harness for testing computer vision detection methods, as refactored from a source PDF.

## Exhaustive Setup Instructions

### Step 1: Create the File Structure

Ensure all 11 Python files and `requirements.txt` are in the same directory.

### Step 2: Set Up Your Environment

1.  **Install Python:** (Python 3.8+ recommended).
2.  **Install Dependencies:** Open a terminal in this folder and run:
    ```bash
    pip install -r requirements.txt
    ```

### Step 3: Configure Your Datasets

1.  **Download Datasets:** Ensure your MVTec AD and CARDD datasets are downloaded.
2.  **Edit `config_and_utils.py`:** Open this file.
3.  **Find `CELL 7.2`:** Scroll to the `DETECTION_TEST_CASES` section.
4.  **Update Paths:** Change all file paths to match the exact location of your images on your file system. This is the most critical step.

### Step 4: (Optional) Pre-train Learning-Based Models

If you plan to test `PatchCore-Lite`, you must pre-train its memory bank.

1.  Run the training script **once**:
    ```bash
    python train_models.py
    ```
2.  This creates a `patchcore_lite_memory.npy` file, which the workers will automatically load.

### Step 5: Run the Benchmark on Multiple Computers

This process is designed for parallel execution. You must launch 36 "worker" jobs in total (12 detectors x 3 pipelines).

1.  **Copy Project:** Copy this entire folder to every computer you will use for the benchmark.
2.  **Install Dependencies:** Run `pip install -r requirements.txt` on *each* computer.
3.  **Create Output Directory:** On your main computer (or a shared network drive), create a folder to hold the results:
    ```bash
    mkdir benchmark_results
    ```
4.  **Launch Workers:** On each computer, run a `run_benchmark_worker.py` command.
    * You must specify a `--detector`, a `--pipeline`, and the `--output-dir`.
    * See `master_run.sh` for the full list of all 36 commands.

    **Example on Computer 1:**
    ```bash
    python run_benchmark_worker.py --detector "SSIM+MultiThreshold" --pipeline "Raw" --output-dir "benchmark_results"
    ```
    **Example on Computer 2:**
    ```bash
    python run_benchmark_worker.py --detector "MultiScale-SSIM" --pipeline "Raw" --output-dir "benchmark_results"
    ```
    *Each worker will run all test cases for its job and save a single `.json` file to the output directory.*

### Step 6: Analyze the Results

1.  **Wait:** Wait for all 36 worker jobs to finish. Your `benchmark_results` folder should have 36 `.json` files.
2.  **Run Analysis:** On your **main computer**, run:
    ```bash
    python analysis.py
    ```
3.  **View Output:** This script combines all results and creates an `analysis_output` folder containing your final reports, charts, and CSVs.
