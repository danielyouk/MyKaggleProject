# Data Schema for Yale_UNC-CH Geophysical Waveform Inversion Competition

This document outlines the data structure for the competition, based on the provided tutorial notebook (`waveform-inversion-kaggle-competition-tutorial.ipynb`) and the `convnext-full-resolution-baseline.ipynb`.

## 1. Input Data (Seismic Waveforms)

-   **Variable Name (in tutorial):** `data`
-   **File Naming Pattern:** `dataN.npy` (e.g., `data1.npy`, `data2.npy`)
-   **Loaded Shape per File (from tutorial):** `(S, C, T, G)`
    -   `S`: Number of individual seismic survey samples within the `.npy` file (e.g., 500 in `data2.npy` from tutorial).
    -   `C`: Number of channels, corresponding to seismic sources/shots. **Value = 5**.
    -   `T`: Number of time steps in the seismic recording. **Value = 1000**. (`n_t`)
    -   `G`: Number of geophones/receivers on the surface. **Value = 70**. (`n_g`)
-   **Shape fed to Model (Batch):** `[BatchSize, 5, 1000, 70]`
-   **Data Type:** `float` (likely `float16` or `float32`). The `convnext-full-resolution-baseline.ipynb` refers to `openfwi_float16` datasets.
-   **Physical Meaning:** Seismic waveform data `p(g,t)` recorded from 5 different surface source locations.
    -   Time sampling interval `d_t = 0.001 s`.
    -   Geophone spacing `d_g = 10 m`.
-   **Path in Competition Dataset (example from tutorial):** `/kaggle/input/waveform-inversion/train_samples/FlatVel_A/data/data2.npy`
-   **Path in Full Dataset (example from baseline):** `/kaggle/input/open-wfi-1/openfwi_float16_1/.../seis_....npy`

## 2. Target Data (Subsurface Velocity Maps)

-   **Variable Name (in tutorial):** `velocity`
-   **File Naming Pattern:** `modelN.npy` (e.g., `model1.npy`, `model2.npy`)
-   **Loaded Shape per File (from tutorial):** `(S, Ch, H, W)`
    -   `S`: Number of samples, corresponding to input data samples (e.g., 500).
    -   `Ch`: Number of channels. **Value = 1** (a single velocity map).
    -   `H`: Height of the velocity map grid (depth). **Value = 70**. (`n_z`)
    -   `W`: Width of the velocity map grid (offset). **Value = 70**. (`n_x`)
-   **Shape fed to Loss Function (Batch):** `[BatchSize, 1, 70, 70]`
-   **Data Type:** `float` (likely `float16` or `float32`).
-   **Physical Meaning:** Subsurface velocity map `c(x,z)` in m/s.
    -   Grid spacing `d_x = 10 m`.
    -   Grid spacing `d_z = 10 m`.
    -   The `convnext-full-resolution-baseline.ipynb` model output is scaled by `* 1500 + 3000`, suggesting target velocities are in this approximate range (e.g., 1500 m/s to 4500 m/s). Tutorial visualization labels velocity in `km/s`.
-   **Path in Competition Dataset (example from tutorial):** `/kaggle/input/waveform-inversion/train_samples/FlatVel_A/model/model2.npy`
-   **Path in Full Dataset (example from baseline):** `/kaggle/input/open-wfi-1/openfwi_float16_1/.../vel_....npy` (or `model_`)

## 3. Metadata / File Lists

-   **`convnext-full-resolution-baseline.ipynb` uses:**
    -   A `folds.csv` file which contains columns like `data_fpath` (relative path to the seismic data `.npy` file) and `fold` (for train/validation split). Paths in this CSV point to files within larger datasets like `openfwi_float16_1` and `openfwi_float16_2`.
-   **`waveform-inversion-kaggle-competition-tutorial.ipynb` uses:**
    -   Text files (e.g., `kaggle_tutorial_train.txt`, `kaggle_tutorial_val.txt`) that list pairs of paths to the model (velocity) `.npy` file and the data (seismic) `.npy` file, one pair per line. Example line: `path/to/model1.npy path/to/data1.npy`.
    -   These paths point to sample files within `/kaggle/input/waveform-inversion/train_samples/`.

## 4. Submission File (`submission.csv`)

-   **Format:** CSV file with a header.
-   **Columns:** `oid_ypos,x_1,x_3,...,x_69`
    -   `oid_ypos`: A unique identifier for each seismic sample (`oid`) combined with the y-position (row index, 0-69) in the velocity map. Example: `000039dca2_y_0`.
    -   `x_1, x_3, ..., x_69`: Predicted velocity values for the odd-indexed columns (1, 3, ..., 69) at that specific `y_pos`.
-   **Evaluation Metric:** Mean Absolute Error (MAE) on these predicted velocity values.

## Data Schema

Based on the competition data, the data schema is as follows:

*   **Input (Source Vibrational Signal):** A NumPy array representing the seismic recording from a single source. This is a 2D array with a shape of `(1000, 70)`, where:
    *   `1000` is the number of time steps.
    *   `70` is the number of geophone locations.
    *   *Note: The full seismic data for a given sample contains recordings from 5 sources, resulting in a shape of `(5, 1000, 70)`. The model will likely process one source at a time.*

*   **Target (Velocity Map):** A NumPy array representing the subsurface velocity map to be predicted. This is a 2D array with a shape of `(70, 70)`.

This schema should guide our local dummy data creation and script adaptation. 