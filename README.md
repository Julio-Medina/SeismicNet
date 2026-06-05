# SeismicNet

Convolutional neural network for automatic seismic phase detection in single-station waveform traces.

SeismicNet was developed as part of the B.Sc. Physics thesis **"Desarrollo de red neuronal convolucional para análisis de señales sísmicas"** at the Universidad de San Carlos de Guatemala. The project combines two parts: a theoretical study of neural networks from the perspective of statistical mechanics, spin glasses, the Hopfield model, and the work of Parisi; and an applied deep-learning system for detecting seismic P- and S-wave arrivals in local seismograms.

## Project overview

Manual seismic phase picking is an important but time-consuming step in routine earthquake analysis. SeismicNet approaches the problem as a supervised sequence-labeling task: given a normalized seismic trace, the model predicts three probability distributions along the time axis:

1. probability of P-wave arrival,
2. probability of S-wave arrival,
3. probability of noise/background.

The final phase picks are obtained by locating the strongest peaks in the predicted P and S probability distributions. The evaluation criterion used in the thesis counts a prediction as correct when the absolute temporal difference between the model pick and the manual label is below a chosen threshold, especially $\Delta t < 1$ s.

## Scientific motivation

The thesis frames neural networks through the language of statistical physics. The theoretical part reviews Ising models, mean-field theory, spin glasses, the Sherrington-Kirkpatrick model, replica symmetry breaking, and statistical models of neural networks such as the Hopfield model. This gives the project a physics-first motivation: neural networks are not treated only as engineering tools, but also as complex systems whose learning dynamics and emergent behavior can be studied with ideas from statistical mechanics.

The applied part then builds SeismicNet as a convolutional model for seismic waveform analysis, using seismological data from INSIVUMEH and testing the trained model on previously unseen 2021 data.

## Repository contents

```text
.
├── SeismicNet_RAW_DATA_acquisition.py          # Reads SEISAN/NORDIC S-files and waveform files, extracts raw traces and phase-pick metadata
├── SeismicNet_DATA_SET_generator.py            # Builds structured training tensors and P/S/noise probability labels
├── SeismicNet_DATA_JOIN.py                     # Joins monthly data into a yearly dataset
├── SeismicNet_DATA_YEARS_JOIN.py               # Joins yearly datasets, e.g. 2019 + 2020
├── SeismicNet_METADATA_JOIN_YEARS.py           # Joins metadata used for model evaluation and visualization
├── SeismicNet_architecture.py                  # Defines, trains, and saves the SeismicNet CNN model
├── SeismicNet_model_statistics.py              # Computes phase-picking statistics for different Δt thresholds
├── SeismicNet_results_collection.py            # Generates and stores prediction plots grouped by quality
├── SeismicNet_results_visualization_tools.py   # Exploratory visualization utilities for model predictions
├── TesisTotal.pdf                              # Full thesis report
└── README.md
```

## Data pipeline

The original workflow is organized around SEISAN-style seismic event data:

1. **Raw acquisition**
   - Reads S-files in NORDIC format.
   - Finds the corresponding waveform files.
   - Extracts station/channel traces with valid P and S picks.
   - Resamples every trace to a fixed length of 3000 points.
   - Normalizes each trace using a machine-learning convention similar to
     $$(x - \bar{x}) / (x_{\max} - x_{\min}).$$

2. **Label generation**
   - Converts manual phase arrival times into indices in the 3000-point trace.
   - Builds Gaussian probability distributions centered at the manual P and S picks.
   - Constructs a third noise/background distribution.
   - Saves the final label tensor with shape approximately:

     ```text
     (n_samples, 3000, 3)
     ```

3. **Dataset consolidation**
   - Joins monthly datasets into yearly datasets.
   - Joins 2019 and 2020 into the training set.
   - Keeps metadata for later evaluation and visualization.

4. **Training**
   - Loads the 2019–2020 dataset.
   - Adds the channel dimension so traces have shape:

     ```text
     (3000, 1)
     ```

   - Trains SeismicNet with TensorFlow/Keras.
   - Saves the trained model as an `.h5` file and stores the training history.

5. **Evaluation**
   - Tests the model on January and February 2021 data.
   - Extracts predicted P and S picks from probability peaks.
   - Computes $\Delta t_P$, $\Delta t_S$, and counts correct picks under 1 s, 2 s, and 3 s thresholds.
   - Saves diagnostic figures into `best`, `worst`, and general-result folders.

## Model architecture

The final SeismicNet implementation uses a 1D convolutional encoder-decoder-like architecture followed by a dense projection into three time-dependent probability channels:

```text
Input: (3000, 1)
↓
Conv1D(filters=175, kernel_size=7, activation='relu')
MaxPooling1D(pool_size=9)
Conv1D(filters=125, kernel_size=7, activation='relu')
Conv1DTranspose(filters=20, kernel_size=7, strides=4, activation='relu')
Conv1DTranspose(filters=25, kernel_size=7, strides=2, activation='relu')
Conv1DTranspose(filters=15, kernel_size=8, strides=2, activation='relu')
MaxPooling1D(pool_size=4)
Flatten()
Dense(9000, activation='linear')
Reshape((3000, 3))
Softmax(axis=2)
```

The model is compiled with the Adam optimizer and categorical cross-entropy loss. Training was configured for up to 75 epochs with large batches, and the model is saved as:

```text
CNN_2021_architecture04_v04.h5
```

## Evaluation logic

For each test trace:

1. SeismicNet predicts three distributions: P, S, and noise.
2. Peaks are detected in the P and S distributions.
3. The highest P peak and highest S peak are selected.
4. If the predicted P position appears after the predicted S position, the script swaps them to enforce the expected physical order.
5. The temporal errors are computed as:

$$
\Delta t_P = \left| \frac{(i_P^{\mathrm{pred}} - i_P^{\mathrm{label}})}{d} (t_{\mathrm{end}} - t_{\mathrm{start}}) \right|
$$

$$
\Delta t_S = \left| \frac{(i_S^{\mathrm{pred}} - i_S^{\mathrm{label}})}{d} (t_{\mathrm{end}} - t_{\mathrm{start}}) \right|
$$

where $d = 3000$ is the fixed trace dimension.

The scripts report:

- number of P phases detected within 1, 2, and 3 seconds,
- number of S phases detected within 1, 2, and 3 seconds,
- number of traces with at least one correct phase,
- number of traces with both phases correct,
- mean and standard deviation of $\Delta t_P$ and $\Delta t_S$.

## Thesis results summary

The thesis reports that SeismicNet improved the phase-picking performance obtained in earlier exploratory models. The strongest results were observed in the less noisy January 2021 test set, while February 2021 was more difficult because the data included a seismic swarm and several traces contained multiple events. The conclusions also note that 1D convolutional models outperformed the earlier 2D-convolution experiments for this problem.

For February 2021, the thesis reports the following SeismicNet counts:

| Criterion | P phases | S phases | At least one phase | Both phases |
|---|---:|---:|---:|---:|
| $\Delta t < 1$ s | 763 | 876 | 1172 | 467 |
| $\Delta t < 2$ s | 949 | 1161 | 1376 | 734 |
| $\Delta t < 3$ s | 1039 | 1277 | 1445 | 871 |

These results show that the model learned meaningful seismic arrival patterns, while also revealing limitations in noisy traces, traces with multiple events, and cases where volcanic or swarm activity produced ambiguous probability peaks.

## Requirements

The scripts were developed as research scripts and use absolute paths from the original workstation. To run the project on a new machine, refactor the paths into a configuration file or command-line arguments.

Core Python dependencies:

```text
tensorflow
numpy
scipy
matplotlib
obspy
```

Optional but recommended for a cleaner modern version:

```text
pandas
pyyaml
scikit-learn
jupyter
```

## Suggested setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install tensorflow numpy scipy matplotlib obspy pandas pyyaml scikit-learn jupyter
```

## Suggested modern project structure

The current repository preserves the original thesis research scripts. For a publication-ready and reproducible version, the project could be reorganized as:

```text
seismicnet/
├── configs/
│   └── default.yaml
├── data/
│   ├── raw/              # ignored by Git
│   ├── processed/        # ignored by Git
│   └── README.md         # explains how to obtain/preprocess data
├── models/               # ignored by Git or tracked with Git LFS
├── notebooks/
├── results/
├── src/
│   └── seismicnet/
│       ├── data.py
│       ├── labels.py
│       ├── model.py
│       ├── train.py
│       ├── evaluate.py
│       └── visualize.py
├── tests/
├── README.md
├── requirements.txt
└── pyproject.toml
```

## Reproducibility notes

The original code depends on local absolute paths and private/raw seismic data. For reproducibility, a future version should include:

- a small anonymized sample dataset,
- a clear data dictionary,
- scripts with command-line interfaces,
- fixed random seeds,
- train/validation/test split documentation,
- saved model configuration files,
- environment files such as `requirements.txt` or `environment.yml`,
- evaluation tables generated automatically from raw prediction outputs.

## Roadmap toward a scientific paper

A strong paper can be built around the following claim:

> A 1D convolutional neural network trained on local Guatemalan seismic waveform data can learn probability distributions for P- and S-wave arrivals and provide automatic phase-picking support for routine seismic analysis.

Recommended next steps:

1. Refactor the code into reusable modules.
2. Re-run the full pipeline with fixed seeds and documented splits.
3. Add baseline comparisons against classical picking methods and existing deep-learning pickers where possible.
4. Report precision/recall-style picking metrics in addition to \(\Delta t\) thresholds.
5. Include error analysis for noisy traces, volcanic activity, multi-event traces, and low-SNR examples.
6. Prepare publication figures directly from reproducible scripts.

## Citation

If you use this repository, cite the thesis/project as:

```bibtex
@thesis{medina2021seismicnet,
  title  = {Desarrollo de red neuronal convolucional para análisis de señales sísmicas},
  author = {Medina Chután, Julio Antonio},
  school = {Universidad de San Carlos de Guatemala, Escuela de Ciencias Físicas y Matemáticas},
  year   = {2021},
  type   = {B.Sc. thesis}
}
```

## License

No license file is currently included. Add a license before distributing or accepting external contributions.
