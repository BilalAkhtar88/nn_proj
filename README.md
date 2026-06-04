# Deep Noise Suppression (DNS) Project

This project is designed for students to explore and train neural network models for **Deep Noise Suppression (DNS)**. The objective is to generate noisy audio samples, train a DNS model to suppress noise, and evaluate the model's performance using various metrics.

---

## Project Overview

### 1. **Dataset**
The dataset directory contains:
- **Clean Speech Data**: Located in `dataset/clean_speech/` with subdirectories for different types of speech (e.g., emotional, read speech).
- **Noise Data**: Located in `dataset/noise/`.

These datasets are used to synthesize noisy audio samples for training.

### 2. **Synthesizer**
The `synthesizer.py` script demonstrates how to synthesize noisy audio samples and corresponding clean target speech for training. It uses the clean speech and noise datasets to create realistic noisy audio with various distortions. 

However, this script is not a complete implementation and does not include a data loader for training. Students are encouraged to extend this script by implementing a proper data loader to feed the synthesized data into their training pipeline.

### 3. **Baseline Model**
The `baseline/` directory contains:
- A pre-trained baseline DNS model (`dec-baseline-model-icassp2022.onnx`).
- The `enhance.py` script to enhance noisy audio using the baseline model.

The baseline model is provided to help students understand what a DNS model architecture could look like. Students are encouraged to:
1. Open the ONNX model in a model viewer (e.g., [Netron](https://netron.app/)) to analyze its architecture.
2. Implement the model in PyTorch.
3. Train the implemented model using the synthesized dataset.
4. Verify the results by comparing the performance of the trained model with the baseline.

### 4. **Evaluation Metrics**
Two scripts are provided to evaluate the performance of DNS models:

- **`sigmos.py`**: Implements the SigMOS metric for evaluating speech quality based on ITU-T P.804. For more details, refer to the [SigMOS paper](https://arxiv.org/pdf/2309.07385).
- **`dnsmos_local.py`**: Computes DNSMOS metrics (SIG, BAK, OVRL) for evaluating the quality of enhanced audio based on ITU-T P.835. For more details, refer to the [DNSMOS paper](https://arxiv.org/pdf/2110.01763).

### 5. **Training**
Students are expected to:
- Use the synthesized noisy audio and clean targets to train their own DNS models.
- Compare their models against the baseline using the provided evaluation scripts.

---

## Environment Setup

Follow these steps to set up the environment:



### 1. **Install Dependencies**
Create a Python virtual environment and install the required packages:

#### Using `venv`:
```bash
python3 -m venv dns_env
source dns_env/bin/activate
pip install -r requirements.txt
```

#### Using `conda`:
```bash
conda create -n dns_env python=3.8 -y
conda activate dns_env
pip install -r requirements.txt
```

---

## How to Begin

### 1. **Generate Training Data**
Use the `synthesizer.py` script to generate noisy audio and clean targets:
```bash
python synthesizer.py
```

### 2. **Train Your Model**
- Use the generated noisy and clean audio to train your DNS model.
- You can use any deep learning framework (e.g., PyTorch, TensorFlow) for training.

### Synthesizer Overview

The synthesizer.py script is designed to generate training data for deep noise suppression (DNS) models. It creates noisy audio samples (`nearend.wav`), clean target speech (`target.wav`), and mic-distorted audio (`mic.wav`) by combining clean speech and noise datasets with various distortions and augmentations.

#### Key Features:
1. **Clean Speech and Noise Mixing**: Combines clean speech with noise at specified Signal-to-Noise Ratios (SNRs).
2. **Distortions**: Applies microphone distortions, bandpass filtering, and other augmentations to simulate real-world conditions.
3. **Normalization**: Normalizes audio to a target volume level.
4. **Configurable**: The behavior of the synthesizer is controlled via a YAML configuration file (`synthesizer_config.yaml`).

#### How to Use:
1. **Prepare the Configuration**: Modify synthesizer_config.yaml to specify datasets, distortions, and augmentation parameters.
2. **Run the Script**:
   ```bash
   python synthesizer.py
   ```
   This will generate `target.wav`, `nearend.wav`, and `mic.wav` in the current directory.

---

### Synthesizer Configuration (`synthesizer_config.yaml`)

The configuration file defines the behavior of the synthesizer, including dataset paths, augmentation probabilities, and distortion parameters.

#### Key Sections:

1. **General Parameters**:
   - `onlinesynth_duration`: Duration (in seconds) of the generated audio samples.
   - `onlinesynth_sampling_rate`: Sampling rate for the generated audio (e.g., 16,000 Hz).
   - `onlinesynth_resampling_type`: Resampling algorithm (e.g., `fft`).

2. **Datasets**:
   - `onlinesynth_nearend_datasets`: Defines clean speech datasets.
     ```yaml
     onlinesynth_nearend_datasets:
       emotional_speech:
         weight: 1.0
         dir: 'dataset/clean_speech/emotional_speech'
       read_speech:
         weight: 1.0
         dir: 'dataset/clean_speech/read_speech'
     ```
     - `weight`: Relative probability of selecting this dataset.
     - `dir`: Path to the dataset directory containing `.wav` files.

   - `onlinesynth_nearend_noises`: Defines noise datasets.
     ```yaml
     onlinesynth_nearend_noises:
       noise_dataset_1:
         weight: 1.0
         dir: 'dataset/noise'
     ```

3. **Distortions**:
   - `onlinesynth_mic_distortions`: Defines microphone distortions and their parameters.
     ```yaml
     onlinesynth_mic_distortions:
       distortion_types:
         param_ranges:
           mic_bandpass:
             likelihood: 0.2
             low_freq: [10, 100]
             high_freq: [2000, 4000]
             order: 7
     ```
     - `likelihood`: Probability of applying the distortion.
     - `low_freq`/`high_freq`: Frequency range for bandpass filtering.
     - `order`: Filter order.

4. **Noise and SNR**:
   - `onlinesynth_nearend_prop_noisy`: Probability of adding noise to the clean speech.
   - `onlinesynth_nearend_snr_interval`: SNR range (in dB) for mixing clean speech and noise.
   - `onlinesynth_nearend_prop_add_gaussian_ne_noise`: Probability of adding Gaussian noise.

5. **Volume Normalization**:
   - `onlinesynth_nearend_normalize_volume`: Target volume level (in dB) for normalization.

6. **Augmentations**:
   - Various augmentations, such as room effects, bad microphone response, and ADC effects, can be configured under `onlinesynth_mic_distortions`.

---

### Output Files

- **`target.wav`**: Clean speech without noise or distortions.
- **`nearend.wav`**: Clean speech mixed with noise at a specified SNR.
- **`mic.wav`**: Nearend signal with additional microphone distortions.

This synthesizer provides a flexible framework for generating realistic training data for DNS models.

### 3. **Evaluate Your Model**
- Use `sigmos.py` or `dnsmos_local.py` to evaluate the quality of your enhanced audio.

### Using `dnsmos_local.py`

The `dnsmos_local.py` script is an example of how to process audio files and compute DNSMOS metrics (SIG, BAK, OVRL) for evaluating the quality of enhanced audio. These metrics are based on ITU-T P.835 and provide insights into the signal quality (SIG), background noise quality (BAK), and overall quality (OVRL) of the audio.

#### Example Usage
To evaluate a directory of audio files and save the results to a CSV file, run the following command:
```bash
python dnsmos_local.py -t <path-to-audio-directory> -o results.csv
```

#### Command-Line Arguments
- `-t` or `--testset_dir`: Path to the directory containing `.wav` audio files to be evaluated.
- `-o` or `--csv_path`: Path to save the output CSV file containing the evaluation results.

#### Output
The script generates a CSV file (if the `-o` argument is provided) with the following columns:
- `filename`: Path to the evaluated audio file.
- `len_in_sec`: Length of the audio file in seconds.
- `sr`: Sampling rate of the audio file.
- `OVRL_raw`: Raw overall quality score.
- `SIG_raw`: Raw signal quality score.
- `BAK_raw`: Raw background noise quality score.
- `OVRL`: Adjusted overall quality score.
- `SIG`: Adjusted signal quality score.
- `BAK`: Adjusted background noise quality score.
- `P808_MOS`: MOS score based on ITU-T P.808.

#### Notes
- The script uses ONNX models to compute the metrics. Ensure the required models (`sig_bak_ovr.onnx` and `model_v8.onnx`) are available in the appropriate directories (`DNSMOS` and `pDNSMOS`).
- This script is an example of how to process audio files and obtain evaluation metrics. Students can use it as a reference to implement their own evaluation pipelines or extend its functionality as needed.

### Using `sigmos.py`

The `sigmos.py` script is an example of how to process audio files and compute SigMOS metrics for evaluating speech quality based on ITU-T P.804. These metrics provide insights into various aspects of audio quality, such as coloration, distortion, loudness, noise, reverberation, signal quality, and overall quality.

#### Example Usage
To evaluate an audio file using the SigMOS estimator, follow these steps:

1. **Prepare the Model**:
   - Ensure the required ONNX model (`model-sigmos_1697718653_41d092e8-epo-200.onnx`) is available in the `SIGMOS` directory.

2. **Run the Script**:
   Modify the script or use the provided sample code to evaluate an audio file. For example:
   ```python
   from sigmos import SigMOS

   model_dir = "./SIGMOS"
   sigmos_estimator = SigMOS(model_dir=model_dir)

   # Input audio data (must have a sampling rate of 16 kHz, or specify the sampling rate for resampling)
   sampling_rate = 16_000
   audio_data = np.random.rand(5 * sampling_rate)  # Replace with actual audio data
   result = sigmos_estimator.run(audio_data, sr=sampling_rate)
   print(result)
   ```

3. **Analyze Results**:
   The output will be a dictionary containing the following metrics:
   - `MOS_COL`: Metric for coloration.
   - `MOS_DISC`: Metric for distortion.
   - `MOS_LOUD`: Metric for loudness.
   - `MOS_NOISE`: Metric for noise.
   - `MOS_REVERB`: Metric for reverberation.
   - `MOS_SIG`: Metric for signal quality.
   - `MOS_OVRL`: Overall quality metric.

#### Notes
- The input audio must have a sampling rate of 16 kHz. If the sampling rate is different, the script will resample the audio internally.
- This script is an example of how to process audio files and obtain SigMOS metrics. Students can use it as a reference to implement their own evaluation pipelines or extend its functionality as needed.
- For more details on the SigMOS metric, refer to the [SigMOS paper](https://arxiv.org/pdf/2309.07385).

#### Sample Output
The script will output a dictionary like the following:
```python
{
    'MOS_COL': 4.5,
    'MOS_DISC': 4.2,
    'MOS_LOUD': 4.8,
    'MOS_NOISE': 4.3,
    'MOS_REVERB': 4.1,
    'MOS_SIG': 4.6,
    'MOS_OVRL': 4.4
}
```
This provides a detailed breakdown of the audio quality across various dimensions.

### 4. **Compare with Baseline**
- Use the `baseline/enhance.py` script to enhance noisy audio with the baseline model:
```bash
python enhance.py --data_dir <path-to-noisy-audio> --output_dir <path-to-output>
```
- Compare the results of your model with the baseline using the evaluation metrics.

---

## Directory Structure

```
NNP_Track02/
├── dataset/
│   ├── clean_speech/
│   │   ├── emotional_speech/
│   │   ├── read_speech/
│   │   └── VocalSet_48kHz_mono/
│   └── noise/
├── synthesizer.py          # Script to generate noisy audio for training
├── synthesizer_config.yaml # Configuration for the synthesizer
├── sigmos.py               # Script for SigMOS evaluation
├── dnsmos_local.py         # Script for DNSMOS evaluation
├── baseline/
│   ├── enhance.py          # Script to enhance audio using the baseline model
│   └── dec-baseline-model-icassp2022.onnx # Pre-trained baseline model
└── README.md               # Project documentation
```
