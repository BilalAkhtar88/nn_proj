# Global Processing Latency Checker

This script, `check_global_processing.py`, is designed to verify whether an enhancing model introduces latency beyond the allowed limit. It ensures that the enhanced audio files processed by the model comply with the latency constraints set by the organizers.

---

## Overview

The script compares pairs of audio files (`*1.wav` and `*2.wav`) in a specified directory to check for:
1. **Latency Compliance**: Ensures that the latency introduced by the enhancing model does not exceed **20ms**.
2. **Sample Rate Consistency**: Verifies that the sample rate of the processed audio remains at the required **48kHz**.
3. **Length Consistency**: Confirms that the length of the processed audio files matches the original input.

---

## How It Works

1. The script reads pairs of audio files (`*1.wav` and `*2.wav`) from the specified directory.
2. It checks the following:
   - The sample rate of both files is **48kHz**.
   - The lengths of the two files are identical.
3. It calculates the absolute difference between the two audio signals and identifies the first index where the difference is non-zero.
4. If the latency (calculated as the difference between the midpoint of the audio and the first non-zero index) exceeds **20ms**, the script raises an error.

---

## Usage

### Command
Run the script with the following command:
```bash
python3 [check_global_processing.py](http://_vscodecontentref_/0) --path <path-to-enhanced-audio-directory>