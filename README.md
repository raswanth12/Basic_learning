# Whisper-Large-V2 Inference Documentation

This repository demonstrates the process of running inference with OpenAI's **Whisper-Large-V2** model for automatic speech recognition (ASR). The system allows users to transcribe audio from `.wav` files, leveraging the Whisper model's capabilities for accurate transcription in various languages.

---

## Overview

This repository includes:

- Runtime audio input support (specify `.wav` file paths).
- Automatic resampling of audio to the required 16kHz sample rate.
- Configurable inference parameters including beam search, temperature, and token limits.
- Support for distributed inference with multiple GPUs.

---

## Running the Inference

To run the inference, use the following command:

```bash
phisonai2 --env_config /path/to/env_config.yaml --exp_config /path/to/exp_config.yaml
```

**Required Arguments:**

1. `--env_config`: Path to the environment configuration file (YAML).
2. `--exp_config`: Path to the experiment configuration file (YAML).

### Example

```bash
phisonai2 --env_config /home/miphi/Documents/speech_inference/env_config.yaml --exp_config /home/miphi/Documents/speech_inference/exp_config.yaml
```

---

## Features

- **Transcription:** Provides transcription for `.wav` files with optional LoRA fine-tuning support for custom models.
- **Dynamic Input:** Accepts `.wav` file paths via terminal input during runtime.
- **Error Handling:** Gracefully handles input errors and allows quitting with the commands `quit` or `exit`.

---

## Configuration Options

### Command-Line Arguments

| Argument                          | Type   | Default    | Description                                                                 |
|-----------------------------------|--------|------------|-----------------------------------------------------------------------------|
| `--model_name_or_path`            | String | Required   | Path to the pre-trained Whisper model (e.g., `openai/whisper-large-v2`).    |
| `--data_path`                     | List   | None       | Dataset paths for training (not required for inference).                    |
| `--nvme_path`                     | String | Required   | Path for temporary storage or caching.                                      |
| `--local_rank`                    | Int    | 0          | Rank of the GPU for distributed inference.                                  |
| `--num_gpus`                      | Int    | 1          | Number of GPUs to use for distributed inference.                            |
| `--max_new_tokens`                | Int    | 30         | Maximum number of tokens to generate per transcription.                     |
| `--temperature`                   | Float  | 1.0        | Sampling temperature for decoding.                                          |
| `--num_beams`                     | Int    | 3          | Number of beams for beam search decoding.                                   |
| `--top_k`                         | Int    | 50         | Top-k filtering for sampling.                                               |
| `--top_p`                         | Float  | 1.0        | Top-p (nucleus) sampling.                                                   |
| `--precision_mode`                | Int    | 1          | Precision mode for inference (`1` for bf16).                                |

### Environment Configuration (`env_config.yaml`)

Specify global environment settings such as paths, distributed training configurations, and GPU details.

### Experiment Configuration (`exp_config.yaml`)

Define model-specific parameters, data settings, and runtime behavior for the transcription process.

---

## Example Workflow

1. **Start the script:**

   Run the command as shown in the **Running the Inference** section.

2. **Input `.wav` file path:**

   The script will prompt for a `.wav` file path:

   ```
   Enter the path to your .wav audio file:
   ```

   Provide the path to a valid `.wav` file. For example:

   ```
   /path/to/audio/sample.wav
   ```

3. **Transcription Output:**

   The system will process the file and print the transcription:

   ```
   [2024-12-02 14:30:00] [Output transcription]: Hello world!
   ```

4. **Exit the System:**

   To terminate the process, type `quit` or `exit`.

---

## Code Highlights

### Dataset for Real-Time Input

The `RuntimeInputDataset` class supports real-time input of `.wav` files and handles:

- Path input validation.
- Waveform normalization and resampling to 16kHz.

```python
class RuntimeInputDataset(Dataset):
    def __getitem__(self, idx):
        # Path input logic
        # Normalize waveform to [-1, 1]
        # Resample audio to 16kHz
```

### Inference Loop

The `eval` function handles the core inference logic:

- Converts waveforms to Whisper model input features.
- Performs token generation using the model.
- Decodes and prints the transcription.

```python
def eval(args, model: nn.Module, dataloader: DataLoader):
    model.eval()
    for waveform in dataloader:
        # Generate transcription
        output = model.generate(input_features, ...)
        transcription = tokenizer.batch_decode(output, skip_special_tokens=True)
        print(f"Transcription: {transcription}")
```

---

## Supported Models

This implementation is tested with **Whisper-Large-V2** from Hugging Face. To adapt to other Whisper models, modify the `--model_name_or_path` argument.

---

## Known Issues

- Only `.wav` files are supported; other audio formats need conversion beforehand.
- Distributed inference requires `NCCL` backend and properly configured GPU environments.

---

## License

This project is licensed under the MIT License.
