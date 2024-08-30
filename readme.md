# TiniestStoryteller

TiniestStoryteller is a project aimed at building a custom language model (LLM) that can be trained on standard desktop PCs or laptops. It challenges the trend of requiring high-end computational resources, offering an accessible approach to language model training and text generation.

## Features

- **Custom LLM Training**: Train language models on modest hardware.
- **Efficient Data Sampling**: Implement efficient data sampling techniques to optimize training.
- **Transformer Model Architecture**: Define and customize transformer-based models.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Arian-Urdu/TiniestStoryteller.git
   ```
2. Navigate to the project directory:
   ```bash
   cd TiniestStoryteller
   ```
3. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

## Files Overview

- **train.py:** Script to train the language model.
- **generate.py:**: Script to generate text based on a trained model.
- **transformer_model.py**: Defines the transformer model architecture.

## Quick Start

Download a already small pretrained checkpoint from: [Here](https://drive.google.com/file/d/1XXfv5G7vwF2VKTRDWPI1IIELUWp_Uc-h/view?usp=drive_link).

Add the downloaded .pth file in output folder.

Now generate text with the downloaded model running `generate.py`.

Example_output:
![Example_output](https://github.com/user-attachments/assets/b0b464b0-4fc5-46fd-ac19-2d98096c5c46)


## Usage

### Data

Download the full preprocessed and filtered by llama dataset from: (TODO).

Add downloaded files to data folder.

### Config

Based on hardware change `batch_size` and `block_size` in the `config.py` file, here you can also configure wandb for logging purposses.
You can also expermient with different Transformer sizes and other training hyperparameters.
Otherwise the default settings are a resonable starting choice.

### Training

Train the model running `train.py`.
Once finished the model will be saved with a timestamp in the output folder.

### Text Generation

Adjust the path in `generate.py` by changing:

```bash
   checkpoint = torch.load('YOUR_MODEL_PATH_HERE')
```

Generate text with the now trained model running `generate.py`.
