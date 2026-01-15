# PyTorch Transformer Featurizer for Rasa

A custom Rasa featurizer that integrates any HuggingFace transformer (including LoRA/PEFT fine-tuned models) with DIET or LogisticRegressionClassifier.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Place `pytorch_transformer_featurizer.py` in your project, then reference in `config.yml`:

```yaml
pipeline:
  - name: WhitespaceTokenizer
  - name: your_module.PyTorchTransformerFeaturizer
    model_path: "/path/to/model"
    use_peft: true  # if using LoRA adapters
  - name: DIETClassifier  # or LogisticRegressionClassifier
```

## Config Options

| Option | Default | Description |
|--------|---------|-------------|
| model_name | xlm-roberta-base | Base model (required for PEFT) |
| model_path | None | Path to fine-tuned model or adapters |
| use_peft | false | Enable for LoRA/PEFT models |
| pooling | mean | mean, cls, max, or last |
| model_type | encoder | encoder or decoder |
| max_length | 512 | Max token sequence length |
| device | auto | cuda, cpu, or auto-detect |

## Supported Models

* Encoders: BERT, RoBERTa, XLM-RoBERTa, DeBERTa, ELECTRA, LaBSE, mBERT
* Decoders: GPT-2, LLaMA, Mistral (use model_type: decoder, pooling: last)
* Fine-tuning: Any model with LoRA/PEFT adapters
