from typing import Any, Dict, List, Text, Type
import numpy as np
import torch

from rasa.engine.graph import ExecutionContext, GraphComponent
from rasa.engine.recipes.default_recipe import DefaultV1Recipe
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.nlu.constants import DENSE_FEATURIZABLE_ATTRIBUTES
from rasa.nlu.featurizers.dense_featurizer.dense_featurizer import DenseFeaturizer
from rasa.nlu.tokenizers.tokenizer import Tokenizer
from rasa.shared.nlu.training_data.message import Message
from rasa.shared.nlu.training_data.training_data import TrainingData


@DefaultV1Recipe.register(
    DefaultV1Recipe.ComponentType.MESSAGE_FEATURIZER, is_trainable=False
)
class PyTorchTransformerFeaturizer(DenseFeaturizer, GraphComponent):
    """Generic featurizer for PyTorch transformer models (supports LoRA/PEFT)."""

    @classmethod
    def required_components(cls) -> List[Type]:
        return [Tokenizer]

    @staticmethod
    def get_default_config() -> Dict[Text, Any]:
        return {
            **DenseFeaturizer.get_default_config(),
            "model_name": "xlm-roberta-base",  # Base model name (for PEFT) or full model
            "model_path": None,  # Path to fine-tuned model or LoRA adapters
            "use_peft": False,  # Set True if using LoRA/PEFT adapters
            "device": None,  # Auto-detect if None
            "max_length": 512,
            "pooling": "mean",  # "mean", "cls", "max", or "last" (for decoder models)
            "model_type": "encoder",  # "encoder" or "decoder"
        }

    def __init__(
        self,
        config: Dict[Text, Any],
        execution_context: ExecutionContext,
    ) -> None:
        super().__init__(execution_context.node_name, config)
        self._load_model()

    def _load_model(self) -> None:
        from transformers import AutoTokenizer, AutoModel

        # Determine device
        if self._config["device"]:
            self.device = torch.device(self._config["device"])
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model_path = self._config["model_path"] or self._config["model_name"]

        if self._config.get("use_peft", False):
            # LoRA/PEFT fine-tuned model
            from peft import AutoPeftModel

            self.model = AutoPeftModel.from_pretrained(
                model_path,
                torch_dtype=torch.float32,
            )
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        else:
            # Standard HuggingFace model
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModel.from_pretrained(model_path)

        self.model.to(self.device)
        self.model.eval()

        # Get hidden size for reference
        self.hidden_size = self.model.config.hidden_size

    @classmethod
    def create(
        cls,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
    ) -> "PyTorchTransformerFeaturizer":
        return cls(config, execution_context)

    @classmethod
    def load(
        cls,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
    ) -> "PyTorchTransformerFeaturizer":
        return cls(config, execution_context)

    @classmethod
    def validate_config(cls, config: Dict[Text, Any]) -> None:
        pass

    @staticmethod
    def required_packages() -> List[Text]:
        return ["transformers", "torch"]

    def _get_embeddings(self, text: Text) -> tuple[np.ndarray, np.ndarray]:
        """Get sequence and sentence embeddings from the model."""
        with torch.no_grad():
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=self._config["max_length"],
                padding=True,
            ).to(self.device)

            outputs = self.model(**inputs)
            hidden_states = outputs.last_hidden_state  # [1, seq_len, hidden_dim]

            # Sequence features (per-token) - for NER
            sequence_features = hidden_states.squeeze(0).cpu().numpy()  # [seq_len, hidden_dim]

            # Sentence features (pooled) - for intent classification
            sentence_features = self._pool_embeddings(hidden_states, inputs)

            return sequence_features, sentence_features

    def _pool_embeddings(
        self, hidden_states: torch.Tensor, inputs: Dict[Text, torch.Tensor]
    ) -> np.ndarray:
        """Pool token embeddings into a single sentence embedding."""
        pooling = self._config["pooling"]
        model_type = self._config.get("model_type", "encoder")

        if pooling == "cls" and model_type == "encoder":
            # Use [CLS] token (first token)
            pooled = hidden_states[:, 0, :]
        elif pooling == "last" or (pooling == "cls" and model_type == "decoder"):
            # Use last token (for decoder models like GPT, LLaMA)
            attention_mask = inputs["attention_mask"]
            seq_lengths = attention_mask.sum(dim=1) - 1
            batch_indices = torch.arange(hidden_states.size(0), device=self.device)
            pooled = hidden_states[batch_indices, seq_lengths, :]
        elif pooling == "max":
            # Max pooling over sequence
            attention_mask = inputs["attention_mask"].unsqueeze(-1)
            hidden_states = hidden_states.masked_fill(attention_mask == 0, -1e9)
            pooled = hidden_states.max(dim=1).values
        else:  # mean (default)
            # Mean pooling over non-padded tokens
            attention_mask = inputs["attention_mask"].unsqueeze(-1).float()
            summed = (hidden_states * attention_mask).sum(dim=1)
            pooled = summed / attention_mask.sum(dim=1).clamp(min=1e-9)

        return pooled.cpu().numpy()  # [1, hidden_dim]

    def process_training_data(self, training_data: TrainingData) -> TrainingData:
        for example in training_data.training_examples:
            self._featurize_message(example)
        return training_data

    def process(self, messages: List[Message]) -> List[Message]:
        for message in messages:
            self._featurize_message(message)
        return messages

    def _featurize_message(self, message: Message) -> None:
        for attribute in DENSE_FEATURIZABLE_ATTRIBUTES:
            text = message.get(attribute)
            if text:
                sequence_features, sentence_features = self._get_embeddings(text)
                self.add_features_to_message(
                    sequence=sequence_features,
                    sentence=sentence_features,
                    attribute=attribute,
                    message=message,
                )