import os
from typing import Any, Dict, List, Text

from rasa.engine.graph import ExecutionContext, GraphComponent
from rasa.engine.recipes.default_recipe import DefaultV1Recipe

from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.nlu.featurizers.featurizer import Featurizer
from rasa.shared.nlu.training_data.message import Message
from rasa.shared.nlu.training_data.training_data import TrainingData
from sentence_transformers import SentenceTransformer


@DefaultV1Recipe.register(DefaultV1Recipe.ComponentType.MESSAGE_FEATURIZER, is_trainable=False)
class FinetunedSbert(Featurizer, GraphComponent):
    def __init__(
        self,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
    ) -> None:
        super(FinetunedSbert, self).__init__(execution_context.node_name, config)
        self.loaded_weights = self._get_finetuned_weights(config.get("model_name"), config.get("model_storage"))
        
    
    def _get_finetuned_weights(self, model_name, model_storage):
        """
        If model storage is not huggingface but internal finetuned weights, send full path
        """
        if model_storage == "huggingface":
            return SentenceTransformer(model_name)
        else:
            model_path = model_name
            tar_name = os.path.basename(model_path)
            if os.path.exists(model_path):
                weights = SentenceTransformer(tar_name)
        return weights

    @classmethod
    def validate_config(cls, config: Dict[Text, Any]) -> None:
        """Validates the configuration."""
        pass

    @classmethod
    def create(
        cls,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
    ):
        """
        Loads the model specified in the config.
        """
        return cls(config, model_storage, resource, execution_context)

    def process_training_data(self, training_data: TrainingData) -> TrainingData:
        self._get_features(training_data.training_examples)
        return training_data

    def _get_features(self, messages: List[Message]) -> Message:
        messages = [m for m in messages if m.get("text") is not None]
        features = self.loaded_weights.encode([m.get("text") for m in messages])
        # Split the features into sentence and sequence vectors
        for idx, m in enumerate(messages):
            self.add_features_to_message(
                sequence=None,
                sentence=features[idx],
                attribute="text",
                message=m,
            )

    def process(self, messages: List[Message]) -> List[Message]:
        """Processes messages by computing tokens and dense features."""
        for message in messages:
            sentence_feature = self.loaded_weights.encode(message.get("text"))
            self.add_features_to_message(
                sequence=None,
                sentence=sentence_feature,
                attribute="text",
                message=message,
            )
        return messages