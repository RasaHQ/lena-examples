import argparse
import random
from typing import List, Tuple


from rasa.shared.importers.rasa import RasaFileImporter
from sentence_transformers import InputExample, SentenceTransformer, losses
from sentence_transformers.datasets import SentenceLabelDataset
from sentence_transformers.losses import BatchHardTripletLossDistanceFunction
from sklearn import preprocessing
from torch.utils.data import DataLoader

################################
# FINE TUNE WEIGHTS ONLY
class FinetuneSBERT:
    def __init__(self) -> None:
        self.model = SentenceTransformer("rasa/LaBSE")  # load from HuggingFace repo
        self.head = None
        self.le = preprocessing.LabelEncoder()

    def load_data(self, path: str) -> Tuple[List[str], List[str]]:
        importer = RasaFileImporter(training_data_paths=path)
        td = importer.get_nlu_data()
        texts = []
        intents = []
        random.shuffle(td.training_examples)
        for example in td.training_examples:
            texts.append(example.get("text"))
            intents.append(example.get_full_intent())

        return texts, intents

    def train(
        self,
        sentences: List[str],
        labels: List[str],
        output_path: str,
        samples_per_label: int = 5,
        epochs: int = 5,
        warmup_steps: int = 100,
    ):
        batch_size = 64
        self.le.fit(labels)
        label_ids = self.le.transform(labels)
        train_examples = [
            InputExample(texts=[sentence], label=label_id)
            for sentence, label_id in zip(sentences, label_ids)
        ]
        train_data_sampler = SentenceLabelDataset(
            train_examples, samples_per_label=samples_per_label
        )
        batch_size = min(batch_size, len(train_data_sampler))
        train_dataloader = DataLoader(train_data_sampler, batch_size=batch_size, drop_last=True)

        train_loss = losses.BatchHardTripletLoss(
            model=self.model,
            distance_metric=BatchHardTripletLossDistanceFunction.cosine_distance,
            margin=0.25,
        )
        self.model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=epochs,
            warmup_steps=warmup_steps,
        )

        self.model.save(output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Finetune LaBSE with training data")

    parser.add_argument(
        "--nlu_data_path",
        help="Path to the nlu data. ",
        default="data",
    )

    parser.add_argument(
        "--output_path",
        help="Path to save the finetuned model to. ",
        default="pipeline/pretrained/",
    )
    parser.add_argument(
        "--epochs",
        help="Epochs to train",
        default=5,
    )
    parser.add_argument(
        "--samples_per_label",
        help="Samples per intent label to use for finetuning",
        default=5,
    )
    args = parser.parse_args()
    finetune_weights = FinetuneSBERT()
    train_x, train_y = finetune_weights.load_data(f"{args.nlu_data_path}/")
    finetune_weights.train(
        train_x,
        train_y,
        output_path=args.output_path,
        epochs=int(args.epochs),
        samples_per_label=int(args.samples_per_label) or 5,
    )