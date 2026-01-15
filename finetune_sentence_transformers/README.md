This recipe contains two files


## Finetune Sentence transformers

Using `sbert_finetune.py`, you can finetune a sentence transformer such `rasa/LaBSE` by showing few samples of your training data which you have in rasa nlu format.

```
pip install -r requirements.txt

````

```
python sbert_finetune.py --nlu_data_path your-rasa-nlu-training-data --output_path path-to-save-finetuned-weights --epochs numberofepochs --samples_per_label number-of-samplesfrom-each-intent

```

## Sbert featurizer
Since the finetuned weights are generated using sentence transformer, to effectively use them in Rasa, we need a custom featurizer component 

`sbert_featurizer.FinetunedSbert` - This will load torch weights and generate sentence embeddings in torch tensor format which can be fed onto a `LogisiticRegressionClassifier`.

p.s. untested against DIET, and most likely it won't work with it. 


## Conclusion

Finetuning provides a certain performance gain with respect to accuracy and F1 score. Pytorch featurization is much faster than tensorflow and thus training as well as inference time is much shorter.

