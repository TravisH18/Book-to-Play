# Load Model
# Load Dataset
# Run through named entity recognition
# For each PER get name to make list of zero - shot classes
# Zero shot classifier pipeline with PER found + narrator
# If it gets more than just dialog we may need to do the following
# Get All Text relating to a specific character
# Classify into text and dialog where dialog is what they say
# text is just narrator lines relating to the given character

from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer
from datasets import load_dataset, interleave_datasets
import evaluate
import torch
import numpy as np

def compute_metrics(eval_pred):
    metric = evaluate.load("accuracy")
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

def main(train, device):
    dataset_names = ["AlekseyKorshuk/fantasy-books",
                     "AlekseyKorshuk/drama-books",
                     "AlekseyKorshuk/mystery-crime-books",
                     "AlekseyKorshuk/thriller-books",
                     "AlekseyKorshuk/romance-books",
                     "AlekseyKorshuk/fairy-tale-books",
                     "AlekseyKorshuk/erotic-books"]
    ds = []
    for i in range(len(dataset_names)):
        ds.append(load_dataset(i))
    
    dataset = interleave_datasets(ds)
    ner_model_name = "xlm-roberta-large-finetuned-conll03-english"
    tokenizer = AutoTokenizer.from_pretrained(ner_model_name).device(device)
    model = AutoModelForTokenClassification.from_pretrained(ner_model_name).device(device)
    tokenized_text = dataset.map(lambda batch: tokenizer(batch["text"], padding="max_length", truncation=True), batched=True)
    print(tokenized_text.column_names)
    print(tokenized_text.shape)

    ner_pipe = pipeline("ner", model=model, tokenizer=tokenizer, device=device)

    preds = ner_pipe()
    # preds = [
    #     {
    #         "entity": pred["entity"],
    #         "score": round(pred["score"], 4),
    #         "index": pred["index"],
    #         "word": pred["word"],
    #         "start": pred["start"],
    #         "end": pred["end"],
    #     }
    #     for pred in preds
    # ]
    # print(*preds, sep="\n")

    if train:
        training_args = TrainingArguments(
            output_dir="model",
            learning_rate=2e-5,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            num_train_epochs=2,
            weight_decay=0.01,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            push_to_hub=True,
            tf_32=True,
            torch_compile=True
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_text["train"],
            eval_dataset=tokenized_text["test"],
            tokenizer=tokenizer,
            compute_metrics=compute_metrics
        )

        trainer.train()

    # Else for inference on entire dataset
    for batch in tokenized_text: # Maybe change to full text of one row and do zero_shot in batches of full text
        entities = ner_pipe(batch["text"])

        character_labels = []
        for e in entities:
            if e["entity"] == 'I-PER':
                character_labels.append(e["word"]) 
        zero_shot_model_name = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
        classifier = pipeline("zero-shot-classification", model=zero_shot_model_name, device=device)

        seperated_dialog = classifier(batch, candidate_labels=["dialog", "narration"])
        character_dialog = classifier(seperated_dialog["dialog"], candidate_labels=character_labels)

# Classify entire book text
    
if __name__ == '__main__':
    train = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    main(train, device)