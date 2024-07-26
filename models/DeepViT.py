from PIL import Image
import torch
from transformers import (
    AutoImageProcessor,
    ViTForImageClassification,
    TrainingArguments,
    Trainer,
)
from sklearn.metrics import accuracy_score
import numpy as np


class DeepViT:
    def __init__(self, weights="artifacts/vit_deepfakes_large.pth"):
        self.id2label = {0: "fake", 1: "real"}
        self.label2id = {"fake": 0, "real": 1}
        self.model = self.load_model(weights)
        self.process = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
        self.trainer = None

    def load_model(self, weights):
        model = ViTForImageClassification.from_pretrained(
            "google/vit-base-patch16-224-in21k",
            id2label=self.id2label,
            label2id=self.label2id,
        )
        if weights:
            if torch.cuda.is_available():
                model.load_state_dict(torch.load(weights))
            else:
                model.load_state_dict(
                    torch.load(weights, map_location=torch.device("cpu"))
                )

        return model

    def predict_file(self, img_path, return_probability=False):
        img = Image.open(img_path)
        return self.predict_image(img, return_probability)

    def predict_image(self, img, return_probability=False):
        image_processor = AutoImageProcessor.from_pretrained(
            "google/vit-base-patch16-224"
        )
        img = image_processor(img, return_tensors="pt")
        with torch.no_grad():
            logits = self.model(**img).logits

        # model predicts one of the 1000 ImageNet classes
        predicted_label = logits.argmax(-1).item()
        if return_probability:
            return logits.softmax(1)[0]
        else:
            return self.model.config.id2label[predicted_label]

    def set_trainer(self, train_dataset, test_dataset, epoches=3, train_batch_size=128):
        metric_name = "accuracy"

        args = TrainingArguments(
            "test-deefakes_large",
            save_strategy="epoch",
            evaluation_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=train_batch_size,
            per_device_eval_batch_size=4,
            num_train_epochs=epoches,
            weight_decay=0.01,
            load_best_model_at_end=True,
            metric_for_best_model=metric_name,
            logging_dir="logs",
            remove_unused_columns=False,
        )

        trainer = Trainer(
            self.model,
            args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            compute_metrics=self.compute_metrics,
            data_collator=self.collate_fn,
            tokenizer=self.process,
        )

        return trainer

    def train(self):
        if self.trainer is None:
            raise RuntimeError("Trainer is not set")
        self.trainer.train()

    def predict_batch(self, dataset):
        if self.trainer is None:
            raise RuntimeError("Trainer is not set")
        return self.trainer.predict(dataset)

    def compute_metrics(self, eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return dict(accuracy=accuracy_score(predictions, labels))

    def collate_fn(examples):
        pixel_values = torch.stack([example[0] for example in examples])
        labels = torch.tensor([example[1] for example in examples])
        return {"pixel_values": pixel_values, "labels": labels}
