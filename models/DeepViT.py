from PIL import Image
import torch
from transformers import AutoImageProcessor, ViTForImageClassification
class DeepViT():
    def __init__(self, weights = "artifacts/vit_deepfakes_large.pth"):
        self.model = self.load_model(weights)
        self.process = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
        self.id2label = {0 : 'fake', 1 : "real"}
        self.label2id = {'fake' : 0, 'real' : 1}


    def load_model(self, weights):

        model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224-in21k",
                                                          id2label = self.id2label,
                                                          label2id = self.label2id)
        if weights:
            model.load_state_dict(torch.load(weights))
        return model

    def predict_image(self, img_path, return_index = False):
        img = Image.open(img_path)
        image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
        img = image_processor(img, return_tensors = 'pt')
        with torch.no_grad():
            logits = self.model(**img).logits

        # model predicts one of the 1000 ImageNet classes
        predicted_label = logits.argmax(-1).item()
        if return_index:
            return predicted_label
        else:
            return self.model.config.id2label[predicted_label]

