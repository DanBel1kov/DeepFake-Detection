from baselines.ViT.ViT_LRP import vit_base_patch16_224 as vit_LRP
from baselines.ViT.ViT_explanation_generator import LRP
import torch
import numpy as np
import cv2
from PIL import Image
from torchvision.transforms import transforms
import matplotlib.pyplot as plt

normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize,
])

def init_model(pretrained = True):
    model = vit_LRP(num_classes=2, pretrained=True).cuda()
    if pretrained:
        model.load_state_dict(torch.load('artifacts/vit_lrp2.pth'))
    return model
class LRPViT():
    """All images should be preprocessed with the transform in the beginning of the file (execpt get_cam and predict methods) """

    def __init__(self, pretrained = True):
        """Setting use_thresholding to True makes LRP red and blue colors, where
         red indicating places that influences that class and blue places indicates different class
         Setting use_thresholding to False makes visualisation in Grad-CAM style (Otsu's method) """
        self.model = init_model(pretrained)
        self.attribution_generator = LRP(self.model)
        self.use_thresholding = True
        self.CLS2IDX = {0: 'Fake', 1: 'Real'}

    def predict(self, img):
        """Main function for prediction. Gets PIL image without transform
           Returns predicted label 0 - fake, 1 - real and probability that photo is fake"""
        img_tr = transform(img)
        pred = self.model(img_tr.unsqueeze(0).cuda())
        prob = torch.softmax(pred, dim=1)
        return prob.argmax().item(), prob[0][0].item()

    def get_cam(self, img, class_index=0):
        """ Main function that gives red and blue regions for predicted image ( red is fake with class_index = 0)
        Input: PIL image and class_index to visualize: 0 - Fake, 1 - Real"""

        img = transform(img)
        transformer_attribution = self.attribution_generator.generate_LRP(img.unsqueeze(0).cuda(), method="transformer_attribution", index=class_index).detach()
        transformer_attribution = transformer_attribution.reshape(1, 1, 14, 14)
        transformer_attribution = torch.nn.functional.interpolate(transformer_attribution, scale_factor=16, mode='bilinear')
        transformer_attribution = transformer_attribution.reshape(224, 224).data.cpu().numpy()
        transformer_attribution = (transformer_attribution - transformer_attribution.min()) / (transformer_attribution.max() - transformer_attribution.min())

        if self.use_thresholding:
            transformer_attribution = transformer_attribution * 255
            transformer_attribution = transformer_attribution.astype(np.uint8)
            ret, transformer_attribution = cv2.threshold(transformer_attribution, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            transformer_attribution[transformer_attribution == 255] = 1

        heatmap = cv2.applyColorMap(np.uint8(255 * transformer_attribution), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        heatmap = heatmap / np.max(heatmap)
        return heatmap


    def show_cam_on_image(self, img, mask):
        heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        cam = heatmap + np.float32(img)
        cam = cam / np.max(cam)
        return cam
    def generate_visualization(self, original_image, class_index=None):
        transformer_attribution = self.attribution_generator.generate_LRP(original_image.unsqueeze(0).cuda(), method="transformer_attribution", index=class_index).detach()
        transformer_attribution = transformer_attribution.reshape(1, 1, 14, 14)
        transformer_attribution = torch.nn.functional.interpolate(transformer_attribution, scale_factor=16, mode='bilinear')
        transformer_attribution = transformer_attribution.reshape(224, 224).data.cpu().numpy()
        transformer_attribution = (transformer_attribution - transformer_attribution.min()) / (transformer_attribution.max() - transformer_attribution.min())

        if self.use_thresholding:
            transformer_attribution = transformer_attribution * 255
            transformer_attribution = transformer_attribution.astype(np.uint8)
            ret, transformer_attribution = cv2.threshold(transformer_attribution, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            transformer_attribution[transformer_attribution == 255] = 1

        image_transformer_attribution = original_image.permute(1, 2, 0).data.cpu().numpy()
        image_transformer_attribution = (image_transformer_attribution - image_transformer_attribution.min()) / (image_transformer_attribution.max() - image_transformer_attribution.min())
        vis = self.show_cam_on_image(image_transformer_attribution, transformer_attribution)
        vis =  np.uint8(255 * vis)
        vis = cv2.cvtColor(np.array(vis), cv2.COLOR_RGB2BGR)
        return vis

    def print_logits(self, predictions, **kwargs):
        prob = torch.softmax(predictions, dim=1)
        class_indices = predictions.data.topk(2, dim=1)[1][0].tolist()
        max_str_len = 0
        class_names = []
        for cls_idx in class_indices:
            class_names.append(self.CLS2IDX[cls_idx])
            if len(self.CLS2IDX[cls_idx]) > max_str_len:
                max_str_len = len(self.CLS2IDX[cls_idx])

        print("Model's prediction: ")
        for cls_idx in class_indices:
            output_string = '\t{} : {}'.format(cls_idx, self.CLS2IDX[cls_idx])
            output_string += ' ' * (max_str_len - len(self.CLS2IDX[cls_idx])) + '\t\t'
            output_string += 'value = {:.3f}\t prob = {:.1f}%'.format(predictions[0, cls_idx], 100 * prob[0, cls_idx])
            print(output_string)












#%%
