# masks - output of FaceSeg {masks = S.Segment}
# im - output of LRPViT {im = model.get_cam(img)}

def Normalize(SegMask, LRPImage):
    target_size = (224, 224) 
    if LRPImage.shape[0:2] != target_size:
        LRPImage = cv2.resize(LRPImage, target_size)
    if SegMask.shape[0:2] != target_size:
        SegMask = cv2.resize(SegMask, target_size)

    if LRPImage.dtype != np.uint8:
        LRPImage = LRPImage.astype(np.uint8)
    if SegMask.dtype != np.uint8:
        SegMask = SegMask.astype(np.uint8)
    
    return SegMask, LRPImage

def MergeSegLRP(SegMasks, LRPImage):
    LRPImage = LRPImage[:, :, 2]
    PIXEL_COUNT = {x:[] for x in SegMasks}
    for i in SegMasks:
        SegMask = SegMasks[i]
        SegMask, LRPImage = Normalize(SegMask, LRPImage)
      
        mask_lrp_classwise = cv2.bitwise_and(SegMask, LRPImage)

        intensity_sum = np.sum(mask_lrp_classwise)
        PIXEL_COUNT[i].append(intensity_sum)
    return PIXEL_COUNT

result = MergeSegLRP(masks, img)
