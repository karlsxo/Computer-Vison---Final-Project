# Proposal Form — Computer Vision Comparison Project

| Field | Your answer |
|---|---|
| **Group number and members** | Group __ : (your group number and every member's full name) |
| **Task** | Image classification: identify the material of one piece of trash from a photo (cardboard, glass, metal, paper, plastic, trash/residual) |
| **classical** | HOG features + linear SVM (Dalal & Triggs, 2005). HOG and the SVM are implemented from scratch in NumPy. |
| **trained from scratch** | Small CNN: our own 4-block CNN in PyTorch (Conv–BatchNorm–ReLU–MaxPool × 4), trained from random weights |
| **pretrained, modern** | MobileNetV3-Large (Howard et al., 2019) with ImageNet weights from torchvision, fine-tuned on our data |
| **Dataset(s) and size** | TrashNet (Thung & Yang, 2016): 2,527 photos, 6 classes, split 70/15/15 into train, validation and test (the same split for all three methods). Extra real-world test set: 600 web photos (100 per class) from the Garbage Classification dataset (Kaggle). |
| **Own local dataset? (yes/no, what)** | No. Optional: phone photos of trash taken in Iloilo, used for testing only. |
| **Expected difficulty or risk** | Class imbalance (few "trash" images); duplicate photos causing train/test leakage; look-alike materials (paper vs cardboard, glass vs clear plastic); studio photos differ from real-world photos. Mitigation: remove duplicates before splitting, macro-F1 and class weights, a separate real-world test set. |
