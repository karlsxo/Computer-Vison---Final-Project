# Proposal Form — Computer Vision Comparison Project

| Field | Answer |
|---|---|
| **Group number and members** | Group __ : ______________________ (fill in) |
| **Task** | Image classification: the material of one piece of trash (cardboard, glass, metal, paper, plastic, trash/residual) |
| **Classical** | HOG features + linear SVM — HOG and the SVM (squared hinge, our own Adam optimiser) implemented from scratch in NumPy (Dalal & Triggs, 2005) |
| **Trained from scratch** | Our own 4-block CNN in PyTorch (Conv–BN–ReLU–MaxPool ×4, double conv in blocks 3–4, 1.13 M parameters), random initialisation |
| **Pretrained, modern** | MobileNetV3-Large (Howard et al., ICCV 2019), ImageNet weights from torchvision, fine-tuned |
| **Dataset(s) and size** | TrashNet (Thung & Yang, 2016): 2,527 photos, 6 classes → 2,519 after cleaning, one saved 70/15/15 split (1,764 / 379 / 376). Real-world test set: 600 web photos (100/class) from the 12-class Garbage Classification dataset (Kaggle). Extension: 1,655 + 240 further web photos for mixed-domain training (Experiment 6). |
| **Own local dataset? (yes/no, what)** | Optional: phone photos taken by the group in Iloilo, put in `data/own_photos/<class>/`; used for testing only (evaluated automatically). |
| **Expected difficulty or risk** | Class imbalance (trash = 5 %), near-duplicate photos causing train/test leakage, look-alike materials (paper vs cardboard, glass vs clear plastic), and a large gap between studio and real-world photos. Mitigation: duplicate removal and a group-aware saved split, macro-F1 and class weights, a separate real-world test set, and mixed-domain training. |
| **Research question** | Which method best sorts trash by material, and does it still work on messy real-world photos after training on clean ones? |
| **Methods differ in** | hand-designed vs learned features; no prior knowledge vs ImageNet knowledge; 0 vs 1.13 M vs 4.2 M parameters |
| **System** | BIN: a phone app (and a desktop version) that opens on the camera, runs all three methods on the photo and answers with the best method, or "not sure". |
