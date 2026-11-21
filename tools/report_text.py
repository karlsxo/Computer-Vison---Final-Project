"""Written analysis for the report. Numbers in {braces} are filled from the result
files by build_report.py, so the text always matches the measured results."""

TEXT = {
"abstract": """We compare three families of computer-vision methods on sorting a photo of one piece of trash into
cardboard, glass, metal, paper, plastic or residual trash: a classical pipeline (HOG features and a linear SVM,
both implemented from scratch in NumPy), a 4-block CNN designed and trained from scratch, and an ImageNet-pretrained
MobileNetV3-Large fine-tuned on the same data. All methods use one leakage-safe split of TrashNet (duplicates and
near-duplicates handled before splitting), identical metric code and an identical timing protocol. On the held-out
test split the pretrained model reaches <b>{C_acc:.1f}%</b> accuracy (macro-F1 {C_f1:.3f}), the scratch CNN {B_acc:.1f}% and
HOG + SVM {A_acc:.1f}%. On 600 unseen real-world web photos every method degrades, but the pretrained model keeps the
most ({C_rw:.1f}% vs. {B_rw:.1f}% and {A_rw:.1f}%). Pretraining also dominates when only 10 images per class are
available. We built a web application that runs all three methods, draws their predictions on the photo and
recommends MobileNetV3 by default.""",

"intro": """<p>Segregating waste by material is the first step of recycling and is required at source by the Philippine
Ecological Solid Waste Management Act (RA 9003), yet in barangay materials recovery facilities, schools and cleanup
drives it is done by hand, slowly and inconsistently. A camera-based tool that names the material of an item could
support these efforts. The problem is also a good test bed for computer vision: materials differ in colour, texture
and shine, several pairs look alike (paper vs. cardboard, glass vs. clear plastic), and studio photos used for
training look very different from real photos.</p>
<p>This project asks which family of methods is best for the job: carefully hand-designed features, a small network
learned from scratch on limited data, or a large network pretrained on ImageNet and adapted. We compare one method of
each family on identical data, splits and metrics, including speed, size and training cost, and test how well each
survives a shift from clean training photos to real-world photos.</p>
<p><b>Objectives.</b> (1) choose and justify three well-matched methods; (2) implement HOG and the SVM ourselves and
verify them with sanity checks; (3) design, train and validate our own CNN with learning curves and overfitting
controls; (4) fine-tune a pretrained MobileNetV3 with correct preprocessing on the same split; (5) report top-1
accuracy, macro-F1, confusion matrices, runtime, parameters and training time; (6) explain with evidence when and why
each method succeeds or fails; (7) build an application that runs all three methods and recommends one.</p>""",

"data": """<p><b>Sources.</b> TrashNet (Thung &amp; Yang, 2016: 2,527 photos of single items on a white/grey background,
512&times;384) is the train/validation/test data. A second, independent dataset, the web-scraped 12-class
<i>Garbage Classification</i> set (Kaggle), is used <i>only</i> as an unseen real-world test set: its three glass
classes are mapped to glass, cardboard/paper/metal/plastic/trash keep their names, and battery, biological, clothes
and shoes are excluded. 100 images per class were sampled with a fixed seed (600 images).</p>
<p><b>Cleaning.</b> Exact duplicates were found by MD5: three files are byte-identical but stored under two classes
(e.g. <code>glass176.jpg</code> = <code>plastic152.jpg</code>); since the true label is unknowable, all six copies were dropped.
TrashNet also contains the same object photographed more than once, which would leak between train and test. A
64-bit difference hash, our first attempt, was rejected: on the plain backgrounds it linked 1,053 pairs of
<i>different-class</i> images. We instead compute a <i>detail signature</i> (64&times;64 grayscale minus a Gaussian blur,
which removes the shared studio lighting, then zero-mean/unit-norm) and link pairs whose correlation exceeds 0.80, a
threshold chosen by inspecting image pairs in several similarity bands. Linked images form groups (29 groups of &ge;2
images); each group is assigned to a single split. A seeded, stratified, group-aware 70/15/15 split is saved to
<code>splits/split.json</code> and loaded by every method. A self-check asserts that no validation or test image has a
similarity above 0.80 to any training image (the maximum is 0.756). Finally, 490 of the sampled web images turned out to
be copies of TrashNet photos (similarity 1.0) and were excluded from the real-world set.</p>""",

"methodA": """<p><b>HOG</b> (Dalal &amp; Triggs, 2005) is implemented in NumPy: BT.601 grayscale; gradients with the
[&minus;1, 0, 1] kernel; magnitude and unsigned orientation (0&ndash;180&deg;); 8&times;8-pixel cells with 9-bin histograms
weighted by magnitude, each vote split linearly between the two nearest bins; overlapping 2&times;2-cell blocks with
L2-Hys normalisation (clip 0.2). A 128&times;128 image gives a 8,100-D descriptor in about 4 ms. HOG describes local edge
orientation, i.e. shape, and deliberately discards colour and absolute intensity.</p>
<p><b>Linear SVM</b>, also ours: one-vs-rest with all six classifiers optimised jointly, objective
&lambda;/2&middot;&#8214;W&#8214;&sup2; + mean<sub>i</sub> &Sigma;<sub>k</sub> c<sub>i</sub>&middot;max(0, 1 &minus; t<sub>ik</sub>(x<sub>i</sub>&middot;w<sub>k</sub>+b<sub>k</sub>))&sup2;
(the squared-hinge "L2-loss" SVM used by LIBLINEAR), class-balanced weights c<sub>i</sub>, standardised features, and our
own full-batch Adam optimiser with cosine learning-rate decay (300 iterations). Profiling showed that mixing float32
data with float64 weights made NumPy up-cast the data matrix on every product; keeping everything in float32 made each
fit 12&times; faster. Cell size {{8, 16}}, soft/hard binning and &lambda; &isin; {{1, 3, 10, 30, 100}} were selected on the validation
split (best: 8-px cells, soft binning, &lambda; = 30; the first grid's optimum lay on its edge, so it was extended).
An extension that appends a 192-bin HSV colour histogram (our own RGB&rarr;HSV) is reported separately as a variant.</p>
<p><b>Sanity checks</b> (all passed):</p>""",

"methodB": """<p>The network is our own design: four Conv-BN-ReLU-MaxPool blocks with 32/64/128/256 channels, a second
convolution in blocks 3 and 4 (where it enlarges the receptive field cheaply on small feature maps; doubling every block
would nearly double the compute: 209 vs. 124 s per CPU epoch), global average pooling, dropout 0.4 and a linear layer;
1.13 M parameters with He initialisation. Training uses class-weighted cross-entropy with label smoothing, AdamW,
warm-up plus cosine decay, batch 32 and our own per-image augmentation (one random affine warp combining zoom-crop,
rotation and horizontal/vertical flips, then colour jitter, tint and random erasing). Early stopping on validation
macro-F1 restores the best checkpoint.</p>
<p><b>Over- vs. under-fitting.</b> A first run with full-strength augmentation, smoothing 0.1 and weight decay 0.05
reached only 47% training and 53% validation accuracy after 15 epochs, both rising very slowly: it was
<i>under</i>-fitting, so we stopped it. The final run uses augmentation strength 0.6 (zoom 70&ndash;100%, rotation &plusmn;15&deg;,
jitter &plusmn;18%), smoothing 0.05, weight decay 0.02 and learning rate 3&middot;10<sup>&minus;3</sup> for 40 epochs; the best
validation macro-F1 was reached at epoch {B_ep}. Validation accuracy was still creeping up at the end, so more epochs
(affordable on a GPU) would probably help a little.</p>""",

"methodC": """<p>MobileNetV3-Large (Howard et al., 2019) with torchvision's <code>IMAGENET1K_V2</code> weights. The 1000-class
output layer is replaced by a 6-class layer (same output format as A and B); inputs are 224&times;224 and normalised with
the ImageNet mean and standard deviation the weights expect. Training has two stages: 3 epochs training only the
classifier head with the backbone frozen (so the random head does not damage the pretrained features), then
fine-tuning at learning rate 5&middot;10<sup>&minus;4</sup> (AdamW, cosine) for up to 12 epochs with early stopping
(best epoch {C_ep}). The stem and the first three inverted-residual blocks stay frozen with their BatchNorm statistics
fixed: they hold generic low-level filters and, operating at high resolution, dominate the cost (320 vs. 215 s per CPU
epoch). The split, augmentation, loss and stopping rule are identical to Method B. We chose the Large variant over
MobileNetV3-Small (proposed originally) for its higher accuracy; it is still a mobile network.</p>""",

"protocol": """<p>Every method loads the same saved split and is scored by the same function (our own confusion
matrix, accuracy, per-class precision/recall/F1 and macro-F1, averaged over the six classes so the small trash class
counts equally). Hyper-parameters and checkpoints are selected on the validation split only; the test split is used
once. Uncertainty is reported as 95% bootstrap confidence intervals (2,000 resamples of the test set) and pairs of
methods are compared with the exact McNemar test on their discordant predictions. Runtime is measured on the same CPU
(Intel i5-1135G7), batch size 1, after 10 warm-up passes, averaged over 200 test images, and includes HOG extraction for
A and normalisation for B and C. Seeds are fixed and library versions are printed in the notebook. Because the laptop's
NVIDIA driver was too old for the CUDA 12.6 runtime of the only PyTorch build for Python 3.14, all training ran on the
CPU; the code automatically uses a GPU when one works (e.g. on Colab).</p>""",

"experiments": """<ol>
<li><b>Main comparison</b> &mdash; accuracy, macro-F1, confusion matrices and per-class precision/recall/F1 of the three
methods (plus the colour variant) on the 376-image test split, with bootstrap CIs and McNemar tests.</li>
<li><b>Domain shift</b> &mdash; the models trained on TrashNet are applied unchanged to the 600 real-world web photos.</li>
<li><b>Small data</b> &mdash; every method is retrained on 10, 25 and 50 images per class (seeded subsamples of the
training split) and scored on the same test split. A re-tunes &lambda;; B and C use a fixed number of optimisation steps.</li>
<li><b>Efficiency</b> &mdash; runtime per image, parameter count, model size and training time on the same hardware,
summarised in accuracy-versus-cost plots.</li>
<li><b>Error analysis</b> &mdash; most-confused class pairs, failure examples, agreement between methods, a
majority-vote ensemble, and visual explanations: HOG gradient maps and Grad-CAM (Selvaraju et al., 2017) for B and C.</li>
</ol>""",

"res1": """<p>The ranking is clear and every pairwise difference is statistically significant (McNemar p &lt; 0.005):
the pretrained MobileNetV3 reaches {C_acc:.1f}% accuracy and macro-F1 {C_f1:.3f}, our scratch CNN {B_acc:.1f}% ({B_f1:.3f}) and
HOG + SVM {A_acc:.1f}% ({A_f1:.3f}). The 95% intervals do not overlap. <b>Why.</b> HOG encodes only the layout of edge
orientations. Material, however, is mostly a matter of colour, reflectance, transparency and fine texture, which HOG
throws away, while shape is shared across materials (a bottle can be glass or plastic, a box cardboard or paper). Adding
a colour histogram raises A to {Av_acc:.1f}% (McNemar p = 0.005), which confirms that the missing colour information is a
large part of A's weakness. The scratch CNN learns colour and texture filters itself, but 1,764 images are not enough
to learn robust features from random weights. MobileNetV3 starts from features learned on 1.28 million ImageNet images,
which already separate glossy from matte, transparent from opaque, printed from plain, so a small amount of trash data
is enough to map them to materials. Its training curves show a small generalisation gap (training accuracy ~99%,
validation ~97%), well controlled by early stopping.</p>""",

"res2": """<p>All methods lose accuracy on real-world photos, and the gap grows as the features become more
hand-made: MobileNetV3 keeps {C_rw:.1f}% (68% of its test accuracy), the scratch CNN {B_rw:.1f}% (51%) and HOG + SVM collapses
to {A_rw:.1f}%, below the 16.7% chance level ({Av_rw:.1f}% with colour). The confusion matrices show why A falls below
chance: it labels most web photos <i>paper</i> and real paper <i>cardboard</i>, i.e. its decision rules encode TrashNet's
edge statistics (one item, centred, on a blank table) rather than material. The scratch CNN also over-predicts paper and
plastic, and Grad-CAM (Fig. 13) shows that it often attends to the background and shadows, cues that are specific to
the studio set-up. MobileNetV3 attends to the object and transfers best: excluding the trash class its real-world
accuracy is 74.8%. <b>No method recognises the web "trash" class</b> (F1 &le; 0.05): in TrashNet "trash" means food
wrappers and packaging, whereas in the web set it is mostly diapers, toothbrushes and similar items. This is a change of
<i>label definition</i>, not only of image style, and no classifier can learn it from the training data. Two lessons
follow: test on data from the deployment setting, and fine-tune on local photos before real use.</p>""",

"res3": """<p>With only 10 images per class (60 images) the pretrained model already reaches {C_10:.1f}%, far above the
scratch CNN ({B_10:.1f}%) and HOG + SVM ({A_10:.1f}%); at 50 images per class the values are {C_50:.1f}%, {B_50:.1f}% and
{A_50:.1f}%. Strikingly, MobileNetV3 trained on only 10 images per class ({C_10:.1f}%) comes close to our scratch CNN trained on
all 1,764 images ({B_acc:.1f}%). This is the pattern transfer-learning theory predicts: the pretrained network only has
to learn a 6-way read-out (plus small adjustments) on top of features that already exist, so its sample complexity is
low. The scratch CNN must learn every filter from the data; it gains the most overall (from {B_10:.1f}% to {B_acc:.1f}%) and its
curve is still rising, so it is data-limited. HOG + SVM has no feature learning at all: it stays between 35% and 41% up
to 50 images per class and reaches only {A_acc:.1f}% with all data, because its fixed features cap what any amount of
data can teach the linear classifier.</p>""",

"res4": """<p>HOG + SVM is by far the cheapest: {A_ms:.1f} ms per image on the CPU, a 0.3 MB model and seconds of
training, but its accuracy is the lowest. The scratch CNN needs {B_ms:.1f} ms and {B_par:.2f} M parameters; MobileNetV3
needs {C_ms:.1f} ms and {C_par:.2f} M parameters (17 MB), still well within real-time on a laptop CPU (about 34 images per
second) and designed for phones. Notably, the pretrained model trained in <i>less</i> time than the scratch CNN (it
converges in a handful of epochs because it starts from good features), so pretraining reduces training cost as well
as error. Per point of accuracy, the extra ~25 ms of MobileNetV3 is clearly worth it for this application.</p>""",

"res5": """<p>A's errors follow its blindness to colour and transparency: glass is read as metal (both are shiny with
strong specular edges) and plastic as cardboard or paper. B's most frequent errors are plastic&rarr;glass (both
transparent) and paper&rarr;trash/metal. C makes few errors, mainly cardboard&rarr;paper and plastic&harr;glass, the pairs that
are hard even for people from one photo. On the test split all three methods are correct on {all_correct} of {n} images
and all wrong on only {all_wrong}; C alone is right on {only_C_correct}, while C is wrong but A or B right on only
{C_wrong_but_B_or_A_right}. A majority vote of the three ({vote:.1f}%) is therefore <i>worse</i> than C alone: the weaker
methods out-vote C more often than they rescue it. The Grad-CAM maps (Fig. 13) explain the domain-shift result: C's
evidence lies on the object (bottle neck, can rim, the item itself), whereas B often relies on background regions.</p>""",

"answer": """<p><b>Which method best sorts trash by material?</b> The pretrained MobileNetV3-Large: {C_acc:.1f}% test
accuracy (macro-F1 {C_f1:.3f}) versus {B_acc:.1f}% for our scratch CNN and {A_acc:.1f}% for HOG + SVM, significant in every pairwise
test, best in the small-data regime, and fast enough for a phone ({C_ms:.0f} ms on a laptop CPU).</p>
<p><b>Does it still work on messy real-world photos after training on clean ones?</b> Only partly. It keeps the most
accuracy ({C_rw:.1f}%, about 75% if the ill-defined trash class is excluded), while the scratch CNN ({B_rw:.1f}%) and especially
HOG + SVM ({A_rw:.1f}%) largely fail. The ranking of the methods is the same in both settings, so the application
recommends MobileNetV3 by default, but a deployment in Iloilo should fine-tune it on locally collected photos and define
the "trash" class the same way as the local MRF does.</p>""",

"system": """<p>The comparison is wrapped in a Gradio web application (<code>app/app.py</code>, logic in
<code>src/system.py</code>). <b>Input:</b> the user uploads a photo (or uses the webcam/clipboard). <b>Processing:</b> the
image is resized for each method and all three run. <b>Output drawn on the input:</b> the recommended method's label and
confidence in a band above the photo, and three coloured bars below it with each method's label, confidence and
measured runtime, plus a table of all class probabilities. <b>Recommendation:</b> computed automatically from our result
files: the method with the highest test macro-F1 is the default, and the explanation quotes its test accuracy, its
real-world accuracy and its CPU runtime. The app also says whether the methods agree, warns when the confidence is below
50% and asks for a retake on a plain background, and gives a disposal tip for the predicted material (RA 9003
categories). It runs locally (<code>python app/app.py</code>) or from the Colab notebook with a public link.</p>""",

"conclusion": """<p>Learned features beat hand-designed ones for material recognition, and pretrained features beat
features learned from scratch on a small dataset, by a wide and statistically significant margin on clean data, in the
small-data regime and under domain shift. The classical pipeline remains valuable for what it teaches: it is
transparent, tiny and fast, and its failures show exactly which visual cues (colour, transparency, texture) material
recognition needs. Our careful data handling also mattered: identical files with conflicting labels, near-duplicate
shots and 490 test images that were copies of training images would otherwise have inflated the results.</p>
<p><b>Limitations.</b> TrashNet is small and studio-like; the real-world set comes from a different source with a
different notion of "trash"; images are squashed to a square; all training ran on a laptop CPU, which limited the
number of epochs for the scratch CNN; each configuration was trained with one seed (the bootstrap CIs capture test-set
noise, not training noise); and the findings apply to these three methods, not to every method of each family.
<b>Future work:</b> collect labelled phone photos from Iloilo (the code evaluates <code>data/own_photos/</code>
automatically) and fine-tune on them; add test-time augmentation; detect several items per photo.</p>""",

"contrib": """<p><i>(To be completed by the group: one paragraph listing what each member did.)</i> Suggested split:
Member 1 &mdash; Method A (HOG, SVM, sanity checks); Member 2 &mdash; Method B (architecture, training, curves); Member 3
&mdash; Method C (fine-tuning, preprocessing); Member 4 &mdash; data cleaning, split, real-world set, experiments 2&ndash;5;
Member 5 &mdash; application, demo video, report and slides. Every member reviewed the others' code.</p>
<p><b>Use of AI tools.</b> An AI coding assistant (Claude) was used to help write and debug code, run experiments,
produce plots and draft the documentation, report and slides. All design decisions, results and text were reviewed by
the group, and every member can explain every line of the code.</p>""",
}
