// Text content of BIN: what to do with each material and short help articles.
// Philippine context: RA 9003 requires segregation at source into biodegradable,
// recyclable and residual waste. The six classes are the TrashNet classes.

export const MATERIALS = {
  cardboard: {
    emoji: "📦", color: "#b9822f", bin: "recyclable",
    tip: "Flatten it and keep it dry. Wet or greasy cardboard (e.g. pizza boxes) goes to residual.",
    guide: [
      "Remove tape, foam and plastic wrap.",
      "Flatten boxes to save space in the recycling bin.",
      "Keep cardboard dry: wet fibres cannot be recycled well.",
      "Greasy or food-soiled cardboard is residual waste.",
    ],
  },
  glass: {
    emoji: "🍾", color: "#2f8f83", bin: "recyclable",
    tip: "Rinse bottles and jars. Wrap broken glass before disposal so collectors are not injured.",
    guide: [
      "Rinse bottles and jars; labels can stay on.",
      "Separate by colour (clear, green, brown) if your MRF asks for it.",
      "Mirrors, window glass and light bulbs are not container glass.",
      "Wrap broken glass in paper and label it.",
    ],
  },
  metal: {
    emoji: "🥫", color: "#5b6b78", bin: "recyclable",
    tip: "Rinse cans and crush them to save space. Junk shops buy aluminium and steel.",
    guide: [
      "Rinse food and drink cans.",
      "Crush aluminium cans to save space.",
      "Aerosol cans must be completely empty.",
      "Scrap metal has value: junk shops buy it.",
    ],
  },
  paper: {
    emoji: "📄", color: "#3d7bd6", bin: "recyclable",
    tip: "Keep paper clean and dry. Tissue and food-soiled paper are not recyclable.",
    guide: [
      "Newspapers, notebooks, office paper and magazines are recyclable.",
      "Remove plastic covers and spiral bindings.",
      "Tissue, napkins and food-soiled paper are residual.",
      "Shredded paper should be bagged.",
    ],
  },
  plastic: {
    emoji: "🧴", color: "#d9534f", bin: "recyclable",
    tip: "Rinse PET bottles and rigid containers. Sachets and soft wrappers are usually residual.",
    guide: [
      "PET bottles (code 1) and HDPE containers (code 2) are the most recycled.",
      "Rinse and drain containers; caps can stay on.",
      "Sachets, candy wrappers and plastic labo are usually residual.",
      "Avoid single-use plastic whenever you can.",
    ],
  },
  trash: {
    emoji: "🗑️", color: "#7a6a5a", bin: "residual",
    tip: "Residual waste: it cannot be recycled. Put it in the residual bin.",
    guide: [
      "Residual waste cannot be recycled or composted.",
      "Examples: diapers, sanitary products, sachets, used tissue, styrofoam.",
      "Keep it separate so recyclables stay clean.",
      "Less residual waste means less landfill: choose refillables.",
    ],
  },
};

export const BIN_LABEL = { recyclable: "Recyclable", residual: "Residual" };

export const ARTICLES = [
  {
    id: "photos", emoji: "📸", title: "How to get an accurate answer",
    summary: "Five habits that make BIN much more reliable.",
    body: `<ul><li>Photograph <b>one item</b> at a time.</li>
<li>Fill most of the <b>scanner frame</b> with the item (BIN only looks inside the frame).</li>
<li>Use a <b>plain background</b>: a table, the floor or a sheet of paper.</li>
<li>Good light, no strong shadows, hold still to avoid blur.</li>
<li>Show the material: the surface, the label or the inside of the item.</li></ul>
<p>When BIN is not confident enough it says <b>"Not sure"</b> instead of guessing. Retake the photo
closer, from another side, or on a plainer background.</p>`,
  },
  {
    id: "ai", emoji: "🤖", title: "How BIN works",
    summary: "Three computer-vision methods on your phone.",
    body: null,   // filled in by app.js with the measured results
  },
  {
    id: "wishcycling", emoji: "🤔", title: "What is wish-cycling?",
    summary: "Putting things in the recycling bin and hoping for the best.",
    body: `<p><b>Wish-cycling</b> is throwing an item into the recycling bin because you <i>hope</i> it can be
recycled, without checking. It feels good, but it often does harm.</p>
<ul><li>One greasy box or a bag of food waste can contaminate a whole load of clean paper.</li>
<li>Sachets, soft plastics and tissues jam sorting machines and end up in landfill anyway.</li>
<li>Contaminated loads are sometimes rejected completely.</li></ul>
<p><b>When in doubt, check.</b> Scan the item with BIN and ask your barangay MRF what they accept.</p>`,
  },
  {
    id: "ra9003", emoji: "⚖️", title: "RA 9003 in one minute",
    summary: "The Philippine law on solid waste segregation.",
    body: `<p>The <b>Ecological Solid Waste Management Act of 2000 (Republic Act No. 9003)</b> requires
waste to be segregated <b>at source</b>: in homes, schools and offices.</p>
<ul><li><b>Biodegradable:</b> food scraps, garden waste (compost them).</li>
<li><b>Recyclable:</b> paper, cardboard, glass, metal, many plastics.</li>
<li><b>Residual:</b> what cannot be recycled or composted.</li>
<li><b>Special waste:</b> batteries, electronics, paint, medicine.</li></ul>
<p>Every barangay is supposed to have a <b>Materials Recovery Facility (MRF)</b> where recyclables are
sorted. BIN helps with the recyclable vs. residual decision.</p>`,
  },
];

// Plain-language explanation of the three slots of the course comparison, shown under
// the comparison table on the result screen (course objectives 1, 2 and 5).
export const SLOT_INFO = {
  A: {
    slot: "A. Classical",
    name: "HOG + linear SVM",
    what: "A method from before deep learning. A person decides which features to measure; the computer only learns how to weigh them.",
    how: [
      "The photo is turned into grayscale and shrunk to 128 × 128 pixels.",
      "<b>HOG</b> (Histogram of Oriented Gradients) finds the edges and, in every small 8 × 8 patch, counts in which direction they point. The result is a list of 8,100 numbers that describes the outline and shape of the item.",
      "A <b>linear SVM</b> (Support Vector Machine) learned one straight-line rule per material that separates these numbers; the material with the highest score wins.",
      "Both HOG and the SVM were written from scratch by our group in NumPy (no library for the main step), and checked against a reference implementation.",
    ],
    good: "Tiny and very fast, has no learned layers (0 parameters) and every step can be explained.",
    bad: "HOG ignores colour, shine and transparency, which is what mostly tells materials apart: a glass bottle and a plastic bottle have the same outline. It also reacts to every edge in the background. That is why it is the least accurate.",
  },
  B: {
    slot: "B. Trained from scratch",
    name: "4-block CNN",
    what: "A small neural network that we designed and trained ourselves, starting from random numbers.",
    how: [
      "A CNN (Convolutional Neural Network) looks at the 128 × 128 photo through many small filters.",
      "Our network has 4 blocks (convolution → normalisation → ReLU → pooling); each block sees a larger part of the photo, from edges and colours to textures and object parts.",
      "Unlike HOG, the network <b>learns its own features</b>, including colour and texture, only from our trash photos (1.13 million numbers learned).",
      "Training used augmentation (random crops, rotations, flips, colour changes), dropout and early stopping to avoid memorising the photos.",
    ],
    good: "Learns colour and texture by itself, so it is much better than HOG + SVM.",
    bad: "A few thousand photos are not enough to learn good features from zero. It still mixes up look-alike materials (clear plastic and glass) and sometimes relies on the background instead of the item.",
  },
  C: {
    slot: "C. Pretrained, modern",
    name: "MobileNetV3-Large",
    what: "A published network (Google, 2019) that first learned from 1.28 million everyday photos (ImageNet) and was then fine-tuned by us on trash photos.",
    how: [
      "The photo is resized to 224 × 224 pixels and normalised the same way as during its original training.",
      "Because of its pre-training, the network already knows what shiny, transparent, printed, folded or crushed surfaces look like.",
      "We replaced its last layer with our 6 materials and fine-tuned it on the same trash photos as the other two methods (transfer learning).",
      "MobileNetV3 is designed for phones: efficient layers keep it small (4.2 million numbers) and fast.",
    ],
    good: "The most accurate in every test, even with only 10 training photos per material, and fast enough for a phone. That is why BIN uses its answer.",
    bad: "Largest model and slowest of the three; it can still be wrong on unusual photos (several items, dark or blurry pictures), so BIN says \"Not sure\" when its confidence is low.",
  },
};
