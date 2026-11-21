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
