// Text content of the app: material guides, tips and Learn articles.
// Philippine context: RA 9003 requires segregation at source into biodegradable,
// recyclable and residual waste. The six classes here are the TrashNet classes.

export const MATERIALS = {
  cardboard: {
    emoji: "📦", color: "#b9822f", bin: "recyclable",
    tip: "Flatten boxes and keep them dry. Wet or greasy cardboard (e.g. pizza boxes) goes to residual.",
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
    tip: "Residual waste. Not recyclable: put it in the residual bin for proper disposal.",
    guide: [
      "Residual waste cannot be recycled or composted.",
      "Examples: diapers, sanitary products, sachets, used tissue.",
      "Keep it separate so recyclables stay clean.",
      "Less residual waste means less landfill: choose refillables.",
    ],
  },
};

export const BIN_LABEL = { recyclable: "Recyclable", residual: "Residual" };

export const TIPS = [
  ["🚿", "Rinse before you recycle: food leftovers can contaminate a whole bale of recyclables."],
  ["📦", "Flatten boxes: they take up to 80% less space in the bin."],
  ["🧴", "Check the number in the recycling triangle: PET (1) and HDPE (2) are the easiest plastics to recycle."],
  ["🛍️", "Bring your own bag and tumbler: the best waste is the waste you never make."],
  ["🔋", "Batteries and electronics are special waste: never put them in the regular bins."],
  ["🌱", "Food scraps belong in the biodegradable bin or your compost, not with recyclables."],
];

export const ARTICLES = [
  {
    id: "wishcycling", emoji: "🤔", title: "What is wish-cycling?",
    summary: "Putting things in the recycling bin and hoping for the best.",
    body: `<p><b>Wish-cycling</b> is throwing an item into the recycling bin because you <i>hope</i> it can be
recycled, without checking. It feels good, but it often does harm.</p>
<ul><li>One greasy box or a bag of food waste can contaminate a whole load of clean paper.</li>
<li>Sachets, soft plastics and tissues jam sorting machines and end up in landfill anyway.</li>
<li>Contaminated loads are sometimes rejected completely.</li></ul>
<p><b>When in doubt, check.</b> Scan the item with this app, read the material guide, and ask your
barangay MRF what they accept.</p>`,
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
sorted. This app helps with the recyclable vs. residual decision.</p>`,
  },
  {
    id: "ai", emoji: "🤖", title: "How the AI decides",
    summary: "Three computer-vision methods, compared fairly.",
    body: null,   // filled in by app.js with the measured results
  },
  {
    id: "photos", emoji: "📸", title: "How to take a good photo",
    summary: "Get the most accurate answer from the scanner.",
    body: `<ul><li>Photograph <b>one item</b> at a time.</li>
<li>Use a <b>plain background</b> (a table, the floor, a sheet of paper).</li>
<li>Fill most of the scanner frame with the item.</li>
<li>Good light, no strong shadows; avoid blur.</li></ul>
<p>The models were trained on photos of single items on a plain background, so messy scenes are harder:
in our tests the best model scored 92.8% on such photos but about 63% on random web photos. If the answer
looks wrong, pick the right material on the result screen.</p>`,
  },
];

export const BADGES = [
  { id: "first", emoji: "🌱", name: "First scan", test: (s) => s.items >= 1 },
  { id: "ten", emoji: "♻️", name: "10 items", test: (s) => s.items >= 10 },
  { id: "allsix", emoji: "🌈", name: "All 6 materials", test: (s) => s.kinds >= 6 },
  { id: "century", emoji: "💯", name: "100 points", test: (s) => s.points >= 100 },
  { id: "fifty", emoji: "🏅", name: "50 items", test: (s) => s.items >= 50 },
  { id: "level5", emoji: "🏆", name: "Level 5", test: (s) => s.level >= 5 },
];
