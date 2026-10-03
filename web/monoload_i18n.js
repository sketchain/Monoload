// Monoload: display-only translation of the Monoload nodes' dropdown options.
// The labels come from the official locales files (locales/<lang>/nodeDefs.json,
// inputs.<name>.options), served by ComfyUI at /i18n. ComfyUI's frontend 1.48.7
// translates node names, input names, outputs and tooltips from those files but
// not combo option labels, so this sets the combo widget's getOptionLabel: only
// what is shown changes, the value saved in the workflow stays the English one.
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const NODES = new Set(["MonoloadLoRASettings", "MonoloadVAESettings", "MonoloadInfo"]);
let table = {};
const ready = api.fetchApi("/i18n").then(r => r.json()).then(j => { table = j || {}; }).catch(() => {});

function locale() {
  let l;
  try { l = app.extensionManager?.setting?.get("Comfy.Locale"); } catch (e) { l = undefined; }
  return l || navigator.language || "en";
}

function optionLabel(nodeType, input, value) {
  const loc = locale();
  for (const l of [loc, loc.split("-")[0], "en"]) {
    const v = table?.[l]?.nodeDefs?.[nodeType]?.inputs?.[input]?.options?.[value];
    if (typeof v === "string" && v) return v;
  }
  return value;
}

function attach(node) {
  const type = node?.comfyClass ?? node?.type;
  if (!NODES.has(type)) return;
  for (const w of node.widgets || []) {
    if (w.type !== "combo" || !w.options) continue;
    const name = w.name;
    w.options.getOptionLabel = (v) => (v == null ? "" : optionLabel(type, name, String(v)));
  }
  node.setDirtyCanvas?.(true, true);
}

app.registerExtension({
  name: "Monoload.I18n",
  async setup() { await ready; app.graph?.setDirtyCanvas?.(true, true); },
  nodeCreated(node) { attach(node); },
  loadedGraphNode(node) { attach(node); },
});
