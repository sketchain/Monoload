// Monoload Info: show the node's text (its UI output "text") in the node box.
// Uses the text preview widget of ComfyUI's frontend (window.comfyAPI.textPreviewWidgets,
// the one of the core "Preview as Text" node); falls back to a read-only multiline text widget.
import { app } from "../../scripts/app.js";

const NODE = "MonoloadInfo";

function textOf(message) {
  const t = message?.text ?? "";
  return Array.isArray(t) ? t.join("\n\n") : String(t);
}

function addFallback(node) {
  const w = node.addWidget("text", "info", "", () => {}, { multiline: true, serialize: false });
  if (w) w.serialize = false;
  return w;
}

app.registerExtension({
  name: "Monoload.Info",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== NODE) return;
    const tp = window.comfyAPI?.textPreviewWidgets;
    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = onNodeCreated?.apply(this, arguments);
      try {
        if (tp?.addTextPreviewWidgets) tp.addTextPreviewWidgets(this);
        else this.monoloadInfo = addFallback(this);
      } catch (e) {
        console.warn("[Monoload] Info: text widget", e);
        this.monoloadInfo = addFallback(this);
      }
      return r;
    };
    const onExecuted = nodeType.prototype.onExecuted;
    nodeType.prototype.onExecuted = function (message) {
      onExecuted?.apply(this, arguments);
      if (this.monoloadInfo) this.monoloadInfo.value = textOf(message);
      else tp?.updateTextPreviewWidgets?.(this, message);
      this.setDirtyCanvas?.(true, true);
    };
  },
});
