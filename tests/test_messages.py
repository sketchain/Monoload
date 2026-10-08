"""Messages and translations (monoload/messages.py, locales/, web/monoload_i18n.js).

No model files needed.

  1. the message table: every key has English and Chinese with the same
     format fields; MONOLOAD_LANG parsing (unset / en -> English, zh / zh-CN
     / zh_CN -> Chinese, anything else -> English with a warning);
  2. Chinese at run time: a budget error (MonoloadError naming what layer 2
     and each scheme need), the OOM messages, an unsupported-LoRA error, the
     nodes' logs and errors, the Info node's text; English by default;
  3. no user-facing Chinese left in the code outside the table;
  4. locales/en and locales/zh nodeDefs.json: valid, every node, every input
     (name, tooltip), every dropdown option, every output; the English input
     names are the real ones; the option keys are the stored (English) values;
     the web extensions are there.

    python tests/test_messages.py
"""

import glob
import json
import os
import re
import string

import torch

from common import REPO, check, expect_raises, finish
from monoload import info, messages, settings
from monoload import vae as mvae
from monoload.errors import MonoloadError, MonoloadUnsupportedError
from test_vae import managed_decode
from test_vae_ldm import ldm_vae
from test_vae_node import LogCapture, node_apply, reset_globals

HAN = re.compile("[一-鿿]")


def fields(s):
    return {f for _, f, _, _ in string.Formatter().parse(s) if f}


def table_tests():
    bad = [k for k, v in messages.M.items() if len(v) != 2 or not v[0] or not v[1] or fields(v[0]) != fields(v[1])]
    norm = lambda t: re.sub(r"\s", "", t.translate(str.maketrans("（）：，；", "():,;")))   # noqa: E731
    zh_missing = [k for k, v in messages.M.items() if not HAN.search(v[1]) and norm(v[0]) != norm(v[1])
                  and not k.startswith(("v.", "src."))]
    check("message table: {} keys, each with English and Chinese and the same fields{}".format(
          len(messages.M), "" if not bad else " -- bad: {}".format(bad)), not bad)
    check("every Chinese text that differs from the English one is Chinese ({})".format(zh_missing or "ok"), not zh_missing)
    saved = os.environ.get("MONOLOAD_LANG")
    got = {}
    try:
        for raw in (None, "en", "zh", "zh-CN", "zh_CN", "ZH", "fr"):
            if raw is None:
                os.environ.pop("MONOLOAD_LANG", None)
            else:
                os.environ["MONOLOAD_LANG"] = raw
            got[raw] = messages._lang_from_env()
    finally:
        if saved is None:
            os.environ.pop("MONOLOAD_LANG", None)
        else:
            os.environ["MONOLOAD_LANG"] = saved
    check("MONOLOAD_LANG: {}".format(", ".join("{} -> {}".format("unset" if k is None else k, v[0]) for k, v in got.items())),
          [got[k][0] for k in (None, "en")] == ["en", "en"] and [got[k][0] for k in ("zh", "zh-CN", "zh_CN", "ZH")] == ["zh"] * 4
          and got["fr"] == ("en", "fr"))


def runtime_tests():
    sd = ldm_vae(4, True)
    lat = torch.randn(1, 4, 12, 10, generator=torch.Generator().manual_seed(7))
    from monoload.nodes import NODE_CLASS_MAPPINGS
    cls = NODE_CLASS_MAPPINGS["MonoloadVAESettings"]
    tiny = node_apply(cls, sd, budget="custom", budget_gib=0.001)
    expect_raises("English by default: budget error", MonoloadError, lambda: managed_decode(tiny, lat),
                  "does not fit the peak budget", "layer 2 needs about", "scheme B")
    messages.set_lang("zh")
    try:
        expect_raises("MONOLOAD_LANG=zh: the budget error in Chinese, naming layer 2 and each scheme", MonoloadError,
                      lambda: managed_decode(tiny, lat), "峰值预算", "第二层需要约", "方案 B")
        oom = messages.msg("vae.err_oom_l2", ws="64 MiB", retries=4, shape=[1, 4, 270, 480], est="17.9 GiB")
        oom1 = messages.msg("vae.err_oom_l1", rows=8, ws="64 MiB", retries=5, shape=[1, 4, 270, 480], est="2.9 GiB", skipped="")
        check("MONOLOAD_LANG=zh: the OOM errors in Chinese ({}...)".format(oom[:40]), "显存不足" in oom and "tiled" in oom and "第二层" in oom1)
        e = MonoloadUnsupportedError("force_patch_weights", messages.msg("lora.force_patch"), key="a.weight")
        check("MONOLOAD_LANG=zh: unsupported-LoRA error in Chinese, kind and key kept ({}...)".format(str(e)[:50]),
              "不支持（force_patch_weights）" in str(e) and "key=a.weight" in str(e) and "烘焙" in str(e))
        expect_raises("MONOLOAD_LANG=zh: node input error in Chinese", ValueError,
                      lambda: node_apply(cls, sd, budget="custom", budget_gib=0.0), "自定义")
        with LogCapture() as cap:
            node_apply(cls, sd, budget="default", budget_gib=2.0)
            mvae.set_budget(None)
            managed_decode(node_apply(cls, sd, mode="native"), lat)
        logs = [line for line in cap.lines if "[Monoload]" in line]
        check("MONOLOAD_LANG=zh: node and decode logs in Chinese, values and sources too ({})".format(" | ".join(x[:50] for x in logs)),
              logs and all(HAN.search(x) for x in logs) and "（node）" not in "".join(logs) and "（default）" not in "".join(logs))
        t = info.report(vae=sd)
        check("MONOLOAD_LANG=zh: the Info node's text in Chinese:\n" + t, "总开关" in t and "设置：模式" in t)
        t = info.report()
        check("MONOLOAD_LANG=zh: global defaults in Chinese", "全局默认值" in t and "内置默认" in t)
    finally:
        messages.set_lang("en")
        reset_globals()
    t = info.report()
    check("back to English", "global defaults" in t and not HAN.search(t))


def code_tests():
    left = []
    for f in sorted(glob.glob(os.path.join(REPO, "monoload", "**", "*.py"), recursive=True)) + [os.path.join(REPO, "__init__.py")]:
        if f.endswith("messages.py"):
            continue
        for i, line in enumerate(open(f, encoding="utf-8"), 1):
            if HAN.search(line):
                left.append("{}:{}".format(os.path.relpath(f, REPO), i))
    check("no Chinese text in the code outside monoload/messages.py ({})".format(left[:5] or "none"), not left)


def locale_tests():
    from monoload.nodes import NODES
    data = {}
    for lang in ("en", "zh"):
        with open(os.path.join(REPO, "locales", lang, "nodeDefs.json"), encoding="utf-8") as f:
            data[lang] = json.load(f)
    problems = []
    for cls in NODES:
        it = cls.INPUT_TYPES()
        inputs = {**it.get("required", {}), **it.get("optional", {})}
        for lang, d in data.items():
            nd = d.get(cls.__name__)
            if not nd or not nd.get("display_name") or not nd.get("description"):
                problems.append("{} {}: name / description".format(lang, cls.__name__))
                continue
            for name, spec in inputs.items():
                e = nd["inputs"].get(name)
                if not e or not e.get("name"):
                    problems.append("{} {}.{}".format(lang, cls.__name__, name))
                    continue
                if lang == "en" and e["name"] != name:
                    problems.append("en {}.{} name {}".format(cls.__name__, name, e["name"]))
                if len(spec) > 1 and spec[1].get("tooltip") and not e.get("tooltip"):
                    problems.append("{} {}.{} tooltip".format(lang, cls.__name__, name))
                if isinstance(spec[0], list) and set(e.get("options", {})) != set(spec[0]):
                    problems.append("{} {}.{} options {}".format(lang, cls.__name__, name, sorted(e.get("options", {}))))
            for i in range(len(cls.RETURN_TYPES)):
                if not nd.get("outputs", {}).get(str(i), {}).get("name"):
                    problems.append("{} {} output {}".format(lang, cls.__name__, i))
        zh = data["zh"].get(cls.__name__, {})
        if not HAN.search(json.dumps(zh, ensure_ascii=False)):
            problems.append("zh {} not translated".format(cls.__name__))
    check("locales/en and locales/zh nodeDefs.json: every node, input (name, tooltip), dropdown option (keys = the stored English "
          "values) and output ({})".format(problems or "complete"), not problems)
    web = sorted(os.path.basename(p) for p in glob.glob(os.path.join(REPO, "web", "*.js")))
    check("web extensions: {}".format(web), web == ["monoload_i18n.js", "monoload_info.js"])


def main():
    if not mvae.is_installed():
        mvae.install()
    messages.set_lang("en")
    settings.set_master(True)
    reset_globals()
    table_tests()
    runtime_tests()
    code_tests()
    locale_tests()
    finish()


if __name__ == "__main__":
    main()
