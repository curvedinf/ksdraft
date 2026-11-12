# SPDX-License-Identifier: Apache-2.0
"""On-policy builder prompt construction (no server)."""

import importlib.util
import random
from pathlib import Path

import torch

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_onpolicy_corpus.py"


def _module():
    spec = importlib.util.spec_from_file_location("_onpolicy_test", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _Tok:
    def decode(self, ids):
        body = ("### Instruction:\nExplain a hash table in full detail, please.\n\n"
                "### Response:\nA hash table maps keys to values. ")
        return body * 100

    def apply_chat_template(self, msgs, tools=None, tokenize=False,
                            add_generation_prompt=True):
        return "<|user|>" + msgs[-1]["content"][:20] + "<|assistant|>"


def test_build_prompts_tiers(tmp_path):
    mod = _module()
    corpus = tmp_path / "corpus.pt"
    torch.save(torch.zeros(64, dtype=torch.int64), corpus)
    args = type("A", (), dict(n_code=2, n_agent=1, n_doc=1, n_chat=1,
                              max_new=64, doc_chars=2000, doc_new=64,
                              max_prompt=12000))()
    prompts = mod.build_prompts(_Tok(), corpus, args, random.Random(0))
    kinds = sorted(p["kind"] for p in prompts)
    assert kinds == ["agent", "alpaca", "alpaca", "chat", "doc"]
    assert all(p["prompt"].endswith("<|assistant|>") for p in prompts)


def test_cli_live_requires_model(capsys):
    from ksdraft.cli import main

    assert main(["mine", "--live"]) == 2
    assert "requires --model" in capsys.readouterr().err
