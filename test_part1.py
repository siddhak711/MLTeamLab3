"""
Tests for Lab 3, Part 1. Don't edit this file.

    python test_part1.py               run every test on your lab3_part1.py
    python test_part1.py softmax       run only the tests whose name contains "softmax"

Each test prints ✅ / ❌ (with what you got vs. what was expected), or ⏭ if you haven't written it yet.
"""
import sys, importlib
import numpy as np
from lab_utils import enc, EOS, N_HEADS, params, check, show_top_k, run_tests, check_against_gpt2

# Load your functions from lab3_part1.py  (instructors: add --solution to test the answer key)
_module = importlib.import_module("lab3_part1_solution" if "--solution" in sys.argv else "lab3_part1")
globals().update({k: v for k, v in vars(_module).items() if not k.startswith("_")})
_filters = [a.lower() for a in sys.argv[1:] if not a.startswith("--")]

# ============================================================================
# Tests
# ============================================================================

def test_embed():
    W_E_toy = np.arange(15).reshape(5, 3)    # vocab of 5 tokens, d = 3
    check("embed looks up the right rows", embed([2, 0, 2], W_E_toy), [[6, 7, 8], [0, 1, 2], [6, 7, 8]])
    check("embed on GPT-2 has shape (n, 768)", embed([464, 1468, 3797], params["W_E"]).shape, (3, 768))


def test_add_positions():
    X_toy = np.ones((3, 2))
    W_pos_toy = np.arange(10).reshape(5, 2)   # room for 5 positions
    check("add_positions adds position t to row t", add_positions(X_toy, W_pos_toy), [[1, 2], [3, 4], [5, 6]])


def test_mlp():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(2, 3))
    W_up, b_up = rng.normal(size=(3, 6)), rng.normal(size=6)
    W_down, b_down = rng.normal(size=(6, 3)), rng.normal(size=3)
    check("mlp on a toy example", mlp(x, W_up, b_up, W_down, b_down),
          [[ 0.340929,  1.598733,  0.786334],
     [-0.729246,  1.436732,  0.655550]])
    check("mlp keeps the shape (n, d)", mlp(x, W_up, b_up, W_down, b_down).shape, (2, 3))


def test_block():
    rng = np.random.default_rng(1)
    d = 4
    toy_block = {"ln1_g": np.ones(d), "ln1_b": np.zeros(d), "ln2_g": np.ones(d), "ln2_b": np.zeros(d),
                 "W_qkv": rng.normal(size=(d, 3 * d)), "b_qkv": np.zeros(3 * d),
                 "W_O": rng.normal(size=(d, d)), "b_O": np.zeros(d),
                 "W_up": rng.normal(size=(d, 4 * d)), "b_up": rng.normal(size=4 * d),
                 "W_down": rng.normal(size=(4 * d, d)), "b_down": rng.normal(size=d)}
    x = rng.normal(size=(3, d))
    check("transformer_block on a toy block", transformer_block(x, toy_block),
          [[  3.828682,  -1.473463,  -6.213097,   3.121150],
     [  1.656373, -10.699569,   2.951125,  -1.889437],
     [ 12.054587,   2.808415,  -4.445967,  -2.040933]])


def test_unembed():
    W_E_toy = np.arange(15).reshape(5, 3)
    check("unembed gives one score per vocab token", unembed(np.array([[1.0, 2.0, 3.0]]), W_E_toy), [[8, 26, 44, 62, 80]])


def test_forward():
    ids = enc.encode("The old cat sat on the")
    check("forward output shape is (n, V)", forward(ids, params).shape, (len(ids), params["W_E"].shape[0]))
    check_against_gpt2(forward, zero_attention=True)   # compares to real GPT-2 with its attention switched off


def test_softmax():
    z = np.array([4.0, 2.0, 1.0])   # Vayun, Owen, Jean from the lecture
    check("softmax at T = 2 (lecture example)", softmax(z, T=2), [0.6285, 0.2312, 0.1402], atol=1e-3)
    check("softmax at T = 0.5 (lecture example)", softmax(z, T=0.5), [0.9796, 0.0179, 0.0024], atol=1e-3)
    check("softmax doesn't overflow on big logits", softmax(np.array([1000.0, 1001.0])), [0.2689, 0.7311], atol=1e-3)
    check("softmax on 2D: every row sums to 1", softmax(np.random.default_rng(0).normal(size=(3, 5))).sum(-1), np.ones(3))


def test_sample():
    p = np.array([0.1, 0.6, 0.3])
    check("greedy picks the most likely token", sample(p, greedy=True), 1)
    rng = np.random.default_rng(0)
    draws = [sample(p, rng=rng) for _ in range(20000)]
    check("random draws match the probabilities", np.bincount(draws, minlength=3) / len(draws), p, atol=0.02)


def test_generate():
    out = generate("The old cat sat on the", max_new_tokens=8, greedy=True)
    print(out)
    check("generate returns the prompt plus new text", out.startswith("The old cat sat on the"), True)


TESTS = [
    ('Embeddings: embed', test_embed),
    ('Embeddings: add_positions', test_add_positions),
    ('MLP', test_mlp),
    ('Transformer block', test_block),
    ('Unembedding', test_unembed),
    ('Forward pass', test_forward),
    ('Softmax', test_softmax),
    ('Sampling', test_sample),
    ('Generate', test_generate),
]

# ============================================================================
# Experiments (run after all tests, when no filter is given)
# ============================================================================

def experiments():
    for prompt in ["The old cat sat on the", "The old dog sat on the"]:
        probs = softmax(forward(enc.encode(prompt), params)[-1])
        print(prompt)
        show_top_k(probs)

    same = np.allclose(forward(enc.encode("The old cat sat on the"), params)[-1],
                       forward(enc.encode("The old dog sat on the"), params)[-1])
    print("Exactly the same prediction?", same)

    for T in [0.3, 1.0, 1.5]:
        print(f"\nT = {T}:", generate("The old cat sat on the", max_new_tokens=15, T=T))


if __name__ == "__main__":
    selected = [(n, f) for n, f in TESTS if not _filters or any(x in n.lower() for x in _filters)]
    if not selected:
        print("No test names match", _filters, "- options:", [n for n, _ in TESTS])
    run_tests(selected)
    if not _filters:
        print("\n=== Experiments ===")
        run_tests([("Experiments", experiments)])
