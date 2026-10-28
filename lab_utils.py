"""
Shared setup for Lab 3 (both parts). You don't need to edit this file, but reading it
is a good way to see how a real model's weights are organized.

What this file does, top to bottom:
    1. Loads GPT-2's tokenizer (text <-> token IDs).
    2. Downloads GPT-2 small's trained weights and converts them to NumPy arrays.
    3. Renames those weights to match the names we used in lecture (W_E, W_pos, W_up, ...).
    4. Defines helper functions you'll call but don't have to write (layer_norm, gelu, ...).
    5. Defines the checker that compares YOUR forward() to the official GPT-2.

Install once:   pip install numpy torch transformers tiktoken matplotlib
The first run downloads GPT-2 small (~500 MB). After that it loads from a local cache.
"""
import numpy as np
import tiktoken                          # OpenAI's fast tokenizer library
import torch                             # only used to load weights and to run the official model for checks
from transformers import GPT2LMHeadModel # Hugging Face's ready-made GPT-2

# Print arrays with 3 decimals and no scientific notation, so test output is readable.
np.set_printoptions(precision=3, suppress=True)


# =============================================================================
# 1. The tokenizer
# =============================================================================
# enc.encode("The old cat")  ->  [464, 1468, 3797]      (text -> list of token IDs)
# enc.decode([464, 1468])    ->  "The old"              (token IDs -> text)
enc = tiktoken.get_encoding("gpt2")

# The special "end of text" token. When the model predicts it, generation stops.
EOS = enc.eot_token        # = 50256, the very last ID in GPT-2's vocabulary

# GPT-2 small runs 12 attention heads in every block (used in Part 2).
N_HEADS = 12


# =============================================================================
# 2. Download the trained weights
# =============================================================================
print("Loading GPT-2 small...")

# from_pretrained("gpt2") downloads the model the first time and caches it afterwards.
# .eval() switches off training-only behaviour (like dropout) so outputs are deterministic.
_hf_model = GPT2LMHeadModel.from_pretrained("gpt2").eval()

# state_dict() is every weight in the model, as {name: PyTorch tensor}.
# We convert each tensor to a NumPy array so the lab is pure NumPy.
# (Names starting with _ are internal to this file and aren't imported by the lab files.)
_sd = {name: tensor.detach().numpy() for name, tensor in _hf_model.state_dict().items()}


# =============================================================================
# 3. Rename the weights to match lecture
# =============================================================================
# Hugging Face's names are things like "transformer.h.3.mlp.c_fc.weight" (block 3's MLP
# expand matrix). This function regroups them into a dict with the names from lecture.
#
# Note on shapes: every matrix is stored as (input size, output size), so you multiply
# with  x @ W  where x has one token per ROW. That's why the lab uses (n, d) for x.
#
# Sizes for GPT-2 small:  V = 50,257 tokens,  d = 768,  4d = 3,072,  n_ctx = 1,024 positions
def _load_params(sd, n_layers=12):
    p = {
        # Embedding matrix: row i is the vector for token i. Also reused as the
        # unembedding matrix at the end ("weight tying").                      (V, d)
        "W_E":   sd["transformer.wte.weight"],
        # Positional embeddings: row t is added to whatever token is at position t.
        # 1,024 rows = GPT-2's context window.                                   (1024, d)
        "W_pos": sd["transformer.wpe.weight"],
        # Final layer norm, applied once after the last block, right before unembedding.  (d,) each
        "lnf_g": sd["transformer.ln_f.weight"],    # g = "gain": per-dimension scale
        "lnf_b": sd["transformer.ln_f.bias"],      # b = "bias": per-dimension shift
        # One dict per transformer block, filled in below.
        "blocks": [],
    }
    for i in range(n_layers):
        h = f"transformer.h.{i}."   # prefix for block i's weights
        p["blocks"].append({
            # --- Attention sub-layer ---
            # Layer norm applied to x before attention.                       (d,) each
            "ln1_g":  sd[h + "ln_1.weight"],
            "ln1_b":  sd[h + "ln_1.bias"],
            # W_Q, W_K and W_V for ALL 12 heads, stored side by side in one matrix:
            # x @ W_qkv gives [Q | K | V], each d wide.                       (d, 3d) and (3d,)
            "W_qkv":  sd[h + "attn.c_attn.weight"],
            "b_qkv":  sd[h + "attn.c_attn.bias"],
            # Output projection: mixes the 12 heads' results back together
            # after they're merged.                                           (d, d) and (d,)
            "W_O":    sd[h + "attn.c_proj.weight"],
            "b_O":    sd[h + "attn.c_proj.bias"],

            # --- MLP sub-layer ---
            # Layer norm applied to x before the MLP.                         (d,) each
            "ln2_g":  sd[h + "ln_2.weight"],
            "ln2_b":  sd[h + "ln_2.bias"],
            # Expand: d -> 4d.                                                (d, 4d) and (4d,)
            "W_up":   sd[h + "mlp.c_fc.weight"],
            "b_up":   sd[h + "mlp.c_fc.bias"],
            # Shrink back: 4d -> d.                                           (4d, d) and (d,)
            "W_down": sd[h + "mlp.c_proj.weight"],
            "b_down": sd[h + "mlp.c_proj.bias"],
        })
    return p


# `params` is what the lab files use. Examples:
#     params["W_E"]                     the embedding matrix
#     params["blocks"][0]["W_up"]       block 0's MLP expand matrix
#     len(params["blocks"])             12
params = _load_params(_sd)
print(f"Loaded: W_E {params['W_E'].shape}, W_pos {params['W_pos'].shape}, {len(params['blocks'])} blocks")


# =============================================================================
# 4. Helpers you'll call (but don't have to write)
# =============================================================================
def layer_norm(x, g, b, eps=1e-5):
    """
    Normalizes each token's vector (each ROW of x) to mean 0 and variance 1,
    then scales by g and shifts by b (both learned during training).
    Keeps the numbers in a stable range as they pass through dozens of layers.

    x: (n, d)    g, b: (d,)    returns: (n, d)
    eps is a tiny number that prevents dividing by zero.
    """
    mu = x.mean(-1, keepdims=True)           # each row's mean, shape (n, 1)
    var = x.var(-1, keepdims=True)           # each row's variance, shape (n, 1)
    return (x - mu) / np.sqrt(var + eps) * g + b


def gelu(x):
    """
    GPT-2's nonlinearity: a smooth version of ReLU.
    Roughly x for large positive x, roughly 0 for large negative x, smooth in between.
    This is the exact tanh formula GPT-2 was trained with.
    """
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))


def check(name, got, expected, atol=1e-4):
    """
    Used by the tests. Prints ✅ if `got` matches `expected` (within atol),
    otherwise ❌ plus both values so you can see what's off.
    """
    got = np.asarray(got, dtype=float)
    expected = np.asarray(expected, dtype=float)
    if got.shape != expected.shape:          # wrong shape is the most common bug, so check it first
        print(f"❌ {name}: got shape {got.shape}, expected {expected.shape}")
        return False
    if np.allclose(got, expected, atol=atol):
        print(f"✅ {name}")
        return True
    print(f"❌ {name}\n   got:      {np.round(got, 4)}\n   expected: {np.round(expected, 4)}")
    return False


def show_top_k(probs, k=5):
    """Prints the k most likely next tokens and their probabilities. probs: (V,)"""
    for i in np.argsort(probs)[::-1][:k]:    # token IDs sorted from most to least likely
        print(f"  {enc.decode([int(i)])!r:>14}  {probs[i]:6.1%}")


def run_tests(tests):
    """
    Used by test_part1.py / test_part2.py. Runs each (name, test_function) pair.
    A function you haven't written yet still raises NotImplementedError, so it's
    reported as ⏭ (skipped) instead of crashing everything after it.
    """
    for name, fn in tests:
        print(f"\n--- {name} ---")
        try:
            fn()
        except NotImplementedError:
            print("⏭  not implemented yet")
        except Exception as e:
            print(f"❌ crashed: {type(e).__name__}: {e}")


# =============================================================================
# 5. Compare YOUR model to the official GPT-2
# =============================================================================
def _gpt2_logits(ids, zero_attention=False):
    """
    Runs the official PyTorch GPT-2 on a list of token IDs and returns its logits as (n, V).

    zero_attention=True is for Part 1: your attention is a placeholder that returns zeros,
    so to compare fairly we make the official model's attention return zeros too.
    We do that with "forward hooks": small functions PyTorch calls right after a layer
    runs, which here replace that layer's output with zeros.
    """
    hooks = []
    if zero_attention:
        def zero(module, inputs, output):
            # Depending on the transformers version, attention returns a tuple
            # (output, extra info...) or just the output tensor. Zero the output either way.
            if isinstance(output, tuple):
                return (torch.zeros_like(output[0]),) + tuple(output[1:])
            return torch.zeros_like(output)
        hooks = [blk.attn.register_forward_hook(zero) for blk in _hf_model.transformer.h]

    with torch.no_grad():                                     # no gradients needed, just a forward pass
        batch = torch.tensor([ids])                           # PyTorch expects a batch: shape (1, n)
        logits = _hf_model(batch).logits[0].numpy()           # take the only item in the batch: (n, V)

    for h in hooks:                                           # put the official model back to normal
        h.remove()
    return logits


def check_against_gpt2(forward_fn, prompt="The old cat sat on the", zero_attention=False):
    """
    Runs your forward() and the official GPT-2 on the same prompt and checks the logits match.
    The tolerance is 0.01 because PyTorch and NumPy add numbers in slightly different orders,
    which causes tiny rounding differences. A real bug gives differences much bigger than that.
    """
    ids = enc.encode(prompt)
    return check(f"forward() matches GPT-2 on {prompt!r}",
                 forward_fn(ids, params), _gpt2_logits(ids, zero_attention), atol=1e-2)
