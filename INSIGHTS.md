# Insights from building napkin-pixels

Same convention as the earlier napkin repos: written in the order I hit them.

## 1. Transposed convolutions don't invert convolutions — they invert their arithmetic on a good day

The reconstruction decoder mirrored the encoder (conv 8/4 then 4/2, so deconv 4/2 then 8/4) and
produced 60×60 from a 64×64 input. Convolution output sizes floor: 64→15→6 loses a pixel of
remainder at each stage, and the mirrored transposed convs faithfully reproduce the floored
sizes, not the original ones (6→14→60). One kernel change (deconv 5/2: 6→15, then 8/4: 15→64)
fixes it. The selfcheck asserts `dec(enc(x)).shape == x.shape` rather than trusting the mirror.

**Takeaway:** encoder/decoder symmetry is an aesthetic, not an arithmetic. Assert round-trip
shapes; never derive them by symmetry.

## 2. "Truncation only happens at the end of the rollout" is false, and the bias lands on your best runs

First version stored a bootstrap observation only at the last step of each 128-step collect
window, on the reasoning that truncation = the env's step cap. But the 1000-step cap fires at
step 1000 *of the episode*, which lands anywhere inside a collect window. Every mid-window
truncation silently bootstrapped V=0 — and since only long rallies ever reach the cap, the bias
would have punished **exactly the best-performing seeds**, late in training, in the arms that
were winning. A second-opinion review caught it before the sweep did; the sweep it invalidated
was already running and got 4 runs deep.

The fix stores (t, env, observation) for every truncation wherever it lands, and the selfcheck
now shrinks the cap to 10 and asserts every truncation in a 32-step collect carries a bootstrap
observation.

**Takeaway** (napkin-diffusion #16's RL cousin): write the assert about the property you
*believe* ("truncations only happen at window ends") — it was false, and the failure mode was
invisible: no crash, no NaN, just quietly pessimistic advantages for your longest rallies.

## 3. Random features die on sparse observations — and the intuition said the opposite

The registered prediction: frozen random convs would land "embarrassingly close" to learned ones,
because the objects are indicator pixels, nearly linearly decodable — the classic random-features
argument. Measured: 0.26 rally vs 5.83, a total failure. The argument forgot *sparsity*: a 3×3
ball in a 64×64 frame is ~9 informative pixels in 4096. A random projection spreads that needle
uniformly across all features; recovering it downstream needs exactly the sharp, localized
filters that training produces and chance does not. Random features work when information is
*dense*; these frames are the opposite.

**Takeaway:** "the task is nearly linear" is a statement about the decision boundary, not about
whether a random basis preserves the signal-to-noise of a sparse input. Check sparsity before
reaching for random-features intuition.

## 4. A tie is a result: auxiliary reconstruction bought nothing

P(recon > stack4) = 0.517 over 10-seed bootstrap. Not "slightly better", not "worse" — a coin
flip, reported as such. Self-supervised auxiliaries have real wins elsewhere; at this scale, with
this much reward signal, reconstruction was redundant with what RL already learned.

## 5. Killing a background sweep with `pkill -f` — the bracket trick, again

`pkill -f "napkin_pixels.py swee[p]"` — the character class stops the pattern matching the
shell that carries it. napkin-diffusion INSIGHTS #3 documented this after getting bitten twice;
this time it worked first try. Institutional memory is a file you actually reread.
