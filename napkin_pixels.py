"""napkin-pixels: what does seeing cost, and what's the cheapest pair of glasses?

Repo 2 of the napkin-gamemaster series. Repo 1 (napkin-returns) settled the
algorithm: PPO-style data reuse does the lifting, the clip is free insurance,
and no table ships without seeds. This repo holds the algorithm fixed and
varies exactly one thing: what the agent OBSERVES.

The env is a batched-numpy mini-Pong that emits BOTH observations for the same
timestep: a privileged 5-float state (ball x, y, vx, vy, paddle y) and a 64x64
grayscale frame rendered from it. Same physics, same rewards, same seeds --
the only difference between arms is the eyes:

    state     MLP on the 5-float state          (the upper bound)
    pixels1   CNN on the current frame          (velocity is PROVABLY invisible;
                                                 selfcheck renders two states
                                                 differing only in velocity and
                                                 asserts identical frames)
    stack4    CNN on the last 4 frames          (full information, quantized)
    random    stack4 through FROZEN random convs; only the 256-unit head and
              policy/value heads train           (is learned vision even needed?)
    recon     stack4 CNN + auxiliary decoder reconstructing the input frames
              (weight 1.0)                       (does self-supervision help RL?)

Reward: +1 each paddle hit, episode ends on a miss (or 1000-step cap). Ball
speeds up 3% per hit, so returns are naturally bounded and every extra point
is genuinely harder. An episode return IS the rally length -- immediately
readable.

Usage:
    PYTHONPATH=.deps python3.10 napkin_pixels.py selfcheck
    PYTHONPATH=.deps python3.10 napkin_pixels.py train --arm stack4 --seed 0
    PYTHONPATH=.deps python3.10 napkin_pixels.py sweep        # 5 arms x 10 seeds
    PYTHONPATH=.deps python3.10 napkin_pixels.py plot
    PYTHONPATH=.deps python3.10 napkin_pixels.py gif
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

OUT = Path(__file__).parent / "out"
DEV = "cuda" if torch.cuda.is_available() else "cpu"

# env
RES = 64                 # frame resolution
BALLV = 0.030            # ball speed per step
SPEEDUP = 1.03           # per paddle hit
PSPEED = 0.045           # paddle speed per step
PHALF = 0.10             # paddle half-height
CAP = 1000               # step cap (truncation)
# ppo (winners from napkin-returns: data reuse does the lifting, clip kept as
# free insurance -- measured there as costing nothing)
GAMMA, LAM, CLIP = 0.99, 0.95, 0.2
EPOCHS, MINIBATCH, VCOEF = 4, 1024, 0.5
LR = 2.5e-4
NENVS, HORIZON = 64, 128          # 8192 env steps per update
TOTAL_STEPS = 800_000
SEEDS = 10

ARMS = ("state", "pixels1", "stack4", "random", "recon")


# ---------------------------------------------------------------- environment

class MiniPong:
    """N mini-Pong games stepped as numpy arrays. Paddle on the left; ball
    bounces off the right wall and top/bottom. Per-env RNG generators, so
    dynamics are independent of batch size (selfcheck asserts batched == single)."""

    def __init__(self, n, seed):
        self.n = n
        self.rngs = [np.random.default_rng(seed + i) for i in range(n)]
        self.x = np.zeros(n, np.float32); self.y = np.zeros(n, np.float32)
        self.vx = np.zeros(n, np.float32); self.vy = np.zeros(n, np.float32)
        self.py = np.zeros(n, np.float32)
        self.speed = np.zeros(n, np.float32)
        self.t = np.zeros(n, np.int64)
        self._serve(np.ones(n, bool))

    def _serve(self, mask):
        for i in np.flatnonzero(mask):
            ang = self.rngs[i].uniform(-np.pi / 4, np.pi / 4)
            self.x[i], self.y[i] = 0.5, self.rngs[i].uniform(0.3, 0.7)
            self.speed[i] = BALLV
            self.vx[i] = np.float32(self.speed[i] * np.cos(ang))
            self.vy[i] = np.float32(self.speed[i] * np.sin(ang))
            self.py[i] = 0.5
            self.t[i] = 0

    def state(self):
        return np.stack([self.x, self.y, self.vx / BALLV, self.vy / BALLV,
                         self.py], 1).astype(np.float32)

    def step(self, actions):
        """actions [N] in {0,1,2} = up/stay/down. Returns (reward, term, trunc).
        Call reset_done(term|trunc) after recording observations."""
        self.py = np.clip(self.py + (actions - 1) * PSPEED, PHALF, 1 - PHALF)
        self.x += self.vx
        self.y += self.vy
        # top/bottom reflect
        lo, hi = self.y < 0, self.y > 1
        self.y[lo] *= -1; self.vy[lo] *= -1
        self.y[hi] = 2 - self.y[hi]; self.vy[hi] *= -1
        # right wall reflects
        r = self.x > 1
        self.x[r] = 2 - self.x[r]; self.vx[r] *= -1
        # left edge: paddle or miss
        left = self.x < 0
        hit = left & (np.abs(self.y - self.py) <= PHALF)
        miss = left & ~hit
        rew = hit.astype(np.float32)
        if hit.any():
            self.x[hit] *= -1
            self.speed[hit] *= SPEEDUP
            self.vy[hit] = self.vy[hit] + 0.5 * self.speed[hit] * \
                ((self.y[hit] - self.py[hit]) / PHALF)
            # renormalize to current speed, keep vx positive and non-degenerate
            ang = np.arctan2(self.vy[hit], np.abs(self.vx[hit]))
            ang = np.clip(ang, -1.1, 1.1)
            self.vx[hit] = (self.speed[hit] * np.cos(ang)).astype(np.float32)
            self.vy[hit] = (self.speed[hit] * np.sin(ang)).astype(np.float32)
        self.t += 1
        trunc = (self.t >= CAP) & ~miss
        return rew, miss, trunc

    def reset_done(self, done):
        if done.any():
            self._serve(done)

    def render(self):
        """[N, RES, RES] uint8. Ball 3x3, paddle 3px wide, row 0 = y 0."""
        f = np.zeros((self.n, RES, RES), np.uint8)
        by = np.clip((self.y * (RES - 1)).round().astype(int), 0, RES - 1)
        bx = np.clip((self.x * (RES - 1)).round().astype(int), 0, RES - 1)
        n_idx = np.arange(self.n)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                f[n_idx, np.clip(by + dy, 0, RES - 1),
                  np.clip(bx + dx, 0, RES - 1)] = 255
        p0 = np.clip(((self.py - PHALF) * (RES - 1)).round().astype(int), 0, RES - 1)
        p1 = np.clip(((self.py + PHALF) * (RES - 1)).round().astype(int), 0, RES - 1)
        rows = np.arange(RES)[None]
        pmask = (rows >= p0[:, None]) & (rows <= p1[:, None])
        for c in range(3):
            f[:, :, c] |= pmask * np.uint8(180)
        return f


def decode_frame(frame):
    """Recover (ball x, y, paddle y) from one frame -- the analytic probe that
    proves the pixels contain the positional state (up to 1/RES quantization)."""
    ball = frame == 255
    ys, xs = np.nonzero(ball)
    bally, ballx = ys.mean() / (RES - 1), xs.mean() / (RES - 1)
    pad = np.nonzero(frame[:, :3] == 180)[0]
    return ballx, bally, pad.mean() / (RES - 1)


# --------------------------------------------------------------------- models

class Encoder(nn.Module):
    def __init__(self, arm):
        super().__init__()
        self.arm = arm
        if arm == "state":
            self.trunk = nn.Sequential(nn.Linear(5, 64), nn.Tanh(),
                                       nn.Linear(64, 256), nn.Tanh())
        else:
            c = 1 if arm == "pixels1" else 4
            self.conv = nn.Sequential(
                nn.Conv2d(c, 16, 8, 4), nn.ReLU(),
                nn.Conv2d(16, 32, 4, 2), nn.ReLU(), nn.Flatten())
            self.fc = nn.Sequential(nn.Linear(32 * 6 * 6, 256), nn.ReLU())
        self.pi = nn.Linear(256, 3)
        self.v = nn.Linear(256, 1)
        if arm == "recon":
            self.dec = nn.Sequential(
                nn.Linear(256, 32 * 6 * 6), nn.ReLU(), nn.Unflatten(1, (32, 6, 6)),
                nn.ConvTranspose2d(32, 16, 5, 2), nn.ReLU(),   # 6 -> 15
                nn.ConvTranspose2d(16, 4, 8, 4))               # 15 -> 64
        if arm == "random":
            for p in self.conv.parameters():
                p.requires_grad_(False)

    def features(self, x):
        if self.arm == "state":
            return self.trunk(x)
        return self.fc(self.conv(x))

    def forward(self, x):
        h = self.features(x)
        return self.pi(h), self.v(h).squeeze(-1), h


def obs_view(arm, stack, state):
    """Pick this arm's observation tensor from the stored pair."""
    if arm == "state":
        return torch.as_tensor(state, device=DEV)
    x = torch.as_tensor(stack, device=DEV).float() / 255.0
    return x[:, -1:] if arm == "pixels1" else x


# -------------------------------------------------------------------- rollout

class Runner:
    def __init__(self, seed, n=NENVS):
        self.env = MiniPong(n, seed)
        self.n = n
        frame = self.env.render()
        self.stack = np.repeat(frame[:, None], 4, axis=1)   # [N,4,R,R] uint8
        self.ep_ret = np.zeros(n)
        self.completed = []

    def collect(self, net, arm, T):
        n = self.n
        stacks = np.zeros((T, n, 4, RES, RES), np.uint8)
        states = np.zeros((T, n, 5), np.float32)
        boots = []   # (t, i, stack, state) for every truncation, wherever it lands
        act = np.zeros((T, n), np.int64)
        logp = np.zeros((T, n), np.float32)
        rew = np.zeros((T, n), np.float32)
        term = np.zeros((T, n), bool)
        trunc = np.zeros((T, n), bool)
        for t in range(T):
            stacks[t], states[t] = self.stack, self.env.state()
            with torch.no_grad():
                logits, _, _ = net(obs_view(arm, self.stack, self.env.state()))
                dist = Categorical(logits=logits)
                a = dist.sample()
                logp[t] = dist.log_prob(a).cpu().numpy()
            act[t] = a.cpu().numpy()
            rew[t], term[t], trunc[t] = self.env.step(act[t])
            self.ep_ret += rew[t]
            done = term[t] | trunc[t]
            frame = self.env.render()
            self.stack = np.concatenate([self.stack[:, 1:], frame[:, None]], 1)
            if trunc[t].any():
                st = self.env.state()   # pre-reset: still the final observation
                for i in np.flatnonzero(trunc[t]):
                    boots.append((t, i, self.stack[i].copy(), st[i].copy()))
            if done.any():
                for i in np.flatnonzero(done):
                    self.completed.append(self.ep_ret[i])
                    self.ep_ret[i] = 0.0
                self.env.reset_done(done)
                frame = self.env.render()
                self.stack[done] = frame[done, None]
        return dict(stacks=stacks, states=states, act=act, logp=logp, rew=rew,
                    term=term, trunc=trunc, boots=boots,
                    last_stack=self.stack.copy(), last_state=self.env.state())

    def pop_completed(self):
        out, self.completed = self.completed, []
        return out


# ------------------------------------------------------------------------ GAE
# verbatim from napkin-returns (its selfcheck asserted the identities)

def gae(rew, term, trunc, val, boot_val, last_val, gamma, lam):
    T, N = rew.shape
    adv = np.zeros((T, N), np.float32)
    nextadv = np.zeros(N, np.float32)
    for t in reversed(range(T)):
        nextval = last_val if t == T - 1 else val[t + 1]
        nextval = np.where(trunc[t], boot_val[t], nextval)
        nextval = np.where(term[t], 0.0, nextval)
        delta = rew[t] + gamma * nextval - val[t]
        done = term[t] | trunc[t]
        nextadv = delta + gamma * lam * (~done) * nextadv
        adv[t] = nextadv
    return adv


# --------------------------------------------------------------------- update

def update(net, opt, batch, arm):
    T, n = batch["rew"].shape
    with torch.no_grad():
        vals = []
        for t in range(0, T, 16):     # chunked value pass to bound VRAM
            o = obs_view(arm, batch["stacks"][t:t + 16].reshape(-1, 4, RES, RES),
                         batch["states"][t:t + 16].reshape(-1, 5))
            vals.append(net(o)[1].cpu().numpy())
        val = np.concatenate(vals).reshape(T, n)
        lv = net(obs_view(arm, batch["last_stack"], batch["last_state"]))[1]
        last_val = lv.cpu().numpy()
        boot_val = np.zeros((T, n), np.float32)
        if batch["boots"]:                       # truncation can land anywhere
            bts = [b[0] for b in batch["boots"]]
            bis = [b[1] for b in batch["boots"]]
            o = obs_view(arm, np.stack([b[2] for b in batch["boots"]]),
                         np.stack([b[3] for b in batch["boots"]]))
            boot_val[bts, bis] = net(o)[1].cpu().numpy()

    adv = gae(batch["rew"], batch["term"], batch["trunc"],
              val, boot_val, last_val, GAMMA, LAM)
    ret = torch.as_tensor((adv + val).reshape(-1), device=DEV)
    adv = torch.as_tensor(adv.reshape(-1), device=DEV)
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    flat_stacks = batch["stacks"].reshape(T * n, 4, RES, RES)
    flat_states = batch["states"].reshape(T * n, 5)
    act = torch.as_tensor(batch["act"].reshape(-1), device=DEV)
    logp_old = torch.as_tensor(batch["logp"].reshape(-1), device=DEV)

    for _ in range(EPOCHS):
        for idx in torch.randperm(T * n).split(MINIBATCH):
            o = obs_view(arm, flat_stacks[idx.numpy()], flat_states[idx.numpy()])
            logits, v, h = net(o)
            dist = Categorical(logits=logits)
            ratio = torch.exp(dist.log_prob(act[idx]) - logp_old[idx])
            clipped = torch.clamp(ratio, 1 - CLIP, 1 + CLIP)
            loss = -torch.min(ratio * adv[idx], clipped * adv[idx]).mean() \
                + VCOEF * ((v - ret[idx]) ** 2).mean()
            if arm == "recon":
                loss = loss + ((net.dec(h) - o) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()


# ---------------------------------------------------------------------- train

def train(arm, seed, total_steps=TOTAL_STEPS, quiet=False):
    torch.manual_seed(seed)
    np.random.seed(seed)
    net = Encoder(arm).to(DEV)
    opt = torch.optim.Adam([p for p in net.parameters() if p.requires_grad], lr=LR)
    runner = Runner(seed)
    curve, last = [], 0.0
    for upd in range(total_steps // (NENVS * HORIZON)):
        batch = runner.collect(net, arm, HORIZON)
        update(net, opt, batch, arm)
        eps = runner.pop_completed()
        last = float(np.mean(eps)) if eps else last
        curve.append((int((upd + 1) * NENVS * HORIZON), last))
        if not quiet and (upd + 1) % 5 == 0:
            print(f"  {arm} seed {seed}  steps {curve[-1][0]:>7}  "
                  f"rally {last:6.2f}", flush=True)
    return curve, net


# ---------------------------------------------------------------------- sweep

def run_sweep(total_steps, seeds):
    OUT.joinpath("sweep").mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    todo = [(a, s) for a in ARMS for s in range(seeds)]
    for k, (arm, seed) in enumerate(todo):
        f = OUT / "sweep" / f"{arm}_{seed}.json"
        if f.exists():
            continue
        curve, _ = train(arm, seed, total_steps, quiet=True)
        f.write_text(json.dumps(curve))
        print(f"[{k + 1:3}/{len(todo)}] {arm:8} seed {seed}  "
              f"final {curve[-1][1]:6.2f}  elapsed {(time.time() - t0) / 60:5.1f} min",
              flush=True)
    print("sweep done")


def load_sweep():
    runs = {}
    for f in (OUT / "sweep").glob("*.json"):
        arm, seed = f.stem.rsplit("_", 1)
        runs.setdefault(arm, {})[int(seed)] = json.loads(f.read_text())
    return runs


def iqm(x, axis=None):
    x = np.sort(np.asarray(x, np.float64), axis=axis)
    n = x.shape[-1] if axis in (None, -1) else x.shape[axis]
    lo, hi = n // 4, n - n // 4
    sl = [slice(None)] * x.ndim
    sl[-1 if axis in (None, -1) else axis] = slice(lo, hi)
    return x[tuple(sl)].mean(axis=axis)


def bootstrap_ci(x, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    stats = [iqm(rng.choice(x, size=len(x), replace=True)) for _ in range(n_boot)]
    return np.percentile(stats, [2.5, 97.5])


def make_plots():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = load_sweep()
    colors = dict(state="#333333", pixels1="#999999", stack4="#228833",
                  random="#4477aa", recon="#ee7733")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    results = {}
    for arm in ARMS:
        seeds = runs[arm]
        curves = np.array([seeds[s] for s in sorted(seeds)])
        x = curves[0, :, 0]
        for c in curves:
            axes[0].plot(c[:, 0], c[:, 1], color=colors[arm], alpha=0.12, lw=0.7)
        axes[0].plot(x, iqm(curves[:, :, 1], axis=0), color=colors[arm],
                     lw=2.2, label=arm)
        tail = curves[:, x >= 0.9 * x[-1], 1].mean(1)
        results[arm] = dict(iqm=float(iqm(tail)),
                            ci=[float(v) for v in bootstrap_ci(tail)],
                            seeds=[float(v) for v in tail])
    axes[0].set_title(f"mini-Pong from 5 observation types -- {len(runs[ARMS[0]])} "
                      "seeds each, IQM bold")
    axes[0].set_xlabel("env steps"); axes[0].set_ylabel("rally length")
    axes[0].legend(fontsize=8)

    xs = np.arange(len(ARMS))
    vals = [results[a]["iqm"] for a in ARMS]
    errs = np.array([[results[a]["iqm"] - results[a]["ci"][0],
                      results[a]["ci"][1] - results[a]["iqm"]] for a in ARMS]).T
    axes[1].bar(xs, vals, yerr=errs, capsize=4, color=[colors[a] for a in ARMS])
    axes[1].set_xticks(xs, ARMS)
    axes[1].set_title("final IQM rally length, 95% bootstrap CI")
    fig.tight_layout()
    fig.savefig(OUT / "results.png", dpi=150)
    (OUT / "results.json").write_text(json.dumps(results, indent=1))
    print(f"wrote {OUT / 'results.png'} and results.json")


# ------------------------------------------------------------------------ gif

def make_gif():
    from PIL import Image
    print(f"training stack4 for the gif ({TOTAL_STEPS} steps)...")
    _, net = train("stack4", seed=0, quiet=True)
    env = MiniPong(1, seed=123)
    frame = env.render()
    stack = np.repeat(frame[:, None], 4, axis=1)
    frames = []
    for t in range(600):
        with torch.no_grad():
            logits, _, _ = net(obs_view("stack4", stack, env.state()))
        a = logits.argmax(-1).cpu().numpy()
        _, term, trunc = env.step(a)
        f = env.render()
        stack = np.concatenate([stack[:, 1:], f[:, None]], 1)
        img = Image.fromarray(f[0]).resize((RES * 4, RES * 4), Image.NEAREST)
        frames.append(img.convert("P"))
        if term[0] or trunc[0]:
            break
    frames[0].save(OUT / "minipong.gif", save_all=True, append_images=frames[1:],
                   duration=30, loop=0)
    print(f"wrote {OUT / 'minipong.gif'}  ({len(frames)} frames)")


# ------------------------------------------------------------------ selfcheck

def selfcheck():
    # 1. batched == single: the same env stepped as a batch of 8 and as 8
    #    independent batch-of-1 envs must agree exactly (the second path that
    #    catches vectorization bugs).
    rng = np.random.default_rng(0)
    big = MiniPong(8, seed=100)
    small = [MiniPong(1, seed=100 + i) for i in range(8)]
    for t in range(600):
        a = rng.integers(0, 3, size=8)
        rew, term, trunc = big.step(a)
        for i, e in enumerate(small):
            r1, t1, tr1 = e.step(a[i:i + 1])
            assert r1[0] == rew[i] and t1[0] == term[i] and tr1[0] == trunc[i], t
            assert np.array_equal(e.render()[0], big.render()[i]), t
        big.reset_done(term | trunc)
        for i, e in enumerate(small):
            e.reset_done((term | trunc)[i:i + 1])
    print("batched env == 8 single envs, 600 steps, exact (incl. renders)")

    # 2. a frame decodes back to the state it was rendered from.
    env = MiniPong(16, seed=3)
    for _ in range(50):
        env.step(rng.integers(0, 3, size=16))
        env.reset_done(np.zeros(16, bool))
    f, s = env.render(), env.state()
    for i in range(16):
        bx, by, py = decode_frame(f[i])
        assert abs(bx - s[i, 0]) < 1.5 / RES and abs(by - s[i, 1]) < 1.5 / RES
        assert abs(py - s[i, 4]) < 1.5 / RES
    print("frame decodes to state within 1.5px (positions are IN the pixels)")

    # 3. velocity is provably invisible in a single frame: two envs at the same
    #    positions with different velocities render identically.
    e1, e2 = MiniPong(1, 5), MiniPong(1, 5)
    e2.vx[0], e2.vy[0] = -e2.vx[0], -e2.vy[0] * 0.5
    assert np.array_equal(e1.render(), e2.render())
    assert not np.array_equal(e1.state(), e2.state())
    print("single frame provably lacks velocity (pixels1's handicap is real)")

    # 4. framestack alignment: stack[t] is frames [t-3..t], oldest first.
    torch.manual_seed(0)
    r = Runner(seed=9, n=2)
    net = Encoder("state").to(DEV)
    hist = [r.stack[:, -1].copy()]
    b = r.collect(net, "state", 12)
    env2 = MiniPong(2, seed=9)     # replay the same actions to rebuild frames
    frames = [env2.render()]
    for t in range(12):
        _, te, tr = env2.step(b["act"][t])
        env2.reset_done(te | tr)
        frames.append(env2.render())
    for t in range(12):
        for k in range(4):
            src = frames[max(0, t - 3 + k)] if t - 3 + k >= 0 else frames[0]
            assert np.array_equal(b["stacks"][t][:, k], src), (t, k)
    print("framestack[t] == frames[t-3..t] (replayed via a fresh env)")

    # 5. reward fires iff the paddle covers the ball at the left edge.
    e = MiniPong(1, 7)
    e.x[0], e.y[0], e.py[0] = 0.01, 0.5, 0.5
    e.vx[0], e.vy[0] = -0.03, 0.0
    rew, term, _ = e.step(np.array([1]))
    assert rew[0] == 1.0 and not term[0]
    e.x[0], e.y[0], e.py[0] = 0.01, 0.9, 0.2
    e.vx[0], e.vy[0] = -0.03, 0.0
    rew, term, _ = e.step(np.array([1]))
    assert rew[0] == 0.0 and term[0]
    print("paddle hit -> +1 and bounce; miss -> termination")

    # 6. every truncation -- wherever it lands inside the horizon -- carries a
    #    bootstrap observation. (A first version assumed truncation could only
    #    happen at the last collect step; it can't only, and the bias would hit
    #    exactly the best-performing runs. See INSIGHTS.md.)
    global CAP
    old_cap, CAP = CAP, 10
    torch.manual_seed(0)
    b = Runner(seed=11, n=4).collect(Encoder("state").to(DEV), "state", 32)
    got = {(t, i) for t, i, _, _ in b["boots"]}
    want = set(zip(*np.nonzero(b["trunc"])))
    assert len(want) >= 4 and got == want, (len(got), len(want))
    CAP = old_cap
    print(f"all {len(want)} mid-horizon truncations carry bootstrap observations")

    # 7. all five arms overfit a tiny budget (end-to-end machinery), and the
    #    recon decoder's output shape matches its input.
    net = Encoder("recon").to(DEV)
    x = torch.zeros(2, 4, RES, RES, device=DEV)
    assert net.dec(net.features(x)).shape == x.shape
    for arm in ARMS:
        torch.manual_seed(0)
        curve, _ = train(arm, seed=0, total_steps=NENVS * HORIZON * 4, quiet=True)
        assert np.isfinite(curve[-1][1]), arm
    print("recon decoder shape ok; all arms run 4 updates without NaNs")

    print("\nall selfchecks passed")


# ------------------------------------------------------------------------ cli

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("selfcheck")
    t = sub.add_parser("train")
    t.add_argument("--arm", choices=ARMS, default="stack4")
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--steps", type=int, default=TOTAL_STEPS)
    s = sub.add_parser("sweep")
    s.add_argument("--steps", type=int, default=TOTAL_STEPS)
    s.add_argument("--seeds", type=int, default=SEEDS)
    sub.add_parser("plot")
    sub.add_parser("gif")
    args = ap.parse_args()

    OUT.mkdir(exist_ok=True)
    if args.cmd == "selfcheck":
        selfcheck()
    elif args.cmd == "train":
        t0 = time.time()
        curve, _ = train(args.arm, args.seed, args.steps)
        print(f"final rally {curve[-1][1]:.2f}  ({time.time() - t0:.0f}s)")
    elif args.cmd == "sweep":
        run_sweep(args.steps, args.seeds)
    elif args.cmd == "plot":
        make_plots()
    elif args.cmd == "gif":
        make_gif()
