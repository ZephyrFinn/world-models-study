"""Build the talk deck as a single self-contained HTML file.

Figures are inlined as data URIs so the deck is one file you can open
anywhere, present from, or print to PDF. Re-run after regenerating figures
(e.g. once the DreamerV3 run finishes) and the numbers on the last slides
pick up the new values.

Usage: python build_slides.py   ->  slides.html
"""
import base64
import json
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.parent
FIG = ROOT / "04_analysis" / "figures"
D1 = ROOT / "01_dreamer_from_scratch" / "results"
METRICS = Path.home() / "workspace/WAM/dreamerv3-torch/logdir/dmc_walker_walk/metrics.jsonl"


def img(path):
    b64 = base64.b64encode(Path(path).read_bytes()).decode()
    return f"data:image/png;base64,{b64}"


def dreamerv3_status():
    """Latest eval point, so the deck never quotes a stale number."""
    try:
        steps, rets = [], []
        for line in open(METRICS):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "eval_return" in r:
                steps.append(r["step"]); rets.append(r["eval_return"])
        if steps:
            return steps[-1], rets[-1]
    except FileNotFoundError:
        pass
    return None, None

STEP, RET = dreamerv3_status()
v3_headline = f"{RET:.0f}" if RET else "--"
v3_sub = (f"eval return at {STEP/1000:.0f}k / 1M env steps"
          if STEP else "run not found")
if STEP and STEP >= 1_000_000:
    v3_note = "Full 1M-step run complete."
elif RET and RET > 900:
    v3_note = ("Plateaued near the task ceiling (~1000) from roughly 300k steps on. "
               "Run continues to 1M.")
else:
    v3_note = "Run still climbing when this deck was built. Rebuild to refresh."

SLIDES = [
    # ---------------------------------------------------------------- 1
    f"""<section class="title">
      <p class="eyebrow">Model-based reinforcement learning</p>
      <h1>Two ways to build a world model<br><span class="thin">and what happened when I tested one</span></h1>
      <p class="byline">Dreamer (generative RSSM) &nbsp;·&nbsp; LeWorldModel (non-generative JEPA)</p>
    </section>""",

    # ---------------------------------------------------------------- 2
    """<section>
      <h2>The split</h2>
      <div class="two">
        <div class="pane">
          <p class="pane-label">Dreamer</p>
          <p class="pane-claim">"I understand it if I can draw it."</p>
          <p>Learns latent dynamics with a decoder — the model is trained to
          reconstruct pixels. Then trains a policy entirely inside its own
          imagination.</p>
          <p class="dim">Cost: spends capacity predicting every leaf of
          background texture, none of which affects the decision.</p>
        </div>
        <div class="pane accent">
          <p class="pane-label">JEPA / LeWorldModel</p>
          <p class="pane-claim">"Predicting pixels is the wrong target."</p>
          <p>No decoder at all. Encodes observations to embeddings and predicts
          the <em>next embedding</em>. Plans at test time by searching action
          sequences (CEM).</p>
          <p class="dim">Cost: nothing stops the encoder from collapsing to a
          constant. Something has to.</p>
        </div>
      </div>
      <p class="aside">They do not share an objective (reward vs. distance-to-goal)
      or a metric (return vs. success rate). That matters later.</p>
    </section>""",

    # ---------------------------------------------------------------- 3
    """<section>
      <h2>What I did</h2>
      <div class="three">
        <div class="card">
          <p class="n">1</p>
          <p class="card-h">Built Dreamer</p>
          <p>From scratch, one file. RSSM, reward head, actor-critic,
          imagination rollout.</p>
        </div>
        <div class="card">
          <p class="n">2</p>
          <p class="card-h">Ran LeWorldModel</p>
          <p>Reproduced the released checkpoint: 86% planning success.</p>
        </div>
        <div class="card accent">
          <p class="n">3</p>
          <p class="card-h">Then tested it</p>
          <p>Four controlled experiments. The fourth one overturns the
          first three &mdash; this is the part worth talking about.</p>
        </div>
      </div>
    </section>""",

    # ---------------------------------------------------------------- 4
    f"""<section>
      <h2>Dreamer, written by hand</h2>
      <p class="lead">~670 lines, no framework. GRU deterministic state + 32&times;16
      categorical stochastic state, straight-through gradients, &lambda;-return
      actor-critic trained only on imagined rollouts.</p>
      <img src="{img(D1/'comparison_multiseed.png')}" alt="Dreamer vs DQN, 3 seeds">
      <div class="row">
        <div><span class="big">240.1</span><span class="unit">Dreamer mean return</span></div>
        <div><span class="big dim">113.8</span><span class="unit">Double-DQN</span></div>
        <div><span class="unit">CartPole-v1 · 20k steps · 3 seeds</span></div>
      </div>
    </section>""",

    # ---------------------------------------------------------------- 5
    f"""<section>
      <h2>Does the imagination correspond to anything?</h2>
      <p class="lead">The actor never sees a real state. So the question that
      matters is whether the imagined rollouts track reality &mdash; or whether the
      model is hallucinating a comfortable world and the policy is optimising
      against a fantasy.</p>
      <img src="{img(D1/'diagnostics.png')}" alt="KL and imagination lambda-return diagnostics">
      <p class="aside">KL holds at ~0.8 with free bits binding (no posterior
      collapse, no blow-up). Imagined &lambda;-return climbs 0 &rarr; 70+ in step with
      real eval return.</p>
    </section>""",

    # ---------------------------------------------------------------- 6
    """<section>
      <h2>LeWorldModel, reproduced</h2>
      <div class="row wide">
        <div><span class="big">86%</span><span class="unit">planning success, released checkpoint</span></div>
        <div><span class="big dim">~4h</span><span class="unit">spent on the install, not the science</span></div>
      </div>
      <p class="lead">CEM-MPC over 50 held-out episodes on tworoom. This is the
      reference number everything after this slide is read against.</p>
      <p class="aside">Four separate breaks between the repo README and the
      published package: a missing <code>swig</code>, <code>datasets</code>
      resolving five years stale, <code>transformers</code> 5.x renaming every
      ViT weight so the checkpoints will not load, and HDF5 support disabled by
      a silent import failure. All written up &mdash; that is the part reproduction
      write-ups usually omit.</p>
    </section>""",

    # ---------------------------------------------------------------- 7
    """<section class="divider">
      <p class="eyebrow">Everything above is reproduction</p>
      <h1>Everything below is mine</h1>
      <p class="byline">Same data, same encoder, same optimizer, same 900 gradient
      steps, same 50 held-out episodes.<br>One variable per experiment &mdash; or so I thought.</p>
    </section>""",

    # ---------------------------------------------------------------- 8
    f"""<section>
      <h2>Experiment 1 &mdash; how much context?</h2>
      <p class="lead"><code>history_size</code> &isin; {{1, 3, 5}} &mdash; how many past
      embeddings the predictor sees. 3 is the published default.</p>
      <img src="{img(FIG/'exp1_horizon.png')}" alt="horizon sweep">
      <p class="aside"><strong>Loss falls with context. Success does not follow.</strong>
      h=1 &rarr; h=3 improves loss and <em>loses</em> two points of success rate. At 50
      episodes the standard error is ~7pp, so that pair is noise &mdash; the claim
      I left standing was that h=5 helps. Remember this success ordering &mdash;
      <strong>52 / 54 / 64</strong> &mdash; slide 14 comes back for it.</p>
    </section>""",

    # ---------------------------------------------------------------- 9
    f"""<section>
      <h2>Experiment 2 &mdash; wrong variable, run again</h2>
      <p class="lead">The plan was "hold the budget, change only the architecture."
      <strong>The design itself was wrong:</strong> JEPA is a family of methods that
      share an architecture and differ in the anti-collapse loss. I swapped the model
      class while the loss stayed LeWM's SIGReg &mdash; <strong>both arms were the same
      method</strong> (both training logs report <code>sigreg_loss</code>; a real PLDM
      run would not).</p>
      <p class="lead">Redone: architecture, data, budget and planner fixed,
      <strong>only the anti-collapse mechanism swapped</strong> (SIGReg &times;1 against
      VCReg &times;4 + temporal alignment + inverse dynamics = 6 terms), the missing
      <code>Manager</code> seed restored, three seeds per arm.</p>
      <img src="{img(FIG/'exp2_seed_variance.png')}" alt="three seeds per arm" style="max-height:240px">
      <p class="aside"><strong>LeWM 55.3% against PLDM 43.3%, t = 0.87, CI [-27, +51]</strong>
      &mdash; the direction the paper claims, nowhere near significant. The real output is
      that standard deviation: seed alone moves success from 36% to 70%,
      <strong>&plusmn;17 points</strong>. Detecting a true 10-point difference at 80% power
      needs about 45 seeds per arm. I had one.</p>
    </section>""",

    # ---------------------------------------------------------------- 10
    f"""<section>
      <h2>Experiment 3 &mdash; so I stopped trusting the loss</h2>
      <p class="lead">LeWorldModel's claim: two loss terms suffice, because
      SIGReg keeps the embedding from collapsing. I swept its weight to 0.001
      and 1.0 &mdash; and measured the embedding directly instead of reading the loss.</p>
      <img src="{img(FIG/'exp3_sigreg.png')}" alt="sigreg sweep and collapse diagnostic">
      <p class="aside">At weight 0.001: <strong>pred_loss 0.004</strong> &mdash; the lowest
      number I produced all night, 60&times; better than normal &mdash; and
      <strong>69 of 192 embedding dimensions are dead</strong>. A collapsed
      encoder makes prediction trivial. The metric inverts.</p>
    </section>""",

    # ---------------------------------------------------------------- 11
    """<section>
      <h2>The regularizer wasn't broken. It was outvoted.</h2>
      <table>
        <tr><th>&lambda;</th><th>pred_loss</th><th>sigreg_loss</th><th>emb std</th><th>dead</th><th>success</th></tr>
        <tr class="bad"><td>0.001</td><td>0.004</td><td>50.75</td><td>0.0012</td><td>69/192</td><td>34%</td></tr>
        <tr class="hi"><td>0.09 &#9733;</td><td>0.266</td><td>5.16</td><td>0.325</td><td>0</td><td>52%</td></tr>
        <tr><td>1.0</td><td>1.074</td><td>11.77</td><td>0.115</td><td>0</td><td>38%</td></tr>
      </table>
      <p class="lead">SIGReg was screaming the whole time &mdash; 50.75, ten times its
      healthy value. But at weight 0.001 its contribution to the total was
      0.001 &times; 50.75 &asymp; 0.05, while collapsing bought the model 0.26 of
      prediction error. It got outbid.</p>
      <p class="aside">The other end over-regularises: no collapse, but the
      embedding is squeezed too tight to stay discriminative. The published
      default sits at the peak of the inverted U. Remember this shape too:
      <strong>34 / 52 / 38</strong>.</p>
    </section>""",

    # ---------------------------------------------------------------- 12
    f"""<section>
      <h2>Six checkpoints, no trend line</h2>
      <img src="{img(FIG/'loss_vs_success.png')}" alt="prediction loss vs planning success, six checkpoints">
      <p class="aside">If validation loss were the right proxy this would trend
      down-and-to-the-right across three orders of magnitude. <strong>It doesn't.</strong>
      Which leaves me one metric, whose noise is &plusmn;17 points. Rather than buy more
      seeds to brute-force that noise, change the question:
      <strong>what decides whether planning works at all?</strong></p>
    </section>""",

    # ------------------- 13 experiment 4, the metrics
    f"""<section>
      <h2>Experiment 4 &mdash; metrics with far less noise</h2>
      <p class="lead">No new training: on the 12 checkpoints that already exist, over
      3000 held-out frames, measure four things &mdash; Epps-Pulley normality
      (<strong>what SIGReg claims to optimise</strong>), held-out R&sup2; of a ridge probe
      from embedding to agent position (the paper's own probing), effective rank, and
      the embedding's absolute scale. Calibrated first on known distributions: a true
      Gaussian gives 0.5, a 4-D manifold in 192-D gives 170.6.</p>
      <img src="{img(FIG/'exp4_scale_vs_success.png')}" alt="embedding scale vs planning success" style="max-height:255px">
      <p class="aside"><strong>How much physical information the representation carries
      barely relates to whether it can plan. Its absolute scale is what decides.</strong>
      Scale r = +0.89 (t = 5.85); probe R&sup2; +0.24, effective rank &minus;0.22, distance
      from Gaussian &minus;0.40 &mdash; none significant. CEM picks actions by comparing
      distances in embedding space; shrink the signal while the predictor's error stays
      put and the cost landscape drowns in noise. This also overturns my own experiment 3
      explanation: the collapsed checkpoint probes at R&sup2; = 0.467 against 0.497 for a
      healthy one &mdash; <strong>collapse costs signal-to-noise, not information.</strong></p>
    </section>""",

    # ------------------- 14 experiment 4, the confound
    f"""<section>
      <h2>So: the first three measured one variable</h2>
      <img src="{img(FIG/'exp4_confound.png')}" alt="the confound, and SIGReg's own statistic">
      <p class="lead">Experiment 1's success ordering (52 / 54 / 64) matches its scale
      ordering (0.319 / 0.348 / 0.389) exactly; experiment 3's inverted U in success
      (34 / 52 / 38) is the same shape as its inverted U in scale
      (0.0012 / 0.319 / 0.116). Three different knobs, one mediating variable.
      <strong>More seeds would not have rescued them &mdash; 45 seeds buys a precise
      estimate of a confounded quantity. The problem is the design, not the power.</strong></p>
      <p class="aside">Incidental finding (right): across the whole 1000&times; weight
      sweep SIGReg's Gaussianity statistic reads <strong>1206 / 1206 / 1206</strong> &mdash;
      unmoved, against 0.5 for a true Gaussian and 170 for a 4-D linear manifold. What it
      actually maintains is embedding scale, not Gaussianity. The mechanism the paper
      names and the mechanism doing the work are not the same.</p>
    </section>""",

    # ---------------------------------------------------------------- 13
    f"""<section>
      <h2>Scale check &mdash; DreamerV3 on DMC walker-walk</h2>
      <p class="lead">The from-scratch build proves I understand the components.
      This is the official implementation on a real benchmark, for a real curve.</p>
      <img src="{img(FIG/'dreamerv3_walker.png')}" alt="DreamerV3 learning curve">
      <div class="row">
        <div><span class="big">{v3_headline}</span><span class="unit">{v3_sub}</span></div>
        <div><span class="unit">{v3_note}</span></div>
      </div>
    </section>""",

    # ---------------------------------------------------------------- 14
    """<section>
      <h2>What actually cost the time</h2>
      <ul class="log">
        <li><strong>Symlinked torch into a shared venv.</strong> uv decided an
        unpinned <code>torch&gt;=2</code> wanted upgrading and wrote
        <em>through</em> the symlink &mdash; silently upgrading another project's
        install in place while a day-long job was running against it.</li>
        <li><strong><code>embed_dim</code> is not a free hyperparameter.</strong>
        The projector reads its input width from it while ViT-tiny is fixed at
        192. My first planned ablation crashed on a shape mismatch; sweeping it
        properly would mean co-scaling the backbone, which stops being a clean
        single-variable ablation.</li>
        <li><strong>stdout buffering hid two hours of progress.</strong> The
        console log sat at step 45000 for over an hour while
        <code>metrics.jsonl</code> had already passed 80000.</li>
      </ul>
      <p class="aside">Full log in the repo. Included because the parts that
      break are the parts nobody writes down.</p>
    </section>""",

    # ------------------- 17 limits
    """<section>
      <h2>What this is, and what it isn't</h2>
      <div class="two">
        <div class="pane accent">
          <p class="pane-label">Holds up (all direct measurements)</p>
          <p>The collapse itself: embedding SD 0.0012 against 0.325, 270&times; apart;
          69 dead dimensions against 0. No room for a noise explanation.</p>
          <p>Scale against planning success: r = 0.89, t = 5.85, n = 11.</p>
          <p>SIGReg's Gaussianity statistic unmoved across a 1000&times; weight sweep
          (1206 / 1206 / 1206).</p>
        </div>
        <div class="pane">
          <p class="pane-label">Doesn't</p>
          <p>The success-rate conclusions of the first three experiments are
          <strong>all withdrawn</strong> &mdash; they measured one confound.</p>
          <p>Experiment 4 is <strong>observational, not interventional</strong>. Proving
          scale <em>causes</em> planning quality means rescaling embeddings at evaluation
          time and watching success follow. That is the obvious next step; I did not run it.</p>
          <p>Everything is a 900-step undertrained model. The released checkpoint has
          scale 0.032 and still scores 86% &mdash; the relationship stops holding once a
          model is trained to convergence.</p>
          <p>No head-to-head Dreamer vs LeWorldModel: different objectives, different metrics.</p>
        </div>
      </div>
      <p class="lead">The next-step priority got rewritten twice: first from "sweep more
      hyperparameters" to "make the evaluation big enough," then from there to "find out
      what the primary metric is actually tracking."
      <strong>Done again, experiment 4 would be experiment 1.</strong></p>
    </section>""",
]

FOOTERS = [
    "", "", "",
    "01_dreamer_from_scratch/",
    "01_dreamer_from_scratch/results/diagnostics.png",
    "02_lewm_reproduction/setup.md",
    "03_experiments/",
    "03_experiments/exp1_horizon/",
    "03_experiments/exp2_pldm/",
    "03_experiments/exp3_sigreg/",
    "03_experiments/exp3_sigreg/probe_collapse.py",
    "04_analysis/summary.csv",
    "03_experiments/exp4_representation/",
    "03_experiments/exp4_representation/",
    "04_analysis/dreamerv3_curve.py",
    "notes/engineering_log.md",
    "README.md",
]

CSS = """
*{box-sizing:border-box;margin:0;padding:0}
:root{
  --ink:#17150f; --ink2:#4a463b; --dim:#8d8779;
  --bg:#faf9f5; --card:#fff; --line:#e0dcd0;
  --accent:#2a78d6; --good:#1a8a4a; --bad:#c8402e;
}
body{background:#e8e5dc;font-family:"IBM Plex Sans",-apple-system,"PingFang SC",sans-serif;color:var(--ink)}
section{
  width:100vw;height:100vh;padding:6vh 7vw;background:var(--bg);
  display:flex;flex-direction:column;justify-content:center;gap:2.2vh;
  page-break-after:always;position:relative;overflow:hidden;
}
.eyebrow{font-family:"IBM Plex Mono",monospace;font-size:1.5vh;letter-spacing:.18em;
  text-transform:uppercase;color:var(--accent)}
h1{font-size:5.6vh;line-height:1.12;font-weight:700;letter-spacing:-.02em}
h1 .thin{font-weight:400;color:var(--ink2)}
h2{font-size:4vh;font-weight:600;letter-spacing:-.015em;line-height:1.15}
.byline{font-size:2vh;color:var(--ink2);line-height:1.5}
.lead{font-size:2.15vh;line-height:1.55;color:var(--ink2);max-width:62ch}
.aside{font-size:1.75vh;line-height:1.5;color:var(--dim);max-width:74ch}
.aside strong{color:var(--ink)}
.dim{color:var(--dim)}
code{font-family:"IBM Plex Mono",monospace;font-size:.92em;background:#efece2;
  padding:.1em .35em;border-radius:3px}
img{max-width:100%;max-height:46vh;object-fit:contain;align-self:center}

.title,.divider{justify-content:center;gap:3vh}
.divider{background:var(--ink);color:var(--bg)}
.divider h1{color:#fff;font-size:6.4vh}
.divider .eyebrow{color:#8fb8e8}
.divider .byline{color:#b8b3a5}

.two{display:grid;grid-template-columns:1fr 1fr;gap:2.5vw}
.three{display:grid;grid-template-columns:repeat(3,1fr);gap:1.8vw}
.pane,.card{background:var(--card);border:1px solid var(--line);border-radius:10px;
  padding:2.6vh 2vw;display:flex;flex-direction:column;gap:1.2vh}
.pane p,.card p{font-size:1.85vh;line-height:1.5;color:var(--ink2)}
.pane.accent,.card.accent{border-color:var(--accent);box-shadow:0 0 0 1px var(--accent)}
.pane-label{font-family:"IBM Plex Mono",monospace;font-size:1.4vh;letter-spacing:.1em;
  text-transform:uppercase;color:var(--accent) !important}
.pane-claim{font-size:2.3vh !important;font-weight:600;color:var(--ink) !important;line-height:1.3}
.card .n{font-family:"IBM Plex Mono",monospace;font-size:3vh;color:var(--dim) !important;line-height:1}
.card-h{font-size:2.1vh !important;font-weight:600;color:var(--ink) !important}

.row{display:flex;gap:4vw;align-items:baseline;flex-wrap:wrap}
.row.wide{gap:6vw;margin:1vh 0}
.big{font-family:"IBM Plex Mono",monospace;font-size:5.5vh;font-weight:600;
  display:block;line-height:1}
.unit{font-size:1.7vh;color:var(--dim);display:block;margin-top:.6vh}

table{border-collapse:collapse;font-size:2vh;width:100%;max-width:60vw}
th{font-family:"IBM Plex Mono",monospace;font-size:1.4vh;letter-spacing:.06em;
  text-transform:uppercase;color:var(--dim);font-weight:500;text-align:right;
  padding:.9vh 1.4vw;border-bottom:1px solid var(--line)}
th:first-child,td:first-child{text-align:left}
td{padding:1.1vh 1.4vw;border-bottom:1px solid var(--line);
  font-variant-numeric:tabular-nums;text-align:right}
tr.hi{background:#e8f3ea}tr.hi td{font-weight:600}
tr.bad td{color:var(--bad)}
.better{color:var(--good);font-size:.72em;margin-left:.4em}
.worse{color:var(--bad);font-size:.72em;margin-left:.4em}

ul.log{list-style:none;display:flex;flex-direction:column;gap:1.8vh;max-width:76ch}
ul.log li{font-size:1.95vh;line-height:1.5;color:var(--ink2);
  padding-left:1.6vw;border-left:2px solid var(--line)}
ul.log strong{color:var(--ink)}

.foot{position:absolute;bottom:3vh;left:7vw;right:7vw;display:flex;
  justify-content:space-between;font-family:"IBM Plex Mono",monospace;
  font-size:1.35vh;color:var(--dim)}
.divider .foot{color:#6f6a5e}

@page{size:1280px 720px;margin:0}
@media print{
  body{background:#fff}
  section{margin:0;box-shadow:none;page-break-after:always;break-after:page}
  section:last-child{page-break-after:auto}
}
"""

JS = """
const slides=[...document.querySelectorAll('section')];
let i=0;
const go=n=>{i=Math.max(0,Math.min(slides.length-1,n));
  slides[i].scrollIntoView({behavior:'smooth'});};
addEventListener('keydown',e=>{
  if(['ArrowRight','ArrowDown',' ','PageDown'].includes(e.key)){e.preventDefault();go(i+1)}
  if(['ArrowLeft','ArrowUp','PageUp'].includes(e.key)){e.preventDefault();go(i-1)}
  if(e.key==='Home'){go(0)} if(e.key==='End'){go(slides.length-1)}
});
new IntersectionObserver(es=>es.forEach(e=>{
  if(e.isIntersecting) i=slides.indexOf(e.target);
}),{threshold:.5}).observe && slides.forEach(s=>
  new IntersectionObserver(es=>es.forEach(e=>{
    if(e.isIntersecting) i=slides.indexOf(e.target);
  }),{threshold:.5}).observe(s));
"""


def build():
    parts = []
    for n, (html, foot) in enumerate(zip(SLIDES, FOOTERS), start=1):
        footer = (f'<div class="foot"><span>{foot}</span>'
                  f'<span>{n} / {len(SLIDES)}</span></div>')
        parts.append(html.replace("</section>", footer + "</section>"))

    doc = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>World models: two approaches, and one that lies</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>{CSS}</style></head>
<body>
{"".join(parts)}
<script>{JS}</script>
</body></html>"""

    out = HERE / "slides.html"
    out.write_text(doc, encoding="utf-8")
    print(f"{len(SLIDES)} slides -> {out}  ({out.stat().st_size/1e6:.1f} MB)")
    if STEP:
        print(f"DreamerV3 slide shows {RET:.0f} at step {STEP}")


if __name__ == "__main__":
    build()
