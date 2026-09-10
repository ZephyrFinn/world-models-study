"""中文版讲稿 deck。

和 build_slides.py 同一套数据、同一套版式，文案换成中文，字体栈改用
Noto Sans CJK，页面尺寸按 16:9 固定（而不是 vh），这样 Chrome 打印出来的
PDF 每页正好一张幻灯片。

用法：
    python build_slides_zh.py            -> slides_zh.html
    python build_slides_zh.py --pdf      -> 顺便调 Chrome 导出 slides_zh.pdf
"""
import base64
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.parent
FIG = ROOT / "04_analysis" / "figures"
D1 = ROOT / "01_dreamer_from_scratch" / "results"
METRICS = Path.home() / "workspace/WAM/dreamerv3-torch/logdir/dmc_walker_walk/metrics.jsonl"

# 16:9，按 px 固定，便于打印分页
W, H = 1280, 720


def img(path):
    return "data:image/png;base64," + base64.b64encode(Path(path).read_bytes()).decode()


def v3_status():
    try:
        pts = []
        for line in open(METRICS):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "eval_return" in r:
                pts.append((r["step"], r["eval_return"]))
        return pts
    except FileNotFoundError:
        return []


PTS = v3_status()
STEP, RET = (PTS[-1] if PTS else (None, None))
tail = [r for _, r in PTS[-8:]]
plateaued = len(tail) >= 8 and (max(tail) - min(tail)) < 0.08 * max(tail)
if STEP and STEP >= 1_000_000:
    v3_note = "1M 步完整跑完。"
elif plateaued:
    v3_note = "约 30 万步起进入平台期，稳定在满分（约 1000）附近。训练继续跑向 1M。"
else:
    v3_note = "曲线仍在上升，重新生成本 deck 可刷新数值。"

SLIDES = [
    # 1 封面
    """<section class="title">
      <p class="eyebrow">基于模型的强化学习 · Model-based RL</p>
      <h1>世界模型的两条路线<br><span class="thin">以及我把其中一条拆开验证之后</span></h1>
      <p class="byline">Dreamer（生成式 RSSM） &nbsp;·&nbsp; LeWorldModel（非生成式 JEPA）</p>
    </section>""",

    # 2 两条路线
    """<section>
      <h2>分歧在哪</h2>
      <div class="two">
        <div class="pane">
          <p class="pane-label">Dreamer</p>
          <p class="pane-claim">“能画出来，才算真懂。”</p>
          <p>世界模型带 decoder，训练目标包含把潜在状态还原成像素。然后让策略完全在这个模型的想象里训练。</p>
          <p class="dim">代价：为了预测球往哪飞，顺带得学会画背景里每一片草叶。</p>
        </div>
        <div class="pane accent">
          <p class="pane-label">JEPA / LeWorldModel</p>
          <p class="pane-claim">“预测像素本身就是错的目标。”</p>
          <p>完全没有 decoder。把观测编码成向量，只预测<em>下一个向量</em>。规划放到测试时做——CEM 搜索动作序列。</p>
          <p class="dim">代价：没有任何东西阻止 encoder 塌缩成一个常数。必须额外加约束。</p>
        </div>
      </div>
      <p class="aside">两者不共享优化目标（奖励 vs 与目标的距离），也不共享评估指标（累计回报 vs 成功率）。这一点后面还会回来。</p>
    </section>""",

    # 3 总览
    """<section>
      <h2>我做了什么</h2>
      <div class="three">
        <div class="card">
          <p class="n">1</p>
          <p class="card-h">手写 Dreamer</p>
          <p>从零实现，单文件。RSSM、reward model、actor-critic、imagination rollout。</p>
        </div>
        <div class="card">
          <p class="n">2</p>
          <p class="card-h">复现 LeWorldModel</p>
          <p>跑通官方实现，复现出 86% 的规划成功率。</p>
        </div>
        <div class="card accent">
          <p class="n">3</p>
          <p class="card-h">然后去验证它</p>
          <p>四组受控实验。第四组推翻了前三组的结论——这才是值得讲的部分。</p>
        </div>
      </div>
    </section>""",

    # 4 Dreamer 手写
    f"""<section>
      <h2>Dreamer，一行行写出来</h2>
      <p class="lead">约 670 行，不依赖任何框架。GRU 确定性状态 + 32×16 categorical 随机状态，
      straight-through 梯度，actor-critic 完全在想象轨迹上用 λ-return 训练。</p>
      <img src="{img(D1/'comparison_multiseed.png')}" alt="Dreamer vs DQN">
      <div class="row">
        <div><span class="big">240.1</span><span class="unit">Dreamer 平均回报</span></div>
        <div><span class="big dim">113.8</span><span class="unit">Double-DQN</span></div>
        <div><span class="unit">CartPole-v1 · 2 万步 · 3 个 seed</span></div>
      </div>
    </section>""",

    # 5 诊断
    f"""<section>
      <h2>想象出来的东西，对应现实吗？</h2>
      <p class="lead">actor 从头到尾没见过一个真实状态。所以关键问题是：想象轨迹到底跟不跟现实对齐，
      还是说世界模型编了一个奖励很高但不存在的世界，策略在里面自我陶醉。</p>
      <img src="{img(D1/'diagnostics.png')}" alt="KL 与想象 λ-return 诊断">
      <p class="aside">KL 稳定在 0.8 附近，free bits 正在起作用（既没后验塌缩也没爆炸）。
      想象中算出的 λ-return 从 0 涨到 70+，与真实评估回报同步——这就是"没有自欺"的证据。</p>
    </section>""",

    # 6 LeWM 复现
    """<section>
      <h2>LeWorldModel，复现</h2>
      <div class="row wide">
        <div><span class="big">86%</span><span class="unit">官方 checkpoint 的规划成功率</span></div>
        <div><span class="big dim">约 4 小时</span><span class="unit">花在装环境上，不是花在科研上</span></div>
      </div>
      <p class="lead">tworoom 任务，CEM-MPC，50 条留出 episode。这是后面所有数字的参照基准。</p>
      <p class="aside">官方 README 和已发布的包之间有四处断裂：<code>box2d-py</code> 缺 <code>swig</code> 编译不过、
      <code>datasets</code> 解析到五年前的版本、<code>transformers</code> 5.x 重命名了所有 ViT 权重导致 checkpoint 加载失败、
      HDF5 支持被一个静默的 import 失败关掉了。这些我全写进了仓库——复现类文章通常略过这段，而这段恰恰是真正卡住人的地方。</p>
    </section>""",

    # 7 分界
    """<section class="divider">
      <p class="eyebrow">以上是复现</p>
      <h1>以下是我自己做的</h1>
      <p class="byline">同一份数据、同一个 encoder、同一个优化器、同样 900 步梯度更新、同样 50 条留出 episode。<br>每组实验只动一个变量——至少我当时是这么以为的。</p>
    </section>""",

    # 8 实验一
    f"""<section>
      <h2>实验一 —— 需要看多少历史？</h2>
      <p class="lead"><code>history_size</code> ∈ {{1, 3, 5}}，即 predictor 能看到几帧过去的 embedding。3 是论文默认值。</p>
      <img src="{img(FIG/'exp1_horizon.png')}" alt="horizon 扫描">
      <p class="aside"><strong>loss 随上下文单调下降，成功率却没跟上。</strong>
      h=1 → h=3 这一步，loss 改善了，成功率反而掉了 2 个点。50 条 episode 的标准误约 7 个百分点，
      所以这一对<strong>就是噪声</strong>，当时我留下的结论只有"h=5 更好"。
      记住这个成功率排序：<strong>52 / 54 / 64</strong> —— 第 14 页会回来推翻它。</p>
    </section>""",

    # 9 实验二
    f"""<section>
      <h2>实验二 —— 变量选错了，重做一次</h2>
      <p class="lead">原计划是"固定预算只换架构"。<strong>但这个设计本身就是错的：</strong>JEPA 是一个方法家族，
      成员共享架构，区别在 anti-collapse loss。我只换了模型类、loss 仍然是 LeWM 的 SIGReg——<strong>两臂跑的是同一个方法</strong>
      （两份训练日志都在报 <code>sigreg_loss</code>，真 PLDM 不该有这一项）。</p>
      <p class="lead">重做：架构、数据、预算、规划器全部固定，<strong>只换防塌缩机制</strong>
      （SIGReg×1 对 VCReg×4 + 时序对齐 + 逆动力学 = 6 项），补上 <code>Manager</code> 的 seed，每臂 3 个 seed。</p>
      <img src="{img(FIG/'exp2_seed_variance.png')}" style="max-height:240px" alt="每臂三个 seed 的成功率分布">
      <p class="aside"><strong>LeWM 55.3% 对 PLDM 43.3%，t = 0.87，CI [-27, +51]</strong>——方向和论文一致，但完全不显著。
      真正的产出是那个标准差：只换 seed，成功率能从 36% 走到 70%，<strong>±17 个百分点</strong>。
      要在 80% 功效下测出 10 个点的差异，每臂约需 45 个 seed。我有 1 个。</p>
    </section>""",

    # 10 实验三
    f"""<section>
      <h2>实验三 —— 于是我不再信 loss</h2>
      <p class="lead">LeWorldModel 的核心主张：两项 loss 就够，因为 SIGReg 会防止表征塌缩。
      我把它的权重扫到 0.001 和 1.0，并且<strong>没有去读 loss，而是直接测量 embedding 本身</strong>。</p>
      <img src="{img(FIG/'exp3_sigreg.png')}" alt="SIGReg 扫描与塌缩诊断">
      <p class="aside">权重降到 0.001 时：<strong>pred_loss = 0.004</strong>，是我整晚训出来的最低值，
      比正常模型好 60 倍——同时 <strong>192 维里有 69 维已经死掉</strong>。
      encoder 塌缩会让预测任务变得极其简单，这个指标在这里是<strong>反向的</strong>。
      成功率 34% 只是印证——它为什么低，第 13 页给的答案和我当时想的不一样。</p>
    </section>""",

    # 11 机制
    """<section>
      <h2>正则项没有失灵，它是被架空了</h2>
      <table>
        <tr><th>λ</th><th>pred_loss</th><th>sigreg_loss</th><th>emb 标准差</th><th>死维度</th><th>成功率</th></tr>
        <tr class="bad"><td>0.001</td><td>0.004</td><td>50.75</td><td>0.0012</td><td>69/192</td><td>34%</td></tr>
        <tr class="hi"><td>0.09 ★</td><td>0.266</td><td>5.16</td><td>0.325</td><td>0</td><td>52%</td></tr>
        <tr><td>1.0</td><td>1.074</td><td>11.77</td><td>0.115</td><td>0</td><td>38%</td></tr>
      </table>
      <p class="lead">SIGReg 一直在报警——它的值是 50.75，是健康状态的十倍。但权重只有 0.001，
      它对总 loss 的贡献只有 0.001 × 50.75 ≈ 0.05，而模型靠塌缩省下了 0.26 的预测误差。<strong>它出价出不过。</strong></p>
      <p class="aside">另一头是过度正则：不塌缩，但 embedding 被压得太紧（标准差 0.325 → 0.115），
      区分状态的能力变弱。论文选的 0.09 正好落在这个倒 U 的顶点。
      记住成功率的这个形状：<strong>34 / 52 / 38</strong>。</p>
    </section>""",

    # 12 散点
    f"""<section>
      <h2>六个 checkpoint，没有趋势线</h2>
      <img src="{img(FIG/'loss_vs_success.png')}" alt="预测 loss vs 规划成功率">
      <p class="aside">如果验证 loss 是可靠的代理指标，横跨三个数量级应该呈现一条向右下的趋势线。<strong>它没有。</strong>
      于是我手上只剩成功率这一个指标，而它的噪声是 ±17 个百分点。与其再多跑 seed 硬扛噪声，
      不如换一个问题：<strong>到底是什么决定规划成不成功？</strong></p>
    </section>""",

    # 13 实验四 · 换指标
    f"""<section>
      <h2>实验四 —— 换一批噪声小得多的指标</h2>
      <p class="lead">不训练任何新模型：在已有的 12 个 checkpoint 上、3000 帧留出数据上测四个量——
      Epps-Pulley 正态性（<strong>SIGReg 声称要优化的东西</strong>）、岭回归探针 embedding → 智能体位置的留出 R²
      （论文自己的 probing 口径）、有效秩、embedding 的绝对尺度。指标先在已知分布上校准过：
      真高斯给 0.5，4 维线性流形嵌入 192 维给 170.6。</p>
      <img src="{img(FIG/'exp4_scale_vs_success.png')}" style="max-height:290px" alt="embedding 尺度与规划成功率">
      <p class="aside"><strong>表征里有多少物理信息，几乎不决定它能不能规划；表征有多大才决定。</strong>
      尺度 r = +0.89（t = 5.85）；探针 R² +0.24、有效秩 −0.22、离高斯距离 −0.40，全部不显著。
      CEM 靠比较 embedding 空间里的距离选动作——信号缩小而 predictor 误差不变，代价地形就被噪声淹没。
      这也推翻了我在实验三给的解释：塌缩那个 checkpoint 探针 R² 是 0.467，健康的是 0.497，
      <strong>塌缩毁掉的不是信息，是信噪比。</strong></p>
    </section>""",

    # 14 实验四 · 混杂
    f"""<section>
      <h2>于是：前三组实验测的是同一个变量</h2>
      <img src="{img(FIG/'exp4_confound.png')}" alt="混杂变量与 SIGReg 的高斯性统计量">
      <p class="lead">实验一的成功率排序（52 / 54 / 64）与它们的尺度排序（0.319 / 0.348 / 0.389）完全一致；
      实验三成功率的倒 U（34 / 52 / 38）与尺度的倒 U（0.0012 / 0.319 / 0.116）同形。
      三个不同的旋钮，落到同一个中介变量上。
      <strong>加 seed 救不了这些实验——45 个 seed 只会得到一个混杂量的精确估计。问题不在功效，在设计。</strong></p>
      <p class="aside">顺带的收获（右图）：整个 1000 倍权重扫描里，SIGReg 的高斯性统计量是
      <strong>1206 / 1206 / 1206</strong>，纹丝不动，而真高斯是 0.5、4 维线性流形是 170。
      它实际维持的是 embedding 尺度，不是高斯化——论文命名的机制和实际生效的机制不是一回事。</p>
    </section>""",

    # 13 v3
    f"""<section>
      <h2>规模验证 —— DreamerV3 在 DMC walker-walk</h2>
      <p class="lead">手写版证明我理解每个组件；这一条是官方实现在真实基准上的完整曲线。</p>
      <img src="{img(FIG/'dreamerv3_walker.png')}" alt="DreamerV3 学习曲线">
      <div class="row">
        <div><span class="big">{f"{RET:.0f}" if RET else "--"}</span>
             <span class="unit">{f"{STEP/1000:.0f}k 步时的评估回报（满分约 1000）" if STEP else "未找到训练记录"}</span></div>
        <div><span class="unit">{v3_note}</span></div>
      </div>
    </section>""",

    # 14 工程坑
    """<section>
      <h2>真正花掉时间的地方</h2>
      <ul class="log">
        <li><strong>把 torch 软链进别人的 venv 复用。</strong>uv 在处理一个没锁版本的
        <code>torch&gt;=2</code> 时顺着软链<em>写穿</em>了，悄悄把另一个项目的 torch 从 2.7 升到 2.14，
        而那个项目当时还有个跑了一天的进程在跑。</li>
        <li><strong><code>embed_dim</code> 不是自由超参数。</strong>projector 的输入宽度取自它，
        而 ViT-tiny 的宽度固定在 192。我原计划的第一个消融直接崩在矩阵维度上——
        要正经扫它必须同步缩放骨干网络，那就不再是单变量消融了。</li>
        <li><strong>stdout 缓冲藏了两小时进度。</strong>终端日志卡在 45000 步一个多小时，
        而显式落盘的 <code>metrics.jsonl</code> 早就过了 80000 步。</li>
      </ul>
      <p class="aside">完整记录在仓库的 engineering log 里。写下来是因为——出问题的地方，恰恰是没人愿意写下来的地方。</p>
    </section>""",

    # 17 局限
    """<section>
      <h2>什么站得住，什么站不住</h2>
      <div class="two">
        <div class="pane accent">
          <p class="pane-label">站得住（都是直接测量）</p>
          <p>塌缩本身：embedding 标准差 0.0012 对 0.325，差 270 倍；死维度 69 对 0。没有噪声解释空间。</p>
          <p>尺度与规划成功率的相关：r = 0.89，t = 5.85，n = 11。</p>
          <p>SIGReg 的高斯性统计量在 1000 倍权重扫描下完全不动（1206 / 1206 / 1206）。</p>
        </div>
        <div class="pane">
          <p class="pane-label">站不住</p>
          <p>前三组的成功率结论<strong>全部作废</strong>——它们测的是同一个混杂变量（embedding 尺度）。</p>
          <p>实验四是<strong>观测，不是干预</strong>。要证明尺度导致规划成败，得在评估时人为缩放 embedding
          看成功率跟不跟着走。这是显然的下一步，我没做。</p>
          <p>全部结论都在 900 步的欠训练模型上。官方 checkpoint 尺度只有 0.032 却拿 86%——这条规律在训练充分后就不成立。</p>
          <p>没有 Dreamer 与 LeWorldModel 的正面对比：目标函数和评估指标都不共享。</p>
        </div>
      </div>
      <p class="lead">下一步的优先级被改写了两次：先是从"扫更多超参数"改成"把评估规模做够"，
      再从那里改成"先搞清楚主指标到底被什么决定"。<strong>回头看，第四组应该是第一组。</strong></p>
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
    "05_dreamerv3_scale/",
    "notes/engineering_log.md",
    "README.md",
]

CSS = f"""
*{{box-sizing:border-box;margin:0;padding:0}}
:root{{
  --ink:#17150f; --ink2:#4a463b; --dim:#8d8779;
  --bg:#faf9f5; --card:#fff; --line:#e0dcd0;
  --accent:#2a78d6; --good:#1a8a4a; --bad:#c8402e;
}}
body{{background:#d8d5cc;
  font-family:"Noto Sans CJK SC","Source Han Sans SC","PingFang SC","Microsoft YaHei",
              "IBM Plex Sans",sans-serif;
  color:var(--ink);-webkit-font-smoothing:antialiased}}
section{{
  width:{W}px;height:{H}px;padding:52px 68px;background:var(--bg);
  display:flex;flex-direction:column;justify-content:center;gap:18px;
  position:relative;overflow:hidden;margin:0 auto 18px;
  box-shadow:0 2px 16px rgba(0,0,0,.12);
}}
.eyebrow{{font-size:13px;letter-spacing:.16em;color:var(--accent);font-weight:500}}
h1{{font-size:52px;line-height:1.18;font-weight:700;letter-spacing:-.01em}}
h1 .thin{{font-weight:400;color:var(--ink2);font-size:38px}}
h2{{font-size:36px;font-weight:700;line-height:1.2;letter-spacing:-.01em}}
.byline{{font-size:18px;color:var(--ink2);line-height:1.7}}
.lead{{font-size:19px;line-height:1.7;color:var(--ink2);max-width:1000px}}
.aside{{font-size:16px;line-height:1.7;color:var(--dim);max-width:1120px}}
.aside strong,.lead strong{{color:var(--ink);font-weight:600}}
.dim{{color:var(--dim)}}
em{{font-style:normal;border-bottom:2px solid var(--accent);padding-bottom:1px}}
code{{font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:.9em;
  background:#efece2;padding:.1em .35em;border-radius:3px}}
img{{max-width:100%;max-height:330px;object-fit:contain;align-self:center}}

.title,.divider{{justify-content:center;gap:26px}}
.divider{{background:var(--ink)}}
.divider h1{{color:#fff;font-size:58px}}
.divider .eyebrow{{color:#8fb8e8}}
.divider .byline{{color:#b8b3a5}}

.two{{display:grid;grid-template-columns:1fr 1fr;gap:26px}}
.three{{display:grid;grid-template-columns:repeat(3,1fr);gap:22px}}
.pane,.card{{background:var(--card);border:1px solid var(--line);border-radius:10px;
  padding:24px 24px;display:flex;flex-direction:column;gap:12px}}
.pane p,.card p{{font-size:16.5px;line-height:1.65;color:var(--ink2)}}
.pane.accent,.card.accent{{border-color:var(--accent);box-shadow:0 0 0 1px var(--accent)}}
.pane-label{{font-size:12.5px;letter-spacing:.1em;color:var(--accent) !important;font-weight:600}}
.pane-claim{{font-size:21px !important;font-weight:700;color:var(--ink) !important;line-height:1.4}}
.card .n{{font-family:"IBM Plex Mono",monospace;font-size:30px;color:var(--dim) !important;line-height:1}}
.card-h{{font-size:19px !important;font-weight:700;color:var(--ink) !important}}

.row{{display:flex;gap:56px;align-items:baseline;flex-wrap:wrap}}
.row.wide{{gap:90px;margin:8px 0}}
.big{{font-family:"IBM Plex Mono",monospace;font-size:52px;font-weight:600;
  display:block;line-height:1}}
.unit{{font-size:15px;color:var(--dim);display:block;margin-top:8px;max-width:420px;line-height:1.5}}

table{{border-collapse:collapse;font-size:18px;width:100%;max-width:960px}}
th{{font-size:13px;letter-spacing:.05em;color:var(--dim);font-weight:500;
  text-align:right;padding:10px 18px;border-bottom:1px solid var(--line)}}
th:first-child,td:first-child{{text-align:left}}
td{{padding:12px 18px;border-bottom:1px solid var(--line);
  font-variant-numeric:tabular-nums;text-align:right;
  font-family:"IBM Plex Mono","Noto Sans CJK SC",monospace}}
tr.hi{{background:#e8f3ea}}tr.hi td{{font-weight:600}}
tr.bad td{{color:var(--bad)}}
.better{{color:var(--good);font-size:.72em;margin-left:.4em}}
.worse{{color:var(--bad);font-size:.72em;margin-left:.4em}}

ul.log{{list-style:none;display:flex;flex-direction:column;gap:16px;max-width:1120px}}
ul.log li{{font-size:17px;line-height:1.7;color:var(--ink2);
  padding-left:20px;border-left:3px solid var(--line)}}
ul.log strong{{color:var(--ink);font-weight:600}}

.foot{{position:absolute;bottom:26px;left:68px;right:68px;display:flex;
  justify-content:space-between;font-family:"IBM Plex Mono",monospace;
  font-size:12px;color:var(--dim)}}
.divider .foot{{color:#6f6a5e}}

@page{{size:{W}px {H}px;margin:0}}
@media print{{
  body{{background:#fff}}
  section{{margin:0;box-shadow:none;page-break-after:always;break-after:page}}
  section:last-child{{page-break-after:auto}}
}}
"""

JS = """
const S=[...document.querySelectorAll('section')];let i=0;
const go=n=>{i=Math.max(0,Math.min(S.length-1,n));S[i].scrollIntoView({behavior:'smooth'})};
addEventListener('keydown',e=>{
  if(['ArrowRight','ArrowDown',' ','PageDown'].includes(e.key)){e.preventDefault();go(i+1)}
  if(['ArrowLeft','ArrowUp','PageUp'].includes(e.key)){e.preventDefault();go(i-1)}
  if(e.key==='Home')go(0); if(e.key==='End')go(S.length-1);
});
S.forEach(s=>new IntersectionObserver(es=>es.forEach(e=>{
  if(e.isIntersecting)i=S.indexOf(e.target)}),{threshold:.5}).observe(s));
"""


def build():
    parts = []
    for n, (html, foot) in enumerate(zip(SLIDES, FOOTERS), start=1):
        footer = f'<div class="foot"><span>{foot}</span><span>{n} / {len(SLIDES)}</span></div>'
        parts.append(html.replace("</section>", footer + "</section>"))

    doc = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>世界模型的两条路线</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>{CSS}</style></head>
<body>
{"".join(parts)}
<script>{JS}</script>
</body></html>"""

    out = HERE / "slides_zh.html"
    out.write_text(doc, encoding="utf-8")
    print(f"{len(SLIDES)} 页 -> {out}  ({out.stat().st_size/1e6:.1f} MB)")
    return out


def to_pdf(html):
    pdf = HERE / "slides_zh.pdf"
    for exe in ("/opt/google/chrome/chrome", "google-chrome", "chromium"):
        try:
            subprocess.run(
                [exe, "--headless", "--disable-gpu", "--no-sandbox",
                 "--no-pdf-header-footer", f"--print-to-pdf={pdf}",
                 f"file://{html.resolve()}"],
                check=True, capture_output=True, timeout=180)
            print(f"PDF -> {pdf}  ({pdf.stat().st_size/1e6:.1f} MB)")
            return pdf
        except (FileNotFoundError, subprocess.CalledProcessError):
            continue
    print("找不到可用的 Chrome，PDF 未生成；用浏览器打开 html 后 Ctrl+P 也可以")
    return None


if __name__ == "__main__":
    html = build()
    if "--pdf" in sys.argv:
        to_pdf(html)
