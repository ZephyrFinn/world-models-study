"""生成原生 .pptx —— 文字是真文字，可以在 PowerPoint / WPS / Keynote 里直接改。

和 build_slides_zh.py 同样的内容和数据来源；版式用 python-pptx 重新搭，
不是把 HTML 截图贴进去。图表以 PNG 插入（它们本来就是图），其余全部是文本框和原生表格。

用法：python build_pptx.py   ->  世界模型的两条路线.pptx
"""
import json
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.util import Inches, Pt, Emu

HERE = Path(__file__).parent
ROOT = HERE.parent
FIG = ROOT / "04_analysis" / "figures"
D1 = ROOT / "01_dreamer_from_scratch" / "results"
METRICS = Path.home() / "workspace/WAM/dreamerv3-torch/logdir/dmc_walker_walk/metrics.jsonl"

# 16:9
W, H = Inches(13.333), Inches(7.5)
M = Inches(0.85)                      # 左右边距
CW = W - 2 * M                        # 内容宽度

INK   = RGBColor(0x17, 0x15, 0x0F)
INK2  = RGBColor(0x4A, 0x46, 0x3B)
DIM   = RGBColor(0x8D, 0x87, 0x79)
BG    = RGBColor(0xFA, 0xF9, 0xF5)
CARD  = RGBColor(0xFF, 0xFF, 0xFF)
LINE  = RGBColor(0xE0, 0xDC, 0xD0)
ACC   = RGBColor(0x2A, 0x78, 0xD6)
GOOD  = RGBColor(0x1A, 0x8A, 0x4A)
BAD   = RGBColor(0xC8, 0x40, 0x2E)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
GREY2 = RGBColor(0xB8, 0xB3, 0xA5)

CN = "微软雅黑"          # Windows/WPS 上通用；缺失时系统会回退
MONO = "Consolas"


def v3_latest():
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


PTS = v3_latest()
STEP, RET = (PTS[-1] if PTS else (None, None))
_tail = [r for _, r in PTS[-8:]]
_flat = len(_tail) >= 8 and (max(_tail) - min(_tail)) < 0.08 * max(_tail)
V3_NOTE = ("1M 步已跑完。" if STEP and STEP >= 1_000_000 else
           "约 30 万步起进入平台期，稳定在满分（约 1000）附近，训练继续跑向 1M。" if _flat else
           "曲线仍在上升，重新生成可刷新数值。")


# ---------------------------------------------------------------- 基础构件
def new_deck():
    prs = Presentation()
    prs.slide_width, prs.slide_height = W, H
    return prs


def blank(prs, bg=BG):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    fill = s.background.fill
    fill.solid()
    fill.fore_color.rgb = bg
    return s


def tb(slide, x, y, w, h, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP):
    box = slide.shapes.add_textbox(x, y, w, h)
    tf = box.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.paragraphs[0].alignment = align
    return tf


def para(tf, text, size, color=INK, bold=False, font=CN,
         space_after=6, line=1.45, first=False, align=None):
    p = tf.paragraphs[0] if first else tf.add_paragraph()
    p.text = text
    p.space_after = Pt(space_after)
    p.line_spacing = line
    if align is not None:
        p.alignment = align
    for r in p.runs:
        r.font.size = Pt(size)
        r.font.color.rgb = color
        r.font.bold = bold
        r.font.name = font
    return p


def heading(slide, text, y=Inches(0.62)):
    tf = tb(slide, M, y, CW, Inches(0.9))
    para(tf, text, 30, INK, bold=True, first=True, line=1.2)


def footer(slide, path, n, total, dark=False):
    c = GREY2 if dark else DIM
    tf = tb(slide, M, H - Inches(0.62), CW - Inches(1.2), Inches(0.32))
    para(tf, path, 9.5, c, font=MONO, first=True)
    tf2 = tb(slide, W - M - Inches(1.2), H - Inches(0.62), Inches(1.2), Inches(0.32),
             align=PP_ALIGN.RIGHT)
    para(tf2, f"{n} / {total}", 9.5, c, font=MONO, first=True)


def card(slide, x, y, w, h, accent=False, bg=CARD):
    from pptx.enum.shapes import MSO_SHAPE
    sh = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, x, y, w, h)
    sh.fill.solid()
    sh.fill.fore_color.rgb = bg
    sh.line.color.rgb = ACC if accent else LINE
    sh.line.width = Pt(1.5 if accent else 0.75)
    sh.shadow.inherit = False
    sh.adjustments[0] = 0.04
    sh.text_frame.word_wrap = True
    return sh


def picture(slide, path, top, max_h):
    """按最大高度等比缩放并水平居中。"""
    from PIL import Image
    iw, ih = Image.open(path).size
    h = max_h
    w = Emu(int(h * iw / ih))
    if w > CW:
        w = CW
        h = Emu(int(w * ih / iw))
    slide.shapes.add_picture(str(path), Emu(int((W - w) / 2)), top, width=w, height=h)
    return h


def table(slide, rows, x, y, w, col_w=None, hi=None, bad=None, fs=13):
    """rows[0] 是表头。hi / bad 是要高亮的数据行下标（从 0 计，不含表头）。"""
    nr, nc = len(rows), len(rows[0])
    shape = slide.shapes.add_table(nr, nc, x, y, w, Inches(0.42) * nr)
    t = shape.table
    if col_w:
        total = sum(col_w)
        for i, cwd in enumerate(col_w):
            t.columns[i].width = Emu(int(w * cwd / total))
    for ri, row in enumerate(rows):
        t.rows[ri].height = Inches(0.4)
        for ci, val in enumerate(row):
            cell = t.cell(ri, ci)
            cell.text = str(val)
            cell.margin_left, cell.margin_right = Inches(0.13), Inches(0.13)
            cell.margin_top, cell.margin_bottom = Inches(0.04), Inches(0.04)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            cell.fill.solid()
            if ri == 0:
                cell.fill.fore_color.rgb = BG
            elif hi is not None and ri - 1 == hi:
                cell.fill.fore_color.rgb = RGBColor(0xE8, 0xF3, 0xEA)
            else:
                cell.fill.fore_color.rgb = CARD
            p = cell.text_frame.paragraphs[0]
            p.alignment = PP_ALIGN.LEFT if ci == 0 else PP_ALIGN.RIGHT
            for r in p.runs:
                r.font.size = Pt(10 if ri == 0 else fs)
                r.font.name = CN if ci == 0 or ri == 0 else MONO
                r.font.bold = (hi is not None and ri - 1 == hi)
                if ri == 0:
                    r.font.color.rgb = DIM
                elif bad is not None and ri - 1 == bad:
                    r.font.color.rgb = BAD
                else:
                    r.font.color.rgb = INK
    return shape


def stat_row(slide, items, y):
    """items: [(大数字, 说明), ...]"""
    x = M
    for big, unit in items:
        tf = tb(slide, x, y, Inches(3.4), Inches(1.2))
        para(tf, big, 34, INK, bold=True, font=MONO, first=True, space_after=2, line=1.0)
        para(tf, unit, 11, DIM, space_after=0, line=1.3)
        x += Inches(3.6)


# ---------------------------------------------------------------- 演讲者备注
# 放映时只有演讲者屏幕能看到。完整逐页讲稿在 interview_prep/07_每页怎么讲.md
NOTES = {
 1: "10秒。不要念标题。\n"
    "「我做的是世界模型方向的复现加实验项目。两条路线都做了，但重点在后面自己设计的实验——"
    "尤其是最后一组，它推翻了我前面三组的结论。」",
 2: "90秒。全场地基，值得花时间。\n"
    "核心：世界模型=在脑子里建模拟器，先预演再行动，省真实交互。\n"
    "两条路线：Dreamer要重建像素（浪费在无关细节）；JEPA不重建（但有塌缩风险）。\n"
    "【必须埋伏笔】「如果目标只是预测下一个向量要准，有个作弊解——让encoder对任何输入都输出同一个常数，"
    "loss直接趋近0。模型赢了，任务输了。这叫表征塌缩。记住它，后面一半的内容都围绕它。」\n"
    "这里省30秒，第10页要多花2分钟解释。",
 3: "20秒。这页只是路标。三件事：手写Dreamer、跑通LeWM、四组受控实验。\n"
    "【提一句】「第四组是诊断前三组的，最后你会看到前三组的结论被它推翻了。」翻页。",
 4: "40秒。\n"
    "「没跑现成仓库，从零写的，670行单文件——没办法躲在别人的train.py后面。」\n"
    "【主动降低本页重要性】「这个结果本身不重要，是教科书预期结果。下一页那个才是我想给你看的。」",
 5: "50秒。Dreamer部分唯一值得细讲的一页。\n"
    "「策略从头到尾没见过真实状态，完全在想象里训练。那万一模型编了个奖励很高但不存在的世界呢？」\n"
    "左图KL稳在0.8，free bits=1.0在起作用，没塌缩也没爆炸。\n"
    "右图关键：想象内部算的λ-return从0涨到70+，和真实回报同步。\n"
    "「如果它自己涨得高但真实表现不动，就是自我欺骗。这图证明没发生。」\n"
    "可能追问 free bits：给KL设下限，防止随机变量被压成不携带信息，同VAE后验塌缩。",
 6: "40秒。\n"
    "86%是后面所有数字的参照基准。\n"
    "重点讲右边那个「4小时」：不是花在科研上，是装环境。四处断裂——缺swig、datasets五年前版本、"
    "transformers 5.x重命名ViT权重导致checkpoint加载不了、HDF5被静默import失败关掉。\n"
    "「复现类文章通常略过这段，但这恰恰是真正卡住人的地方。」",
 7: "15秒。必须停下来，让对方意识到叙事切换。\n"
    "「以上都是复现别人的。【停】接下来是我自己设计的。」\n"
    "【别省这句】「四组实验共用同一份数据、同一个encoder、同一个优化器、同样900步、同样50条episode——"
    "每组只动一个变量。【停】至少我当时是这么以为的。」\n"
    "最后半句是钩子，别漏。",
 8: "50秒。语气要中立，别把它讲成一个结论。\n"
    "扫history_size 1/3/5，3是论文默认。左图loss单调下降（符合直觉），右图成功率没跟上。\n"
    "h=1→h=3：loss改善，成功率反而掉2个点。\n"
    "【统计素养，主动说】「50条episode、p≈0.5，二项标准误约7个百分点，这一对就是噪声。」\n"
    "【重要：不要在这页下结论】「当时我留下的结论是h=5更好。第四组实验会告诉你这个结论也不成立——"
    "先记着这个成功率排序：52、54、64。」",
 9: "70秒。主动认错，讲法很关键——干脆，不辩解，而且要讲清「错在哪一层」。\n"
    "「原计划是固定预算只换架构，把PLDM和LeWM对比。」\n"
    "【停】「但这个设计本身是错的。JEPA是一个方法家族，家族成员共享架构，区别在anti-collapse loss。」\n"
    "「我只换了模型类，loss还是LeWM的SIGReg——等于两臂跑的是同一个方法。证据很硬：两份训练日志"
    "都在报sigreg_loss，真PLDM不该有这一项。」\n"
    "还有第二个bug：train.py没给Manager传seed，两次初始权重不同。\n"
    "【重做】「所以我重写了训练脚本，只换loss，补上seed，每臂跑3个seed。」\n"
    "结果：LeWM 55.3%，PLDM 43.3%，t=0.87，置信区间跨过0。方向对得上论文，但完全不显著。\n"
    "【真正的产出】「重做的价值不在这个对比，在那个标准差：同配置同数据，只换seed，成功率能从36%到70%——"
    "±17个百分点。要在80%功效下测出10个点的差异，每臂需要约45个seed。我有1个。」\n"
    "被问「怎么发现的」：老实说是被追问「官方不是说LeWM更好吗」之后才去diff的。比假装主动发现可信。",
 10: "60秒。语速放慢。\n"
     "「把SIGReg权重从0.09调到0.001，几乎关掉。结果预测loss掉到0.004——整晚最低，比正常好60倍。」\n"
     "【停】「但这个数字是假的。」【停】\n"
     "「我没有相信它。我写了个脚本，加载checkpoint、编码256帧真实数据、直接测每维标准差——"
     "192维里69维方差趋近于零。encoder塌缩了。」\n"
     "「模型找到作弊解：输出常数向量。预测常数当然不费力，所以loss反而好看。这个指标在这里是反向的。」\n"
     "三要素缺一不可：数字反常 → 我不信 → 我设计独立测量去验证。\n"
     "【同样别急着下结论】成功率34%只是印证，第四组会说明它为什么低——原因和我当时想的不一样。",
 11: "50秒。\n"
     "看sigreg_loss那一列：塌缩时50.75，是健康情况的十倍。\n"
     "「正则项一直在报警，它完全检测到了塌缩。但权重只有0.001，贡献只有0.05；"
     "而模型靠塌缩把预测误差从0.27降到0.004，净赚0.26。」\n"
     "【重】「它不是失灵，是被架空了——出价出不过。」\n"
     "【加分：现场心算】总loss 0.0548 = 预测0.004 + 0.001×50.75。严丝合缝。\n"
     "另一头10倍权重不塌缩但压太紧，成功率38%。倒U，论文的0.09在顶点。\n"
     "【记住这个倒U的形状：34、52、38。下一页之后要用。】",
 12: "35秒。这页是转折点，不是结论页。\n"
     "「六个checkpoint放一起，横轴loss对数轴跨三个数量级，纵轴成功率。如果loss是可靠代理，"
     "应该有一条向右下的趋势线。【停】没有。」\n"
     "【把它变成问题】「那我手上就只剩一个成功率指标，而它的噪声是±17个点。"
     "与其再多跑seed硬扛噪声，我想换个问题：到底什么决定规划成不成功？」\n"
     "这句话是第四组实验的动机，一定要说出来。",
 13: "80秒。全场最重要的一页。\n"
     "「第四组不训练任何新模型——在已有的12个checkpoint上，换一批低方差指标，3000帧留出数据。」\n"
     "四个指标：Epps-Pulley正态性（SIGReg自己声称要优化的东西）、岭回归探针embedding→智能体位置的"
     "留出R²（论文自己的probing口径）、有效秩、embedding绝对尺度。\n"
     "【先说可信度】「指标先在已知分布上校准过：真高斯给0.5，4维线性流形嵌进192维给170，有效秩还原到3.93。」\n"
     "【结果】「只有一个指标和成功率相关：embedding的绝对尺度，r=0.89，t=5.85。"
     "而物理探针R²是+0.24、有效秩−0.22、离高斯距离−0.30，全部不显著。」\n"
     "【一句话总结，慢】「表征里有多少物理信息，几乎不决定它能不能规划；表征有多大才决定。」\n"
     "机制：CEM靠比较embedding空间的距离选动作。信号缩小而predictor误差不变，代价地形就被噪声淹没。\n"
     "【回头改自己的结论】「这直接推翻我在实验三写的解释：塌缩那个checkpoint探针R²是0.467，"
     "健康的是0.497——位置信息几乎完好。塌缩毁掉的不是信息，是信噪比。」\n"
     "右下那个孤点是官方checkpoint：尺度0.032、有效秩1.66，却86%。所以「尺度决定」只在欠训练模型里成立。",
 14: "70秒。认错的第二层，也是整个项目的方法论落点。\n"
     "【左图】「实验一的成功率排序52、54、64，和它们的尺度排序0.319、0.348、0.389完全一致。」\n"
     "「实验三成功率的倒U——34、52、38，和尺度的倒U——0.0012、0.319、0.116，同一个形状。」\n"
     "【停，说重话】「三组实验拧的是三个不同的旋钮，但都落到同一个中介变量上。我以为在测三件事，"
     "其实一直在测同一件事。」\n"
     "【最有价值的一句】「关键是：加seed救不了这些实验。跑45个seed只会得到一个混杂量的精确估计。"
     "问题不在功效，在设计。」\n"
     "【右图，顺带的收获】「SIGReg的高斯性统计量在1000倍权重扫描里是1206、1206、1206，纹丝不动。"
     "参照值：真高斯0.5，4维线性流形170。」\n"
     "「所以它实际起的作用是维持embedding尺度，不是高斯化。论文命名的机制和实际生效的机制不是一回事。」\n"
     "被问「这是不是你实现错了」：可能，但同一个式子在已知分布上校准是对的，而且它确实随λ改变了尺度——"
     "该动的量动了，声称的量没动。这是我目前能给的最诚实的说法。",
 15: "25秒。时间紧可跳。\n"
     "「手写版证明我理解组件，规模上另外跑了官方DreamerV3在walker-walk，接近任务天花板。"
     "两条线分工：一条证明理解，一条拿真实曲线。」",
 16: "40秒。看岗位决定讲不讲——工程岗讲，研究岗一句带过。\n"
     "坑1：软链torch复用，包管理器顺着软链把另一个项目的torch就地升级了，"
     "而它还有跑了一天的进程在跑（靠Linux对已打开inode的保护才没崩）。\n"
     "坑2：原计划扫latent dimension直接崩——那参数不是自由的，和编码器宽度硬绑定。失败本身加深了理解。\n"
     "坑3：终端日志卡住一小时以为挂了，其实是stdout缓冲，指标文件早跑过去了。",
 17: "60秒。决定最后印象。干脆，不要显得沮丧——被推翻的是结论，不是能力。\n"
     "站得住的三样都是直接测量：塌缩本身（标准差差270倍、死维度69对0）、"
     "尺度与成功率的相关（r=0.89、t=5.85）、SIGReg的高斯统计量在1000倍扫描下不动。\n"
     "站不住的：前三组的成功率结论全部作废，因为它们测的是同一个混杂变量。\n"
     "【主动说出第四组自己的短板，这比等对方问出来强得多】「第四组也只是观测，不是干预。"
     "n=11，而且是相关。要证明尺度真的导致规划成败，得做干预实验——在评估时人为把embedding乘以一个系数，"
     "看成功率跟不跟着走。这是显然的下一步，我没做。」\n"
     "还有一条边界：官方checkpoint尺度0.032却拿86%，说明这条规律在训练充分之后就不成立了。\n"
     "【收尾】「所以下一步的优先级被改写了两次：第一次是从『扫更多超参数』改成『先把评估规模做够』，"
     "第二次是从那里改成『先搞清楚主指标被什么决定』。回头看，第四组应该是第一组。」",
}


# ---------------------------------------------------------------- 幻灯片
def build():
    prs = new_deck()
    T = 17
    n = 0

    def nxt(dark=False):
        nonlocal n
        n += 1
        return blank(prs, RGBColor(0x17, 0x15, 0x0F) if dark else BG)

    # 1 封面
    s = nxt()
    tf = tb(s, M, Inches(2.4), CW, Inches(3))
    para(tf, "基于模型的强化学习 · Model-based RL", 13, ACC, bold=True, first=True, space_after=14)
    para(tf, "世界模型的两条路线", 44, INK, bold=True, space_after=4, line=1.15)
    para(tf, "以及我把其中一条拆开验证之后", 27, INK2, space_after=18, line=1.2)
    para(tf, "Dreamer（生成式 RSSM）  ·  LeWorldModel（非生成式 JEPA）", 15, INK2)
    footer(s, "", 1, T)

    # 2 分歧
    s = nxt()
    heading(s, "分歧在哪")
    cw2 = (CW - Inches(0.35)) / 2
    for i, (label, claim, body, note, acc) in enumerate([
        ("Dreamer", "“能画出来，才算真懂。”",
         "世界模型带 decoder，训练目标包含把潜在状态还原成像素。然后让策略完全在这个模型的想象里训练。",
         "代价：为了预测球往哪飞，顺带得学会画背景里每一片草叶。", False),
        ("JEPA / LeWorldModel", "“预测像素本身就是错的目标。”",
         "完全没有 decoder。把观测编码成向量，只预测下一个向量。规划放到测试时做——CEM 搜索动作序列。",
         "代价：没有任何东西阻止 encoder 塌缩成一个常数。必须额外加约束。", True)]):
        x = M + i * (cw2 + Inches(0.35))
        card(s, x, Inches(1.65), cw2, Inches(3.5), accent=acc)
        tf = tb(s, x + Inches(0.3), Inches(1.9), cw2 - Inches(0.6), Inches(3))
        para(tf, label, 11, ACC, bold=True, first=True, space_after=8)
        para(tf, claim, 17, INK, bold=True, space_after=10, line=1.3)
        para(tf, body, 13, INK2, space_after=8)
        para(tf, note, 12, DIM)
    tf = tb(s, M, Inches(5.45), CW, Inches(0.8))
    para(tf, "两者不共享优化目标（奖励 vs 与目标的距离），也不共享评估指标（累计回报 vs 成功率）。这一点后面还会回来。",
         12, DIM, first=True)
    footer(s, "", 2, T)

    # 3 总览
    s = nxt()
    heading(s, "我做了什么")
    cw3 = (CW - Inches(0.6)) / 3
    for i, (num, h, body, acc) in enumerate([
        ("1", "手写 Dreamer", "从零实现，单文件。RSSM、reward model、actor-critic、imagination rollout。", False),
        ("2", "复现 LeWorldModel", "跑通官方实现，复现出 86% 的规划成功率。", False),
        ("3", "然后去验证它", "四组受控实验。第四组推翻了前三组的结论——这才是值得讲的部分。", True)]):
        x = M + i * (cw3 + Inches(0.3))
        card(s, x, Inches(2.0), cw3, Inches(2.7), accent=acc)
        tf = tb(s, x + Inches(0.28), Inches(2.25), cw3 - Inches(0.56), Inches(2.2))
        para(tf, num, 22, DIM, bold=True, font=MONO, first=True, space_after=8, line=1.0)
        para(tf, h, 16, INK, bold=True, space_after=8)
        para(tf, body, 13, INK2)
    footer(s, "", 3, T)

    # 4 手写 Dreamer
    s = nxt()
    heading(s, "Dreamer，一行行写出来")
    tf = tb(s, M, Inches(1.5), CW, Inches(0.7))
    para(tf, "约 670 行，不依赖任何框架。GRU 确定性状态 + 32×16 categorical 随机状态，"
             "straight-through 梯度，actor-critic 完全在想象轨迹上用 λ-return 训练。",
         13, INK2, first=True)
    picture(s, D1 / "comparison_multiseed.png", Inches(2.25), Inches(3.15))
    stat_row(s, [("240.1", "Dreamer 平均回报"), ("113.8", "Double-DQN"),
                 ("", "CartPole-v1 · 2 万步 · 3 个 seed")], Inches(5.65))
    footer(s, "01_dreamer_from_scratch/", 4, T)

    # 5 诊断
    s = nxt()
    heading(s, "想象出来的东西，对应现实吗？")
    tf = tb(s, M, Inches(1.5), CW, Inches(0.8))
    para(tf, "actor 从头到尾没见过一个真实状态。所以关键问题是：想象轨迹跟不跟现实对齐，"
             "还是说世界模型编了一个奖励很高但不存在的世界，策略在里面自我陶醉。",
         13, INK2, first=True)
    picture(s, D1 / "diagnostics.png", Inches(2.35), Inches(3.0))
    tf = tb(s, M, Inches(5.6), CW, Inches(0.9))
    para(tf, "KL 稳定在 0.8 附近，free bits 正在起作用（既没后验塌缩也没爆炸）。"
             "想象中算出的 λ-return 从 0 涨到 70+，与真实评估回报同步——这就是“没有自欺”的证据。",
         12, DIM, first=True)
    footer(s, "01_dreamer_from_scratch/results/diagnostics.png", 5, T)

    # 6 LeWM 复现
    s = nxt()
    heading(s, "LeWorldModel，复现")
    stat_row(s, [("86%", "官方 checkpoint 的规划成功率"),
                 ("约 4 小时", "花在装环境上，不是花在科研上")], Inches(1.75))
    tf = tb(s, M, Inches(3.25), CW, Inches(0.6))
    para(tf, "tworoom 任务，CEM-MPC，50 条留出 episode。这是后面所有数字的参照基准。",
         14, INK2, first=True)
    tf = tb(s, M, Inches(4.0), CW, Inches(2.0))
    para(tf, "官方 README 和已发布的包之间有四处断裂：", 12.5, DIM, first=True, space_after=4)
    for line in ["box2d-py 缺 swig 编译不过",
                 "datasets 解析到五年前的版本，与现代 pyarrow 不兼容",
                 "transformers 5.x 重命名了所有 ViT 权重，导致 checkpoint 加载失败",
                 "HDF5 支持在包里，但被一个静默的 import 失败关掉了"]:
        para(tf, "·  " + line, 12.5, DIM, space_after=3)
    para(tf, "这些我全写进了仓库——复现类文章通常略过这段，而这段恰恰是真正卡住人的地方。",
         12.5, DIM, space_after=0)
    footer(s, "02_lewm_reproduction/setup.md", 6, T)

    # 7 分界
    s = nxt(dark=True)
    tf = tb(s, M, Inches(2.6), CW, Inches(2.6))
    para(tf, "以上是复现", 14, RGBColor(0x8F, 0xB8, 0xE8), bold=True, first=True, space_after=16)
    para(tf, "以下是我自己做的", 42, WHITE, bold=True, space_after=18, line=1.2)
    para(tf, "同一份数据、同一个 encoder、同一个优化器、同样 900 步梯度更新、同样 50 条留出 episode。",
         14, GREY2, space_after=2)
    para(tf, "每组实验只动一个变量——至少我当时是这么以为的。", 14, GREY2)
    footer(s, "03_experiments/", 7, T, dark=True)

    # 8 实验一
    s = nxt()
    heading(s, "实验一 —— 需要看多少历史？")
    tf = tb(s, M, Inches(1.5), CW, Inches(0.5))
    para(tf, "history_size ∈ {1, 3, 5}，即 predictor 能看到几帧过去的 embedding。3 是论文默认值。",
         13, INK2, first=True)
    picture(s, FIG / "exp1_horizon.png", Inches(2.1), Inches(2.9))
    tf = tb(s, M, Inches(5.2), CW, Inches(1.2))
    para(tf, "loss 随上下文单调下降，成功率却没跟上。", 13, INK, bold=True, first=True, space_after=4)
    para(tf, "h=1 → h=3 这一步，loss 改善了，成功率反而掉了 2 个点。50 条 episode 的标准误约 7 个百分点，"
             "所以这一对就是噪声，当时我留下的结论只有“h=5 更好”。"
             "记住这个成功率排序：52 / 54 / 64 —— 第 14 页会回来推翻它。",
         12, DIM)
    footer(s, "03_experiments/exp1_horizon/", 8, T)

    # 9 实验二
    s = nxt()
    heading(s, "实验二 —— 变量选错了，重做一次")
    tf = tb(s, M, Inches(1.45), CW, Inches(1.0))
    para(tf, "原计划：固定预算只换架构，把 PLDM 和 LeWM 放进同一个训练循环和规划器。"
             "跑出来 PLDM 预测更差却规划更好，看着像个漂亮的反例。", 13, INK2, first=True, space_after=6)
    para(tf, "但这个设计本身就是错的：JEPA 是一个方法家族，成员共享架构，区别在 anti-collapse loss。"
             "我只换了模型类、loss 仍然是 LeWM 的 SIGReg——两臂跑的是同一个方法。"
             "证据：两份训练日志都在报 sigreg_loss，真正的 PLDM 不该有这一项。", 12.5, BAD)
    tf = tb(s, M, Inches(2.78), CW, Inches(0.55))
    para(tf, "重写训练脚本：架构、数据、预算、规划器全部固定，只换防塌缩机制（SIGReg×1 对 VCReg×4 + "
             "时序对齐 + 逆动力学 = 6 项），补上 Manager 的 seed，每臂 3 个 seed。", 12.5, INK2, first=True)
    picture(s, FIG / "exp2_seed_variance.png", Inches(3.42), Inches(2.1))
    tf = tb(s, M, Inches(5.75), CW, Inches(1.1))
    para(tf, "LeWM 55.3% 对 PLDM 43.3%，t = 0.87，95% CI [-27, +51]。方向和论文一致，但完全不显著。",
         13, INK, bold=True, first=True, space_after=4)
    para(tf, "真正的产出是那个标准差：同配置、同数据，只换 seed，成功率能从 36% 走到 70%——±17 个百分点。"
             "要在 80% 功效下测出 10 个点的差异，每臂约需 45 个 seed。我有 1 个。",
         12, DIM)
    footer(s, "03_experiments/exp2_pldm/", 9, T)

    # 10 实验三
    s = nxt()
    heading(s, "实验三 —— 于是我不再信 loss")
    tf = tb(s, M, Inches(1.5), CW, Inches(0.7))
    para(tf, "LeWorldModel 的核心主张：两项 loss 就够，因为 SIGReg 会防止表征塌缩。"
             "我把它的权重扫到 0.001 和 1.0，并且没有去读 loss，而是直接测量 embedding 本身。",
         13, INK2, first=True)
    picture(s, FIG / "exp3_sigreg.png", Inches(2.3), Inches(2.85))
    tf = tb(s, M, Inches(5.35), CW, Inches(1.2))
    para(tf, "权重 0.001 时：pred_loss = 0.004，是我整晚训出来的最低值，比正常模型好 60 倍——"
             "同时 192 维里有 69 维已经死掉。", 12.5, INK, bold=True, first=True, space_after=4)
    para(tf, "encoder 塌缩会让预测任务变得极其简单，这个指标在这里是反向的。"
             "成功率 34% 只是印证——它为什么低，第 13 页给的答案和我当时想的不一样。", 12, DIM)
    footer(s, "03_experiments/exp3_sigreg/", 10, T)

    # 11 机制
    s = nxt()
    heading(s, "正则项没有失灵，它是被架空了")
    table(s, [["λ", "pred_loss", "sigreg_loss", "emb 标准差", "死维度", "成功率"],
              ["0.001", "0.004", "50.75", "0.0012", "69/192", "34%"],
              ["0.09  ★", "0.266", "5.16", "0.325", "0", "52%"],
              ["1.0", "1.074", "11.77", "0.115", "0", "38%"]],
          M, Inches(1.55), Inches(10.2), col_w=[2, 2, 2.2, 2.2, 1.8, 1.8], hi=1, bad=0)
    tf = tb(s, M, Inches(3.65), CW, Inches(2.2))
    para(tf, "SIGReg 一直在报警——它的值是 50.75，是健康状态的十倍。", 14, INK, bold=True,
         first=True, space_after=6)
    para(tf, "但权重只有 0.001，它对总 loss 的贡献只有 0.001 × 50.75 ≈ 0.05，"
             "而模型靠塌缩省下了 0.26 的预测误差。它出价出不过。", 13.5, INK2, space_after=10)
    para(tf, "另一头是过度正则：不塌缩，但 embedding 被压得太紧（标准差 0.325 → 0.115），"
             "区分状态的能力变弱。论文选的 0.09 正好落在这个倒 U 的顶点。"
             "记住成功率的这个形状：34 / 52 / 38。", 12, DIM)
    footer(s, "03_experiments/exp3_sigreg/probe_collapse.py", 11, T)

    # 12 散点
    s = nxt()
    heading(s, "六个 checkpoint，没有趋势线")
    picture(s, FIG / "loss_vs_success.png", Inches(1.6), Inches(3.7))
    tf = tb(s, M, Inches(5.6), CW, Inches(1.0))
    para(tf, "如果验证 loss 是可靠的代理指标，横跨三个数量级应该呈现一条向右下的趋势线。它没有。",
         13, INK, bold=True, first=True, space_after=4)
    para(tf, "于是我手上只剩成功率这一个指标，而它的噪声是 ±17 个百分点。与其再多跑 seed 硬扛噪声，"
             "不如换一个问题：到底是什么决定规划成不成功？", 12, DIM)
    footer(s, "04_analysis/summary.csv", 12, T)

    # 13 实验四 · 换指标
    s = nxt()
    heading(s, "实验四 —— 换一批噪声小得多的指标")
    tf = tb(s, M, Inches(1.42), CW, Inches(0.85))
    para(tf, "不训练任何新模型：在已有的 12 个 checkpoint 上、3000 帧留出数据上测四个量——"
             "Epps-Pulley 正态性（SIGReg 声称要优化的东西）、岭回归探针 embedding → 智能体位置的留出 R²"
             "（论文自己的 probing 口径）、有效秩、embedding 的绝对尺度。",
         12.5, INK2, first=True, space_after=4)
    para(tf, "指标先在已知分布上校准：真·各向同性高斯给 0.5，4 维线性流形嵌入 192 维给 170.6，"
             "有效秩还原到 3.93。", 12, DIM)
    picture(s, FIG / "exp4_scale_vs_success.png", Inches(2.7), Inches(2.7))
    tf = tb(s, M, Inches(5.55), CW, Inches(1.35))
    para(tf, "表征里有多少物理信息，几乎不决定它能不能规划；表征有多大才决定。",
         14, INK, bold=True, first=True, space_after=5)
    para(tf, "尺度 r = +0.89（t = 5.85）；探针 R² +0.24、有效秩 −0.22、离高斯距离 −0.30，全部不显著。"
             "CEM 靠比较 embedding 空间里的距离选动作——信号缩小而 predictor 误差不变，代价地形就被噪声淹没。"
             "这也推翻了我在实验三给的解释：塌缩那个 checkpoint 探针 R² 是 0.467，健康的是 0.497，"
             "位置信息几乎完好。塌缩毁掉的不是信息，是信噪比。", 11.5, DIM)
    footer(s, "03_experiments/exp4_representation/", 13, T)

    # 14 实验四 · 混杂
    s = nxt()
    heading(s, "于是：前三组实验测的是同一个变量")
    picture(s, FIG / "exp4_confound.png", Inches(1.45), Inches(2.9))
    tf = tb(s, M, Inches(4.6), CW, Inches(2.4))
    para(tf, "实验一的成功率排序（52 / 54 / 64）与它们的尺度排序（0.319 / 0.348 / 0.389）完全一致；"
             "实验三成功率的倒 U（34 / 52 / 38）与尺度的倒 U（0.0012 / 0.319 / 0.116）同形。"
             "三个不同的旋钮，落到同一个中介变量上。",
         13, INK2, first=True, space_after=7)
    para(tf, "加 seed 救不了这些实验——45 个 seed 只会得到一个混杂量的精确估计。问题不在功效，在设计。",
         13.5, BAD, bold=True, space_after=9)
    para(tf, "顺带的收获（右图）：整个 1000 倍权重扫描里，SIGReg 的高斯性统计量是 1206 / 1206 / 1206，"
             "纹丝不动，而真高斯是 0.5、4 维线性流形是 170。它实际维持的是 embedding 尺度，不是高斯化——"
             "论文命名的机制和实际生效的机制不是一回事。", 12, DIM)
    footer(s, "03_experiments/exp4_representation/", 14, T)

    # 15 v3
    s = nxt()
    heading(s, "规模验证 —— DreamerV3 在 DMC walker-walk")
    tf = tb(s, M, Inches(1.45), CW, Inches(0.5))
    para(tf, "手写版证明我理解每个组件；这一条是官方实现在真实基准上的完整曲线。",
         13, INK2, first=True)
    picture(s, FIG / "dreamerv3_walker.png", Inches(2.05), Inches(3.1))
    stat_row(s, [(f"{RET:.0f}" if RET else "--",
                  f"{STEP/1000:.0f}k 步时的评估回报（满分约 1000）" if STEP else "未找到训练记录")],
             Inches(5.4))
    tf = tb(s, M + Inches(3.6), Inches(5.5), Inches(8.0), Inches(0.9))
    para(tf, V3_NOTE, 12, DIM, first=True)
    footer(s, "05_dreamerv3_scale/", 15, T)

    # 16 工程坑
    s = nxt()
    heading(s, "真正花掉时间的地方")
    tf = tb(s, M, Inches(1.7), CW, Inches(3.8))
    for i, (h, body) in enumerate([
        ("把 torch 软链进别人的 venv 复用。",
         "uv 在处理一个没锁版本的 torch>=2 时顺着软链写穿了，悄悄把另一个项目的 torch 从 2.7 升到 2.14，"
         "而那个项目当时还有个跑了一天的进程在跑。"),
        ("embed_dim 不是自由超参数。",
         "projector 的输入宽度取自它，而 ViT-tiny 的宽度固定在 192。我原计划的第一个消融直接崩在矩阵维度上——"
         "要正经扫它必须同步缩放骨干网络，那就不再是单变量消融了。"),
        ("stdout 缓冲藏了两小时进度。",
         "终端日志卡在 45000 步一个多小时，而显式落盘的 metrics.jsonl 早就过了 80000 步。")]):
        para(tf, h, 14, INK, bold=True, first=(i == 0), space_after=3)
        para(tf, body, 12.5, INK2, space_after=14)
    tf = tb(s, M, Inches(5.9), CW, Inches(0.6))
    para(tf, "完整记录在仓库的 engineering log 里。写下来是因为——出问题的地方，恰恰是没人愿意写下来的地方。",
         12, DIM, first=True)
    footer(s, "notes/engineering_log.md", 16, T)

    # 17 局限
    s = nxt()
    heading(s, "什么站得住，什么站不住")
    cw2 = (CW - Inches(0.35)) / 2
    for i, (label, items, acc) in enumerate([
        ("站得住（都是直接测量）",
         ["塌缩本身：embedding 标准差 0.0012 对 0.325，差 270 倍；死维度 69 对 0。没有噪声解释空间。",
          "尺度与规划成功率的相关：r = 0.89，t = 5.85，n = 11。",
          "SIGReg 的高斯性统计量在 1000 倍权重扫描下完全不动（1206 / 1206 / 1206）。"], True),
        ("站不住",
         ["前三组的成功率结论全部作废——它们测的是同一个混杂变量（embedding 尺度）。",
          "实验四是观测，不是干预。要证明尺度导致规划成败，得在评估时人为缩放 embedding 看成功率跟不跟着走。这是显然的下一步，我没做。",
          "全部结论都在 900 步的欠训练模型上。官方 checkpoint 尺度只有 0.032 却拿 86%——这条规律在训练充分后就不成立。",
          "没有 Dreamer 与 LeWorldModel 的正面对比：目标函数和评估指标都不共享。"], False)]):
        x = M + i * (cw2 + Inches(0.35))
        card(s, x, Inches(1.55), cw2, Inches(3.5), accent=acc)
        tf = tb(s, x + Inches(0.3), Inches(1.8), cw2 - Inches(0.6), Inches(3.0))
        para(tf, label, 12, ACC if acc else DIM, bold=True, first=True, space_after=10)
        for it in items:
            para(tf, it, 12, INK2, space_after=8)
    tf = tb(s, M, Inches(5.45), CW, Inches(1.4))
    para(tf, "下一步的优先级被改写了两次：先是从“扫更多超参数”改成“把评估规模做够”，"
             "再从那里改成“先搞清楚主指标到底被什么决定”。回头看，第四组应该是第一组。",
         13, INK2, first=True)
    footer(s, "README.md", 17, T)

    # 写入演讲者备注
    for i, sl in enumerate(prs.slides, start=1):
        if i in NOTES:
            sl.notes_slide.notes_text_frame.text = NOTES[i]

    out = HERE / "世界模型的两条路线.pptx"
    prs.save(out)
    print(f"{len(prs.slides.__iter__.__self__._sldIdLst)} 页 -> {out}  "
          f"({out.stat().st_size/1e6:.1f} MB)")
    return out


if __name__ == "__main__":
    build()
