"""第 1 章 EDA：认识 BTCUSDT 永续合约资金费率数据。

运行：python eda/01_eda_btcusdt.py
输出：终端打印统计量，图片保存到 eda/figs/
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.path.join(ROOT, "data", "funding_rate", "BTCUSDT_fundingRate.csv")
FIG = os.path.join(ROOT, "eda", "figs")
os.makedirs(FIG, exist_ok=True)

# ---- 画图样式 ----
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({
    "font.sans-serif": ["Microsoft YaHei", "SimHei"],
    "axes.unicode_minus": False,
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 110, "savefig.bbox": "tight",
})


def save(fig, name):
    fig.savefig(os.path.join(FIG, name))
    plt.close(fig)


def section(title):
    print("\n" + "=" * 8, title, "=" * 8)


# ============ 1.1 / 1.2 数据类型与矩形数据 ============
df = pd.read_csv(CSV)
section("1.1/1.2 原始数据")
print(df.head(), "\n")
print(df.dtypes)

# calc_time 是毫秒时间戳，末尾偶尔带几毫秒抖动，取整到分钟
df["time"] = pd.to_datetime(df["calc_time"], unit="ms").dt.round("min")
# 原始值是小数（0.0001），换成百分比（0.01%）更直观
df["rate"] = df["last_funding_rate"] * 100
df["year"] = df["time"].dt.year

print("\n记录数:", len(df))
print("时间范围:", df["time"].min(), "->", df["time"].max())
print("结算间隔取值:", df["funding_interval_hours"].value_counts().to_dict())
print("相邻两条的时间差:", df["time"].diff().value_counts().to_dict())
print("重复时间点:", df["time"].duplicated().sum())

# ============ 1.3 位置估计 ============
r = df["rate"]
section("1.3 位置估计（单位：% / 8小时）")
loc = pd.Series({
    "均值 mean": r.mean(),
    "中位数 median": r.median(),
    "截尾均值 trim 10%": stats.trim_mean(r, 0.1),
    "众数 mode": r.round(6).mode()[0],
})
print(loc.round(5).to_string())
print("众数 0.01% 出现的比例: {:.1%}".format((r.round(6) == 0.01).mean()))
print("年化均值（×3×365）: {:.2f}%".format(r.mean() * 3 * 365))

# ============ 1.4 变异性估计 ============
section("1.4 变异性估计")
q = r.quantile([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
var = pd.Series({
    "标准差 std": r.std(),
    "平均绝对偏差 mean abs dev": (r - r.mean()).abs().mean(),
    "MAD（未缩放）": stats.median_abs_deviation(r),
    "IQR": q[0.75] - q[0.25],
    "极差 range": r.max() - r.min(),
})
print(var.round(5).to_string())
print("\n百分位数:\n" + q.round(5).to_string())
print("\n最大值 {:.4f}% @ {}".format(r.max(), df.loc[r.idxmax(), "time"]))
print("最小值 {:.4f}% @ {}".format(r.min(), df.loc[r.idxmin(), "time"]))
print("触及 ±0.3% 上下限的次数:", (r.abs() >= 0.3 - 1e-9).sum())


def spread(x):
    x = np.asarray(x)
    q25, q75 = np.percentile(x, [25, 75])
    return pd.Series({
        "标准差": x.std(ddof=1),
        "平均绝对偏差": np.abs(x - x.mean()).mean(),
        "MAD": stats.median_abs_deviation(x),
        "IQR": q75 - q25,
        "极差": x.max() - x.min(),
    })


# 敏感度实验 A：去掉两头最极端的 p%，看每个指标变了多少
section("1.4 敏感度实验 A：去掉极端值")
rows = {"全部数据": spread(r)}
for p in [0.1, 1, 5]:
    lo, hi = np.percentile(r, [p / 2, 100 - p / 2])
    rows["去掉两头共 {}%".format(p)] = spread(r[(r >= lo) & (r <= hi)])
rows["额外加 1 个 1.0%"] = spread(np.append(r.values, 1.0))
tbl = pd.DataFrame(rows).T
print(tbl.round(5).to_string())
print("\n相对全部数据的变化:")
print((tbl / tbl.iloc[0] - 1).applymap("{:+.0%}".format).to_string())

# 敏感度实验 B：最极端的少数观测贡献了多少“分散程度”
section("1.4 敏感度实验 B：极端观测的贡献")
dev = r - r.mean()
sq = (dev ** 2).sort_values(ascending=False).values
ab = dev.abs().sort_values(ascending=False).values
for n in [2, 10, 73, 365]:
    print("最大的 {:>3} 条（{:.1%}）占 偏差平方和 {:.1%}，占 偏差绝对值和 {:.1%}".format(
        n, n / len(r), sq[:n].sum() / sq.sum(), ab[:n].sum() / ab.sum()))

# 图6：累计贡献曲线
share_x = np.arange(1, len(r) + 1) / len(r) * 100
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(share_x, np.cumsum(sq) / sq.sum() * 100, color=ORANGE, lw=2)
ax.plot(share_x, np.cumsum(ab) / ab.sum() * 100, color=BLUE, lw=2)
ax.plot([0, 100], [0, 100], color=INK2, lw=0.8, ls="--")
ax.text(13, 82, "偏差平方（方差 / 标准差用）", color=INK, fontsize=9)
ax.text(34, 70, "偏差绝对值（平均绝对偏差用）", color=INK, fontsize=9)
ax.text(66, 59, "每条贡献一样多", color=INK2, fontsize=9, rotation=25)
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.set_xlabel("按偏差从大到小取前 x% 的观测")
ax.set_ylabel("占总和的累计比例（%）")
ax.set_title("少数极端观测贡献了大部分方差", loc="left")
save(fig, "06_variance_contribution.png")

# ============ 1.5 探索分布 ============
section("1.5 分布")
print("偏度 skew: {:.2f}   峰度 kurtosis(超额): {:.2f}".format(r.skew(), r.kurt()))

# 图1：时间序列
fig, ax = plt.subplots(figsize=(11, 4))
ax.plot(df["time"], r, color=BLUE, lw=0.6, alpha=0.45, label="每期费率")
ax.plot(df["time"], r.rolling(90).mean(), color=BLUE, lw=2, label="30 天滚动均值（90 期）")
ax.axhline(0.01, color=INK2, lw=1, ls="--")
ax.text(df["time"].iloc[-1], 0.012, "0.01% 基准", color=INK2, ha="right", va="bottom", fontsize=9)
ax.axhline(0, color=INK2, lw=0.8)
ax.set_ylim(-0.1, 0.2)
ax.set_ylabel("资金费率（% / 8 小时）")
ax.set_title("BTCUSDT 资金费率 2020–2026（纵轴截在 −0.1%~0.2%，极端值见图 2）", loc="left")
ax.legend(loc="upper right", frameon=False)
save(fig, "01_timeseries.png")

# 图2：直方图，全范围 + 放大
fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
axes[0].hist(r.values, bins=120, color=BLUE, edgecolor="#fcfcfb", linewidth=0.3)
axes[0].set_title("全范围（−0.3% ~ 0.3%）", loc="left")
axes[0].set_yscale("log")
axes[0].set_ylabel("次数（对数刻度）")
zoom = r[(r > -0.03) & (r < 0.06)]
axes[1].hist(zoom.values, bins=np.arange(-0.03, 0.0605, 0.001), color=BLUE,
             edgecolor="#fcfcfb", linewidth=0.3)
axes[1].annotate("尖峰：恰好 0.01% 的有 {:.0%}".format((r.round(6) == 0.01).mean()),
                 xy=(0.0105, 2500), xytext=(0.02, 2000), color=INK2, fontsize=9,
                 arrowprops=dict(arrowstyle="->", color=INK2, lw=0.8))
axes[1].axvline(r.mean(), color=ORANGE, lw=1.5, label="均值 {:.4f}%".format(r.mean()))
axes[1].axvline(r.median(), color=INK, lw=1.5, ls="--", label="中位数 {:.4f}%".format(r.median()))
axes[1].set_title("放大到 −0.03% ~ 0.06%（占 {:.1%}）".format(len(zoom) / len(r)), loc="left")
axes[1].legend(frameon=False)
for a in axes:
    a.set_xlabel("资金费率（%）")
save(fig, "02_histogram.png")

# 图3：按年份的箱线图（1.8 数值 × 类别）
years = sorted(df["year"].unique())
fig, ax = plt.subplots(figsize=(11, 4))
ax.boxplot([df.loc[df["year"] == y, "rate"] for y in years], labels=years, widths=0.5,
           medianprops=dict(color=ORANGE, linewidth=2), boxprops=dict(color=BLUE),
           whiskerprops=dict(color=BLUE), capprops=dict(color=BLUE),
           flierprops=dict(marker="o", markersize=2, markerfacecolor=INK2,
                           markeredgecolor="none", alpha=0.4))
ax.axhline(0.01, color=INK2, lw=1, ls="--")
ax.set_ylim(-0.06, 0.12)
ax.set_ylabel("资金费率（%）")
ax.set_title("按年份分组的箱线图（纵轴截在 −0.06%~0.12%，橙线 = 中位数）", loc="left")
save(fig, "03_boxplot_by_year.png")

# ============ 1.6 类别数据 ============
section("1.6 把费率分成类别")
df["状态"] = np.select(
    [r < 0, r.round(6) == 0.01, r > 0.01],
    ["负费率", "=0.01%", ">0.01%"], default="0~0.01%")
order = ["负费率", "0~0.01%", "=0.01%", ">0.01%"]
share = df["状态"].value_counts(normalize=True).reindex(order)
print(share.map("{:.1%}".format).to_string())

# ============ 1.8 两个变量：年份 × 状态 列联表 ============
section("1.8 列联表：年份 × 状态（行百分比）")
ct = pd.crosstab(df["year"], df["状态"], normalize="index")[order]
print((ct * 100).round(1).to_string())

section("按年份的位置与变异性")
by_year = df.groupby("year")["rate"].agg(
    均值="mean", 中位数="median", 标准差="std",
    IQR=lambda s: s.quantile(0.75) - s.quantile(0.25), 条数="count")
by_year["年化均值%"] = by_year["均值"] * 3 * 365
print(by_year.round(4).to_string())

# 图4：各年状态占比（堆叠条形图）
colors = [ORANGE, "#b8b6ae", BLUE, AQUA]
fig, ax = plt.subplots(figsize=(11, 4))
bottom = np.zeros(len(ct))
for c, col in zip(order, colors):
    ax.bar(ct.index.astype(str), ct[c] * 100, bottom=bottom, color=col,
           edgecolor="#fcfcfb", linewidth=2, width=0.6, label=c)
    for i, v in enumerate(ct[c] * 100):
        if v >= 6:
            ax.text(i, bottom[i] + v / 2, "{:.0f}%".format(v), ha="center", va="center",
                    fontsize=8, color="white" if col != "#b8b6ae" else INK)
    bottom += ct[c].values * 100
ax.set_ylabel("占比（%）")
ax.set_ylim(0, 100)
ax.grid(axis="x", visible=False)
ax.set_title("每年各类费率的占比", loc="left")
ax.legend(ncol=4, frameon=False, loc="upper left", bbox_to_anchor=(0, -0.08))
save(fig, "04_category_by_year.png")

# ============ 1.7 相关性：费率和它自己的过去（自相关） ============
section("1.7 自相关（今天的费率 vs N 期前的费率）")
lags = [1, 3, 9, 21, 90]
acf = {k: r.autocorr(k) for k in lags}
print("  ".join("lag{}={:.3f}".format(k, v) for k, v in acf.items()))
print("（lag3 = 1 天前，lag21 = 1 周前，lag90 = 30 天前）")

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
prev = r.shift(1)
axes[0].scatter(prev, r, s=8, color=BLUE, alpha=0.25, edgecolors="none")
axes[0].plot([-0.3, 0.3], [-0.3, 0.3], color=INK2, lw=0.8, ls="--")
axes[0].set_xlim(-0.1, 0.2)
axes[0].set_ylim(-0.1, 0.2)
axes[0].set_xlabel("上一期费率（%）")
axes[0].set_ylabel("本期费率（%）")
axes[0].set_title("散点图：本期 vs 上一期（r = {:.2f}）".format(acf[1]), loc="left")
ks = np.arange(1, 181)
axes[1].fill_between(ks, [r.autocorr(k) for k in ks], color=BLUE, alpha=0.85, linewidth=0)
axes[1].axhline(0, color=INK2, lw=0.8)
axes[1].set_xlabel("滞后期数（1 期 = 8 小时，180 期 = 60 天）")
axes[1].set_ylabel("相关系数")
axes[1].set_title("自相关函数 ACF", loc="left")
axes[1].set_ylim(0, 0.85)
save(fig, "05_autocorrelation.png")

print("\n图片已保存到", FIG)
