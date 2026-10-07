"""Rebuild presentation figures from the frozen analysis CSVs, without API access."""
import textwrap

from common import ROOT, rows

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analyze import NAMES, ORDER
from multimodel_bias.constants import NEGATIVE_IAT_WORDS, POSITIVE_IAT_WORDS

OUT=ROOT / "plots"
COLORS={"Black":"#0072B2","White":"#E69F00"}
PANELS={"neutral12":"12-face", "utkface":"UTKFace"}


def save(fig, name, finding, bottom=.11):
    fig.text(.5,.015,textwrap.fill(finding,145),ha="center",va="bottom",fontsize=10,color="#303030")
    fig.tight_layout(rect=(0,bottom,1,.99))
    fig.savefig(OUT / name,dpi=220,bbox_inches="tight",facecolor="white")
    plt.close(fig)


def error(r, prefix):
    value=float(r[prefix+"_rate"])*100
    return [max(0,value-float(r[prefix+"_ci_low"])*100), max(0,float(r[prefix+"_ci_high"])*100-value)]


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    summary=rows(ROOT / "results/analysis/summary.csv")
    significance=rows(ROOT / "results/analysis/significance.csv")
    words=rows(ROOT / "results/analysis/word_counts.csv")
    for panel,title in PANELS.items():
        data=sorted([r for r in summary if r["panel"]==panel],key=lambda r:ORDER.index(r["model_key"]))
        sig={r["model_key"]:r for r in significance if r["panel"]==panel}
        labels=[NAMES[r["model_key"]]+f"\n(n={int(r['responses'])}{'*' if r['complete']=='False' else ''})" for r in data]
        x=np.arange(len(data))
        fig,ax=plt.subplots(figsize=(15,8))
        for race,offset in (("Black",-.2),("White",.2)):
            ax.bar(x+offset,[100*float(r[race.lower()+"_rate"]) for r in data],.38,label=race+" photos",color=COLORS[race],
                   yerr=np.asarray([error(r,race.lower()) for r in data]).T,capsize=3)
        ax.set(title=f"{title} IAT-style negative-term comparison",ylabel="Negative terms (%)",xticks=x)
        ax.set_xticklabels(labels,rotation=30,ha="right",fontsize=9)
        ax.legend(frameon=False)
        ax.grid(axis="y",alpha=.2)
        ax.set_axisbelow(True)
        positive_gaps=sum(float(sig[r['model_key']]['effect'])>0 for r in data)
        save(fig,f"{panel}_01_negative_term_rates.png",
             "Bars show the average percentage of selected words that are negative for each image group. "
             f"Black-photo rates are higher in {positive_gaps} of {len(data)} conditions; this descriptive count is not a significance test. "
             "Error bars are 95% image-bootstrap intervals; an asterisk marks incomplete sampling and expression or serving paths can affect differences.",.12)
        fig,ax=plt.subplots(figsize=(15,8))
        effects=[100*float(sig[r["model_key"]]["effect"]) for r in data]
        errors=np.array([[max(0,float(sig[r["model_key"]]["effect"])-float(sig[r["model_key"]]["ci_low"]))*100,
                          max(0,float(sig[r["model_key"]]["ci_high"])-float(sig[r["model_key"]]["effect"]))*100] for r in data]).T
        ax.errorbar(x,effects,yerr=errors,fmt="o",capsize=4,color="#0072B2")
        ax.axhline(0,color="#666666",lw=1)
        ax.set(title=f"{title} Black-minus-White negative-term gap",ylabel="Race gap (percentage points)",xticks=x)
        ax.set_xticklabels([label+f"\np={float(sig[r['model_key']]['raw_p_value']):.3g}; q={float(sig[r['model_key']]['bh_adjusted_q_value']):.3g}"
                            for label,r in zip(labels,data,strict=True)],rotation=30,ha="right",fontsize=9)
        n_sig=sum(float(s["bh_adjusted_q_value"])<.05 for s in sig.values())
        save(fig,f"{panel}_02_race_gap_significance.png",
             f"Above zero means more negative words for Black photos; below zero means more for White photos. "
             f"{n_sig} of {len(sig)} available conditions have exploratory BH-adjusted q < 0.05. "
             "Intervals are 95% cluster bootstraps; p and q do not establish causality or absence of bias.",.12)
        fig,axes=plt.subplots(1,2,figsize=(15,7))
        axes[0].bar(x,[100*float(r["validity_rate"]) for r in data],color="#009E73")
        axes[0].set(title="Valid three-word outputs",ylabel="Attempted slots valid (%)",ylim=(0,105))
        axes[1].bar(x,[float(r["normalized_word_entropy"]) for r in data],color="#0072B2")
        axes[1].set(title="Word diversity",ylabel="Normalized entropy (0 to 1)",ylim=(0,1))
        for ax in axes:
            ax.set_xticks(x)
            ax.set_xticklabels([NAMES[r["model_key"]] for r in data],rotation=45,ha="right",fontsize=8)
        fig.suptitle(f"{title} output validity and vocabulary diversity")
        save(fig,f"{panel}_03_validity_diversity.png",
             "Validity means the legacy parser found three permitted word matches, not necessarily three unique words or exact formatting. Entropy measures how broadly words are distributed: "
             "zero is concentration in one word and one is uniform use of all 61. Diversity is not a general model-accuracy test.",.12)
        order=list(POSITIVE_IAT_WORDS)+list(NEGATIVE_IAT_WORDS)
        for r in data:
            key=r["model_key"]
            selected=[w for w in words if w["panel"]==panel and w["model_key"]==key]
            counts={(w["race"],w["word"]):int(w["count"]) for w in selected}
            fig,ax=plt.subplots(figsize=(17,8))
            wx=np.arange(61)
            for race,offset in (("Black",-.2),("White",.2)):
                ax.bar(wx+offset,[counts.get((race,w),0) for w in order],.38,color=COLORS[race],label=race+" photos")
            ax.axvline(len(POSITIVE_IAT_WORDS)-.5,lw=1,color="#555555")
            ax.set_xticks(wx)
            ax.set_xticklabels(order,rotation=90,fontsize=8)
            ax.set(title=f"{NAMES[key]}: {title} IAT-style word counts",ylabel="Word selections (count)")
            ax.legend(frameon=False)
            negatives={race:sum(counts.get((race,w),0) for w in NEGATIVE_IAT_WORDS) for race in COLORS}
            save(fig,f"{panel}_{key}_word_counts.png",
                 f"The x-axis lists all 61 permitted words, including words never selected; positive terms precede the divider. "
                 f"Negative words were selected {negatives['Black']} times for Black photos and {negatives['White']} times for White photos. "
                 f"Counts include {r['valid_responses']} valid responses; compare rates when valid sample counts differ.",.10)
    path=ROOT / "results/analysis/gemma_paired_effect.csv"
    if path.exists():
        data=rows(path)
        fig,ax=plt.subplots(figsize=(10,4.5))
        y=np.arange(len(data))
        effect=np.array([float(r["effect"])*100 for r in data])
        err=np.array([[max(0,float(r["effect"])-float(r["ci_low"]))*100,max(0,float(r["ci_high"])-float(r["effect"]))*100] for r in data]).T
        ax.errorbar(effect,y,xerr=err,fmt="o",capsize=5,color="#0072B2")
        ax.axvline(0,color="#555555",lw=1)
        ax.set_yticks(y)
        ax.set_ylim(-.6,len(data)-.4)
        ax.set_yticklabels([PANELS[r["panel"]]+f"\np={float(r['raw_p_value']):.3g}; q={float(r['bh_adjusted_q_value']):.3g}" for r in data])
        ax.set(title="Gemma LoRA: change in the signed race gap",xlabel="Tuned minus base gap (percentage points)")
        save(fig,"gemma_04_paired_tuning_effect.png",
             "Each point directly compares tuned and base outputs on the same images and seeds. Negative values mean the Black-minus-White gap fell; "
             "this is beneficial only in context of the original gap and preserved validity/diversity. Intervals use image-level pairing.",.12)
        fig,axes=plt.subplots(1,2,figsize=(12,6))
        for ax,(panel,title) in zip(axes,PANELS.items(),strict=True):
            focused=[next(r for r in summary if r["panel"]==panel and r["model_key"]==key) for key in ("gemma_base","gemma_tuned")]
            for race,offset in (("Black",-.2),("White",.2)):
                ax.bar(np.arange(2)+offset,[100*float(r[race.lower()+"_rate"]) for r in focused],.38,
                       label=race+" photos",color=COLORS[race],
                       yerr=np.array([error(r,race.lower()) for r in focused]).T,capsize=4)
            ax.set(title=title,ylabel="Negative terms (%)",xticks=[0,1],xticklabels=["Base Gemma","Gemma + LoRA"])
            ax.legend(frameon=False)
        fig.suptitle("Pinned Gemma: base versus August 10 LoRA")
        significant=sum(float(r['bh_adjusted_q_value'])<.05 for r in data)
        save(fig,"gemma_05_base_vs_lora.png",
             "Bars show image-averaged negative-word percentages with 95% bootstrap intervals; the two panels use different y-axis scales. "
             f"{significant} of {len(data)} paired changes pass exploratory BH q < 0.05. "
             "Lower rates alone do not establish improved fairness or preserved accuracy.",.12)
        fig,ax=plt.subplots(figsize=(10,6))
        for field,offset,label,color in (("word_entropy_bits",-.2,"Word entropy","#0072B2"),("unique_response_tuples",.2,"Unique responses","#009E73")):
            retention=[]
            for panel in PANELS:
                a=next(r for r in summary if r["panel"]==panel and r["model_key"]=="gemma_base")
                b=next(r for r in summary if r["panel"]==panel and r["model_key"]=="gemma_tuned")
                retention.append(100*float(b[field])/float(a[field]) if float(a[field]) else np.nan)
            bars=ax.bar(np.arange(2)+offset,retention,.38,label=label,color=color)
            ax.bar_label(bars,labels=[f'{v:.1f}%' for v in retention],padding=4,fontsize=9)
        ax.axhline(100,color="#555555",lw=1,label="Base level")
        ax.axhline(95,color="#777777",lw=1,ls="--",label="Historical 95% retention screen")
        ax.set(title="Gemma LoRA vocabulary retention",ylabel="Adapter / base (%)",xticks=[0,1],xticklabels=list(PANELS.values()),ylim=(0,122))
        ax.legend(frameon=False,fontsize=9,loc='upper center',ncol=2)
        save(fig,"gemma_06_diversity_retention.png",
             "Below 100% means less observed diversity after adaptation; the dashed line is the historical anti-collapse screen, not a validated accuracy threshold. "
             "Unique-response counts depend on valid sample size; interpret with the validity table.",.12)
    print(f"Wrote {len(list(OUT.glob('*.png')))} figures")


if __name__ == "__main__":
    main()
