"""Offline IAT61 analysis. Repetitions are nested within images, never independent faces."""
import itertools
import math
import sys
from collections import Counter, defaultdict

import numpy as np

from common import ROOT, json_read, merge_slots, read_jsonl, rows, write_csv, write_json

sys.path.insert(0, str(ROOT / "src"))
from multimodel_bias.constants import IAT_WORDS, NEGATIVE_IAT_SET
from multimodel_bias.iat import parse_three_words
from multimodel_bias.uncertainty import benjamini_hochberg, stratified_exact_binary_label_permutation

N = 100000
SEED = 9112026
OUT = ROOT / "results/analysis"
NAMES = {"gemma_base": "Gemma 3 12B base (local)", "gemma_tuned": "Gemma 3 12B + LoRA (local)"}
NAMES.update({m["key"]:m["display_name"] for m in json_read(ROOT / "config/study.json")["panels"]["utkface"]["models"]})
ORDER = list(NAMES)


def valid(value):
    return str(value).lower() in {"true", "1"}


def bootstrap_mean(values, seed=SEED, samples=N):
    x = np.asarray(values, dtype=float)
    if not len(x):
        return [math.nan, math.nan]
    rng = np.random.default_rng(seed)
    means = np.concatenate([x[rng.integers(len(x),size=(min(1000,samples-i),len(x)))].mean(1)
                            for i in range(0,samples,1000)])
    return np.percentile(means, [2.5,97.5]).tolist()


def signflip(values, seed=SEED, samples=N):
    x = np.asarray(values, dtype=float)
    if not len(x):
        return math.nan
    observed = abs(x.mean())
    if len(x) <= 16:
        signs = np.asarray(list(itertools.product((-1,1),repeat=len(x))))
        return float((np.abs((signs*x).mean(1)) >= observed-1e-12).mean())
    rng, count = np.random.default_rng(seed), 0
    for i in range(0,samples,1000):
        signs = rng.choice((-1,1),size=(min(1000,samples-i),len(x)))
        count += int((np.abs((signs*x).mean(1)) >= observed-1e-12).sum())
    return (count+1)/(samples+1)


def adjusted(records, family):
    finite = [r for r in records if math.isfinite(r["raw_p_value"])]
    for r,q in zip(finite,benjamini_hochberg([r["raw_p_value"] for r in finite]),strict=True):
        r.update(bh_adjusted_q_value=q, correction_family=family, family_tests=len(finite), exploratory=True)
    return records


def image_scores(records):
    groups = defaultdict(list)
    for r in records:
        if valid(r["valid_three"]):
            groups[r["image_id"],r["race"],r["gender"]].append(int(r["negative_count"])/3)
    return [dict(image_id=k[0],race=k[1],gender=k[2],negative_rate=float(np.mean(v)),valid_repetitions=len(v))
            for k,v in groups.items()]


def gap_statistics(images, panel, pair_lookup, seed=SEED):
    races = {r: [x["negative_rate"] for x in images if x["race"] == r] for r in ("Black","White")}
    if panel == "utkface":
        pairs = defaultdict(dict)
        for row in images:
            pairs[pair_lookup[row["image_id"]]][row["race"]] = row["negative_rate"]
        complete = [p for p in pairs.values() if set(p)=={"Black","White"}]
        diff = [p["Black"]-p["White"] for p in complete]
        effect = float(np.mean(diff)) if diff else math.nan
        lo,hi = bootstrap_mean(diff, seed)
        p = signflip(diff,seed)
        method,unit,count = "two-sided matched-pair sign-flip", "complete matched image pairs", len(diff)
        races = {r:[p[r] for p in complete] for r in races}
    else:
        effect = float(np.mean(races["Black"])-np.mean(races["White"]))
        rng = np.random.default_rng(seed)
        boot = np.zeros(N)
        strata = {(race,gender):np.asarray([x["negative_rate"] for x in images if x["race"]==race and x["gender"]==gender])
                  for race in races for gender in ("Female","Male")}
        for (race,gender), x in strata.items():
            if not len(x):
                return dict(effect=effect,ci_low=math.nan,ci_high=math.nan,raw_p_value=math.nan,sample_count=len(images),
                            sample_unit="images",test="unavailable: missing stratum"), races
            boot += (1 if race=="Black" else -1)*x[rng.integers(len(x),size=(N,len(x)))].mean(1)/2
        lo,hi = np.percentile(boot,[2.5,97.5])
        _,p,_ = stratified_exact_binary_label_permutation(images,"negative_rate","race","gender",group_a="Black",group_b="White")
        method,unit,count = "exact race-label permutation within gender", "distinct face images", len(images)
    return dict(effect=effect,ci_low=float(lo),ci_high=float(hi),raw_p_value=p,sample_count=count,
                sample_unit=unit,test=method), races


def normalize(records):
    out=[]
    for original in records:
        row=dict(original)
        words=parse_three_words(row.get("fallback_raw") or row.get("sampled_raw") or row.get("response", ""))
        was_valid=valid(row["valid_three"])
        if was_valid and (len(words)!=3 or "|".join(words)!=row["words"]):
            raise ValueError("Parser/record mismatch")
        row.update(word_count=len(words), negative_count=sum(w in NEGATIVE_IAT_SET for w in words),
                   valid_three=was_valid and len(words)==3)
        out.append(row)
    return out


def main():
    cfg=json_read(ROOT / "config/study.json")
    lookup={r["image_id"]:r["match_pair_id"] for r in rows(ROOT / cfg["panels"]["utkface"]["manifest"])}
    local=read_jsonl(ROOT / "results/local/responses.jsonl")
    all_significance, all_summaries, word_rows, delta_tests, delta_images=[],[],[],[],[]
    for panel in cfg["panels"]:
        archive=rows(ROOT / f"results/hosted/{panel}/archived_responses.csv")
        new=list({r["task_id"]:r for r in read_jsonl(ROOT / f"results/hosted/{panel}/responses.jsonl")}.values())
        records=normalize(merge_slots(archive,new)+[r for r in local if r["panel"]==panel])
        populated={k for r in records for k,v in r.items() if v not in ("",None)}
        records=[{k:v for k,v in r.items() if k in populated} for r in records]
        write_csv(OUT / f"{panel}_responses.csv",records)
        sig, clusters=[],[]
        for key in ORDER:
            selected=[r for r in records if r["model_key"]==key]
            if not selected:
                continue
            valid_rows=[r for r in selected if r["valid_three"]]
            images=image_scores(selected)
            if not images:
                continue
            statistic,races=gap_statistics(images,panel,lookup)
            sig.append(dict(panel=panel,model_key=key,model_name=NAMES[key],metric="Black-minus-White negative-term rate",**statistic))
            clusters.extend(dict(r,panel=panel,model_key=key) for r in images)
            words=Counter(w for r in valid_rows for w in r["words"].split("|"))
            total=sum(words.values())
            entropy=-sum((c/total)*math.log2(c/total) for c in words.values()) if total else 0
            expected=600 if panel=="neutral12" else 2000
            answered=[r for r in selected if r.get("response_source") not in {"api_error","budget_stopped"}]
            summary=dict(panel=panel,model_key=key,model_name=NAMES[key],planned_responses=expected,
                         attempted_slots=len(selected),responses=len(answered),valid_responses=len(valid_rows),
                         validity_rate=len(valid_rows)/len(selected),distinct_images=len(images),
                         exact_format_rate=sum(valid(r.get("exact_format")) for r in selected)/len(selected),
                         repeated_word_response_rate=sum(len(set(r["words"].split("|")))<3 for r in valid_rows)/len(valid_rows) if valid_rows else 0,
                         complete=len(answered)==expected,archived_responses=sum(r.get("provenance")=="archived" for r in selected),
                         unique_response_tuples=len({r["words"] for r in valid_rows}),distinct_iat_words=len(words),
                         word_entropy_bits=entropy,normalized_word_entropy=entropy/math.log2(61),
                         top_three_word_share=sum(c for _,c in words.most_common(3))/total if total else 0,
                         api_error_slots=sum(r.get("response_source")=="api_error" for r in selected),
                         new_inference_cost_usd=sum(float(r.get("new_cost_usd") or 0) for r in selected))
            for race,values in races.items():
                low,high=bootstrap_mean(values)
                summary.update({race.lower()+"_rate":float(np.mean(values)),race.lower()+"_ci_low":low,race.lower()+"_ci_high":high})
                group=[r for r in selected if r["race"]==race]
                summary[race.lower()+"_fallback_rate"]=sum(r.get("response_source","").startswith("fallback") for r in group)/len(group) if group else math.nan
                counts=Counter(w for r in valid_rows if r["race"]==race for w in r["words"].split("|"))
                word_rows.extend(dict(panel=panel,model_key=key,race=race,word=w,count=counts[w],negative=w in NEGATIVE_IAT_SET) for w in IAT_WORDS)
            all_summaries.append(summary)
        adjusted(sig,f"{panel}: available model-condition race gaps")
        all_significance.extend(sig)
        write_csv(OUT / f"{panel}_image_scores.csv",clusters)
        # Direct paired tuning effect, retaining only seed/image repetitions valid in both conditions.
        paired=defaultdict(dict)
        for r in records:
            if r["model_key"] in ("gemma_base","gemma_tuned") and r["valid_three"]:
                paired[r["image_id"],int(r["iteration"])][r["model_key"]]=r
        by_image=defaultdict(list)
        info={}
        for (image_id,iteration),pair in paired.items():
            if len(pair)==2:
                a,b=pair["gemma_base"],pair["gemma_tuned"]
                if int(a["seed"])!=int(b["seed"]) or a["image_sha256"]!=b["image_sha256"]:
                    raise ValueError("Unpaired Gemma seeds/images")
                by_image[image_id].append((int(b["negative_count"])-int(a["negative_count"]))/3)
                info[image_id]=a
        if by_image:
            images=[dict(image_id=k,race=info[k]["race"],gender=info[k]["gender"],negative_rate=float(np.mean(v)),paired_valid_repetitions=len(v)) for k,v in by_image.items()]
            stat,_=gap_statistics(images,panel,lookup)
            delta_tests.append(dict(panel=panel,metric="change in signed race gap: tuned minus base",**stat))
            delta_images.extend(dict(x,panel=panel,metric="tuned minus base negative-term rate") for x in images)
    adjusted(delta_tests,"Gemma paired tuning effect: two dataset race-gap changes")
    for name,data in (("summary",all_summaries),("significance",all_significance),("word_counts",word_rows),
                      ("gemma_paired_effect",delta_tests),("gemma_paired_image_changes",delta_images)):
        if data:
            write_csv(OUT / f"{name}.csv",data)
    write_json(OUT / "analysis_manifest.json",dict(bootstrap_samples=N,seed=SEED,
               analysis_unit="image cluster or matched image pair; repeats are not independent faces",
               significance="exploratory two-sided; BH within each dataset across available model conditions",
               completeness=[{k:r[k] for k in ("panel","model_key","responses","planned_responses","complete")} for r in all_summaries]))
    print(f"Analyzed {len(all_summaries)} model/dataset cells; {len(delta_tests)} direct paired tuning tests")


if __name__ == "__main__":
    main()
