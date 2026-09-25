"""
comutation_robust.py

Robustness add-ons for a custom co-mutation metric (used earlier):
    Cij = (mij)^2 / (mi * mj)

Covers the four robustness points:
  (a) LD reference statistics (D, D', r^2) to benchmark Cij against:
  (b) Pseudocount smoothing + MAF floor filtering for small-n stability.
  (c) Phylogenetic non-independence correction via identity-based
      sequence reweighting.
  (d) Multiple testing correction (Benjamini-Hochberg FDR).

Assumes an aligned MSA as a list of equal-length strings (one per sequence).
Positions are 0-indexed columns in the alignment. Gap/ambiguous characters
default to {'-', 'N', 'n', '.'} and are excluded from frequency counts.

Dependencies: numpy, scipy, statsmodels.
"""

from collections import Counter, defaultdict
from itertools import combinations
import numpy as np

GAP_CHARS = {'-', 'N', 'n', '.', 'X', 'x'}


# ---------------------------------------------------------------------------
# Loading an aligned MSA from a FASTA file
# ---------------------------------------------------------------------------

def load_fasta(path, uppercase=True, check_aligned=True):
    """
    Minimal FASTA parser
    path          : path to a .fasta / .fa / .aln file
    uppercase     : force all bases to uppercase (avoids 'a' vs 'A' mismatches)
    check_aligned : if True, raise a clear error if sequences are NOT all the
                    same length, reports error.

    Returns: (headers, seqs) -- two parallel lists (headers[i] corresponds
    to seqs[i]), in file order.
    """
    headers = []
    seqs = []
    current = []
    with open(path) as f:
        for line in f:
            line = line.rstrip('\n')
            if not line:
                continue
            if line.startswith('>'):
                if current:
                    seqs.append(''.join(current))
                    current = []
                headers.append(line[1:].strip())
            else:
                current.append(line.strip())
        if current:
            seqs.append(''.join(current))

    if not seqs:
        raise ValueError(f"No sequences found in {path} -- is this a valid FASTA file?")

    if uppercase:
        seqs = [s.upper() for s in seqs]

    if check_aligned:
        lengths = sorted(set(len(s) for s in seqs))
        if len(lengths) > 1:
            raise ValueError(
                f"Sequences in {path} are not all the same length "
                f"(lengths found: {lengths}). This looks like unaligned "
                "FASTA, not an MSA -- align it first (e.g. MAFFT: "
                "`mafft input.fasta > aligned.fasta`) before using this pipeline."
            )

    return headers, seqs


# Alternative, if you wish to use biopython
# (handles more FASTA edge cases / other alignment formats via AlignIO):
#
#   from Bio import SeqIO
#   records = list(SeqIO.parse(path, "fasta"))
#   headers = [r.id for r in records]
#   seqs = [str(r.seq).upper() for r in records]


# ---------------------------------------------------------------------------
# (0) Identify variable sites from a raw MSA
# ---------------------------------------------------------------------------

def find_variable_sites(seqs, min_allele_count=2, min_minor_freq=0.0, require_aligned=True):
    """
    Scan an aligned MSA (list of equal-length strings) and return the column
    indices that are polymorphic -- i.e. have more than one observed allele
    among non-gap characters.

    seqs             : list of equal-length aligned sequences (str)
    min_allele_count : minimum number of sequences carrying the minor allele
                        for a site to be reported. Raise this (e.g. to 2-3)
                        to filter out singleton calls that are more likely to
                        be sequencing/alignment errors than real variants.
    min_minor_freq    : minimum RAW (unweighted, unsmoothed) minor allele
                        frequency required for a site to be reported.
                        0.0 = report any site with >=2 alleles, however rare.
    require_aligned    : if True, raise if sequences aren't all the same
                        length (catches "this isn't actually an MSA" early).

    Returns: sorted list of ints (1-indexed column positions).

    This is intentionally a permissive, first-pass filter -- it just narrows
    "every alignment column" down to "columns worth testing at all." Run the
    returned positions through site_maf() + maf_floor_filter() (or straight
    into robust_pairwise_scan) for the real, pseudocount-smoothed MAF floor
    before anything is treated as significant.
    """
    if not seqs:
        return []

    length = len(seqs[0])
    if require_aligned:
        for i, s in enumerate(seqs):
            if len(s) != length:
                raise ValueError(
                    f"Sequence {i} has length {len(s)}, expected {length} -- "
                    "sequences must be aligned (equal length) to call variable sites."
                )

    variable_positions = []
    for pos in range(length):
        counts = Counter(s[pos] for s in seqs if s[pos] not in GAP_CHARS)
        if len(counts) < 2:
            continue  # invariant column, or all-gap/missing at this position
        total = sum(counts.values())
        ranked = counts.most_common()
        minor_count = ranked[1][1]
        minor_freq = minor_count / total if total > 0 else 0.0
        if minor_count < min_allele_count or minor_freq < min_minor_freq:
            continue
        variable_positions.append(pos+1)

    return variable_positions


def describe_variable_sites(seqs, positions=None):
    """
    Summary of variable sites -- useful for scanning a new
    MSA before committing to a MAF floor / min_allele_count.

    positions: if None, runs find_variable_sites() with permissive defaults
               (min_allele_count=2, min_minor_freq=0.0) first.

    Returns: list of dicts, one per site:
        {pos, n_alleles, major_allele, major_count,
         minor_allele, minor_count, minor_freq, n_scored}
    """
    if positions is None:
        positions = find_variable_sites(seqs)

    summary = []
    for pos in positions:
        counts = Counter(s[pos] for s in seqs if s[pos] not in GAP_CHARS)
        ranked = counts.most_common()
        total = sum(counts.values())
        summary.append({
            "pos": pos,
            "n_alleles": len(ranked),
            "major_allele": ranked[0][0], "major_count": ranked[0][1],
            "minor_allele": ranked[1][0], "minor_count": ranked[1][1],
            "minor_freq": ranked[1][1] / total if total > 0 else 0.0,
            "n_scored": total,
        })
    return summary


# ---------------------------------------------------------------------------
# (a) LD reference statistics: D, D', r^2
# ---------------------------------------------------------------------------

def ld_stats(mi, mj, mij):
    """
    Standard linkage-disequilibrium statistics, computed alongside Cij so you
    can see whether Cij is tracking something LD doesn't already capture.

    mi, mj  : minor allele frequencies at site i, j (0-1)
    mij     : joint minor-allele frequency at (i, j) (0-1)

    Returns dict with D, D_prime, r2, and the original Cij for comparison.
    """
    D = mij - mi * mj

    denom_r2 = mi * (1 - mi) * mj * (1 - mj)
    r2 = (D ** 2) / denom_r2 if denom_r2 > 0 else np.nan

    if D >= 0:
        Dmax = min(mi * (1 - mj), (1 - mi) * mj)
    else:
        Dmax = min(mi * mj, (1 - mi) * (1 - mj))
    Dprime = D / Dmax if Dmax > 0 else np.nan

    Cij = (mij ** 2) / (mi * mj) if (mi > 0 and mj > 0) else np.nan

    return {"D": D, "Dprime": Dprime, "r2": r2, "Cij": Cij}


# ---------------------------------------------------------------------------
# (b) Pseudocount smoothing + MAF floor
# ---------------------------------------------------------------------------

def site_maf(seqs, pos, pseudocount=0.5, weights=None):
    """
    Minor allele frequency at a single column, with Laplace/Jeffreys
    smoothing (pseudocount) so 0- or 1-count minor alleles don't produce
    wildly unstable frequencies at small n.

    weights: optional per-sequence weights (see sequence_weights below).
             If None, all sequences are weighted equally.

    Returns None if the column has <2 observed alleles (invariant / all-gap).
    """
    if weights is None:
        weights = np.ones(len(seqs))

    counts = defaultdict(float)
    total_w = 0.0
    for s, w in zip(seqs, weights):
        a = s[pos]
        if a in GAP_CHARS:
            continue
        counts[a] += w
        total_w += w

    if len(counts) < 2 or total_w == 0:
        return None

    ranked = sorted(counts.items(), key=lambda kv: -kv[1])
    major_allele, major_w = ranked[0]
    minor_allele, minor_w = ranked[1]

    maf = (minor_w + pseudocount) / (total_w + 2 * pseudocount)
    return {
        "minor_allele": minor_allele,
        "major_allele": major_allele,
        "maf": maf,
        "n_effective": total_w,
    }


def joint_maf(seqs, pos_i, pos_j, minor_i, minor_j, pseudocount=0.5, weights=None):
    """
    Joint minor-allele frequency mij, smoothed with a Jeffreys-style
    pseudocount over the 2x2 (minor/not-minor at i) x (minor/not-minor at j)
    table (total added pseudo-mass = 4 * pseudocount across the table,
    matching the per-cell pseudocount used in site_maf's 2-category case).
    """
    if weights is None:
        weights = np.ones(len(seqs))

    joint_w = 0.0
    total_w = 0.0
    for s, w in zip(seqs, weights):
        ai, aj = s[pos_i], s[pos_j]
        if ai in GAP_CHARS or aj in GAP_CHARS:
            continue
        total_w += w
        if ai == minor_i and aj == minor_j:
            joint_w += w

    if total_w == 0:
        return None

    mij = (joint_w + pseudocount) / (total_w + 4 * pseudocount)
    return mij, total_w


def maf_floor_filter(site_maf_dict, floor=0.10):
    """
    site_maf_dict: {pos: result_of_site_maf(...)}
    Drop sites whose (smoothed) MAF is below `floor` before they ever enter
    a Cij calculation, this protects from single
    sequencing/alignment errors dominating a pair's score at n~30.
    """
    return {p: v for p, v in site_maf_dict.items() if v is not None and v["maf"] >= floor}


# ---------------------------------------------------------------------------
# (c) Phylogenetic non-independence: identity-based sequence reweighting
# ---------------------------------------------------------------------------

def sequence_weights(seqs, identity_threshold=0.80, ignore_gaps=True):
    """
    DCA/Henikoff-style reweighting: each sequence's weight is 1 / (number of
    sequences, including itself, within `identity_threshold` fractional
    identity of it). Near-duplicate / clonal sequences from the same clade
    end up downweighted as a group, which is a cheap first-line defense
    against phylogenetic clustering inflating Cij.

    Returns: weights (np.ndarray), Neff = weights.sum()
    """
  
    n = len(seqs)
    L = len(seqs[0])
    arr = np.array([list(s) for s in seqs])

    weights = np.ones(n)
    for i in range(n):
        sim_count = 0
        for j in range(n):
            if ignore_gaps:
                mask = (arr[i] != '-') & (arr[j] != '-') & (arr[i] != 'N') & (arr[j] != 'N')
                denom = mask.sum()
                if denom == 0:
                    continue
                identity = np.sum((arr[i] == arr[j]) & mask) / denom
            else:
                identity = np.mean(arr[i] == arr[j])
            if identity >= identity_threshold:
                sim_count += 1
        weights[i] = 1.0 / sim_count

    return weights, weights.sum()


def check_clade_confound(seqs, pos_i, pos_j, weights, unweighted_cij, pseudocount=0.5):
    """
    Quick diagnostic: recompute Cij using reweighted (phylogeny-corrected)
    frequencies and compare to the unweighted value. A large drop suggests
    the original signal was substantially driven by shared ancestry rather
    than independent co-occurrence. Returns both values plus % change.
    """
    mi_info = site_maf(seqs, pos_i, pseudocount=pseudocount, weights=weights)
    mj_info = site_maf(seqs, pos_j, pseudocount=pseudocount, weights=weights)
    if mi_info is None or mj_info is None:
        return {"weighted_cij": np.nan, "unweighted_cij": unweighted_cij, "pct_change": np.nan}

    mij_w, _ = joint_maf(seqs, pos_i, pos_j, mi_info["minor_allele"], mj_info["minor_allele"],
                          pseudocount=pseudocount, weights=weights)
    weighted_cij = (mij_w ** 2) / (mi_info["maf"] * mj_info["maf"])
    pct_change = 100 * (weighted_cij - unweighted_cij) / unweighted_cij if unweighted_cij else np.nan
    return {"weighted_cij": weighted_cij, "unweighted_cij": unweighted_cij, "pct_change": pct_change}


# ---------------------------------------------------------------------------
# Bootstrap CI on Cij (pairs naturally with the reweighting/pseudocount steps)
# ---------------------------------------------------------------------------

def bootstrap_cij_ci(seqs, pos_i, pos_j, n_boot=2000, pseudocount=0.5,
                      weights=None, ci=95, random_state=None):
    """
    Sequence-level (row) bootstrap: resample sequences with replacement,
    recompute Cij each time. Gives a confidence interval to report alongside
    your permutation p-value -- the p-value tells you "is this above chance
    given the marginals", the CI tells you "how precisely is this estimated
    given only n sequences".
    """
    rng = np.random.default_rng(random_state)
    n = len(seqs)
    if weights is None:
        weights = np.ones(n)

    boot_vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        boot_seqs = [seqs[k] for k in idx]
        boot_w = weights[idx]

        mi_info = site_maf(boot_seqs, pos_i, pseudocount=pseudocount, weights=boot_w)
        mj_info = site_maf(boot_seqs, pos_j, pseudocount=pseudocount, weights=boot_w)
        if mi_info is None or mj_info is None:
            continue
        result = joint_maf(boot_seqs, pos_i, pos_j, mi_info["minor_allele"], mj_info["minor_allele"],
                            pseudocount=pseudocount, weights=boot_w)
        if result is None:
            continue
        mij, _ = result
        cij = (mij ** 2) / (mi_info["maf"] * mj_info["maf"])
        boot_vals.append(cij)

    boot_vals = np.array(boot_vals)
    lower = np.percentile(boot_vals, (100 - ci) / 2)
    upper = np.percentile(boot_vals, 100 - (100 - ci) / 2)
    return {"ci_lower": lower, "ci_upper": upper, "boot_values": boot_vals}


# ---------------------------------------------------------------------------
# (d) Multiple testing correction: Benjamini-Hochberg FDR
# ---------------------------------------------------------------------------

def bh_fdr(pvalues, alpha=0.05):
    """
    Benjamini-Hochberg FDR correction across all tested site pairs.
    Implemented directly (no statsmodels dependency required).

    pvalues: array-like of raw permutation p-values, one per pair tested.
    Returns: (qvalues, is_significant) both aligned to input order.
    """
    pvals = np.asarray(pvalues, dtype=float)
    n = len(pvals)
    order = np.argsort(pvals)
    ranked = pvals[order]

    ranks = np.arange(1, n + 1)
    bh_vals = ranked * n / ranks
    # enforce monotonicity (running minimum from the largest p-value down)
    bh_vals = np.minimum.accumulate(bh_vals[::-1])[::-1]
    bh_vals = np.clip(bh_vals, 0, 1)

    qvals = np.empty(n)
    qvals[order] = bh_vals
    is_significant = qvals <= alpha
    return qvals, is_significant


# ---------------------------------------------------------------------------
# Pair-wise calculation of the co-mutation
# ---------------------------------------------------------------------------

def robust_pairwise_scan(seqs, variable_positions, maf_floor=0.10, pseudocount=0.5,
                          identity_threshold=0.80, n_boot=1000, fdr_alpha=0.05,
                          use_reweighting=True, random_state=0):
    """
      1. compute (weighted) per-site MAF, apply MAF floor
      2. compute Cij + LD stats (D, D', r2) for surviving pairs
      3. bootstrap CI on each Cij
      4. flag pairs with a large weighted-vs-unweighted Cij drop (clade confound)
    """
    rng_state = random_state
    weights = np.ones(len(seqs))
    if use_reweighting:
        weights, neff = sequence_weights(seqs, identity_threshold=identity_threshold)
    else:
        neff = len(seqs)

    site_info = {}
    for pos in variable_positions:
        info = site_maf(seqs, pos, pseudocount=pseudocount, weights=weights)
        if info is not None:
            site_info[pos] = info
    site_info = maf_floor_filter(site_info, floor=maf_floor)

    surviving_positions = sorted(site_info.keys())
    results = []
    for pos_i, pos_j in combinations(surviving_positions, 2):
        mi_info = site_info[pos_i]
        mj_info = site_info[pos_j]
        mij, n_used = joint_maf(seqs, pos_i, pos_j, mi_info["minor_allele"], mj_info["minor_allele"],
                                 pseudocount=pseudocount, weights=weights)

        stats = ld_stats(mi_info["maf"], mj_info["maf"], mij)
        ci = bootstrap_cij_ci(seqs, pos_i, pos_j, n_boot=n_boot, pseudocount=pseudocount,
                               weights=weights, random_state=rng_state)

        unweighted_ci_info = None
        if use_reweighting:
            unweighted_ci_info = check_clade_confound(
                seqs, pos_i, pos_j, weights=np.ones(len(seqs)), unweighted_cij=stats["Cij"],
                pseudocount=pseudocount
            )

        results.append({
            "pos_i": pos_i, "pos_j": pos_j,
            "mi": mi_info["maf"], "mj": mj_info["maf"], "mij": mij,
            "Cij": stats["Cij"], "D": stats["D"], "Dprime": stats["Dprime"], "r2": stats["r2"],
            "ci_lower": ci["ci_lower"], "ci_upper": ci["ci_upper"],
            "n_used": n_used, "Neff": neff,
            "clade_confound_check": unweighted_ci_info,
        })

    return results


# ---------------------------------------------------------------------------
# Runnning the pipeline for a given fasta file
# ---------------------------------------------------------------------------

import csv
import os


def batch_process_fasta_files(fasta_paths, min_allele_count=2, maf_floor=0.10,
                               identity_threshold=0.90, n_boot=1000, pseudocount=0.5,
                               variable_sites_out="variable_sites.csv",
                               metrics_out="comutation_metrics.csv"):
    """
    Run load_fasta -> find_variable_sites -> robust_pairwise_scan over a list
    of aligned MSA FASTA files, and write two consolidated CSVs:

      variable_sites_out : one row per (file, variable site), with allele
                            identity, counts, and raw MAF -- your first-pass
                            inventory of what's variable in each file.
      metrics_out         : one row per (file, site pair), with every metric
                            from robust_pairwise_scan (Cij, D, D', r2,
                            bootstrap CI, Neff, weighted/unweighted Cij and
                            the clade-confound %change) flattened into columns.


    Returns (all_variable_sites, all_metrics) as lists of dicts, in case you
    want to inspect results in-session (e.g. in a notebook) instead of, or
    in addition to, reading the CSVs back in.
    """
    all_variable_sites = []
    all_metrics = []

    for fasta_path in fasta_paths:
        file_label = os.path.basename(fasta_path)
        print(f"\n=== Processing {file_label} ===")

        try:
            headers, seqs = load_fasta(fasta_path)
        except Exception as e:
            print(f"  SKIPPED -- could not load ({e})")
            continue

        print(f"  {len(seqs)} sequences, length {len(seqs[0])}")
        if len(seqs) < 30:
            print(f"  Warning: only {len(seqs)} sequences (below the 30-sequence minimum).")

        variable_positions = find_variable_sites(seqs, min_allele_count=min_allele_count)
        if not variable_positions:
            print("  No variable sites found -- skipping.")
            continue

        for row in describe_variable_sites(seqs, variable_positions):
            row_out = {"source_file": file_label, "n_sequences": len(seqs)}
            row_out.update(row)
            all_variable_sites.append(row_out)

        if len(variable_positions) < 2:
            print("  Fewer than 2 variable sites -- no pairs to test.")
            continue

        pair_results = robust_pairwise_scan(
            seqs, variable_positions, maf_floor=maf_floor, pseudocount=pseudocount,
            identity_threshold=identity_threshold, n_boot=n_boot
        )

        for row in pair_results:
            flat = {"source_file": file_label, "n_sequences": len(seqs)}
            for k, v in row.items():
                if k == "clade_confound_check":
                    if v is not None:
                        flat["weighted_cij"] = v["weighted_cij"]
                        flat["unweighted_cij"] = v["unweighted_cij"]
                        flat["clade_pct_change"] = v["pct_change"]
                    else:
                        flat["weighted_cij"] = flat["unweighted_cij"] = flat["clade_pct_change"] = None
                else:
                    flat[k] = v
            all_metrics.append(flat)

        print(f"  {len(variable_positions)} variable sites -> {len(pair_results)} pairs tested")

    if all_variable_sites:
        with open(variable_sites_out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_variable_sites[0].keys()))
            writer.writeheader()
            writer.writerows(all_variable_sites)
        print(f"\nWrote {len(all_variable_sites)} variable-site rows -> {variable_sites_out}")
    else:
        print("\nNo variable sites found in any file -- nothing written to variable_sites_out.")

    if all_metrics:
        with open(metrics_out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_metrics[0].keys()))
            writer.writeheader()
            writer.writerows(all_metrics)
        print(f"Wrote {len(all_metrics)} pair rows -> {metrics_out}")
    else:
        print("No pairs tested in any file -- nothing written to metrics_out.")

    return all_variable_sites, all_metrics


# ---------------------------------------------------------------------------
# Run it: edit FASTA_FILES below and run `python comutation_robust.py`
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # ------------------------------------------------------------------
    # EDIT THIS LIST: add the paths to all the aligned MSA FASTA files you
    # want processed. Each file is its own independent alignment (sequences
    # must be equal length WITHIN a file; different files can differ).
    # ------------------------------------------------------------------
    FASTA_FILE = [
      "/path/to/alignment_1.fasta",
    ]

    MIN_ALLELE_COUNT = 2
    MAF_FLOOR = 0.10
    IDENTITY_THRESHOLD = 0.90
    N_BOOT = 1000

    VARIABLE_SITES_OUT = "variable_sites.csv"
    METRICS_OUT = "comutation_metrics.csv"

    existing_files = [p for p in FASTA_FILE if os.path.exists(p)]
    missing_files = [p for p in FASTA_FILE if not os.path.exists(p)]
    if missing_files:
        print("Warning: these paths in FASTA_FILE were not found and will be skipped:")
        for p in missing_files:
            print(" ", p)

    if not existing_files:
        print("\nNo valid FASTA_FILES found -- running on a built-in toy example instead "
              "so you can see the expected output format.")

    batch_process_fasta_files(
        existing_files,
        min_allele_count=MIN_ALLELE_COUNT,
        maf_floor=MAF_FLOOR,
        identity_threshold=IDENTITY_THRESHOLD,
        n_boot=N_BOOT,
        variable_sites_out=VARIABLE_SITES_OUT,
        metrics_out=METRICS_OUT,
    )
