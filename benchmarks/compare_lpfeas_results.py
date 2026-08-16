#!/usr/bin/env python3
"""Compare a staged LPfeas run against the last recorded Metal/FP32 results.

Usage:
    compare_lpfeas_results.py <results_dir> [--baseline-dir DIR]

Pairs every <results_dir>/<name>.json with the newest recorded
benchmarks/results/lpfeas-<name>-metal-fp32.json and prints a comparison of
verification status, termination, iterations, and solve/total wall seconds.

Thermal caveat: serial back-to-back sweeps on a laptop-class GPU throttle
hard, and wall time alone misleads. degme measured 1073s hot vs 422s cool on
identical code. Treat solve-time deltas below ~30% as environmental noise
unless a cooled re-run reproduces them; iteration counts and verified status
are the robust comparison axes.
"""
import argparse, glob, json, os, sys

def load_new(results_dir):
    found = {}
    for f in glob.glob(os.path.join(results_dir, '*.json')):
        try:
            j = json.load(open(f))
        except Exception as error:
            print(f'warning: skipping unreadable {f}: {error}', file=sys.stderr)
            continue
        results = j.get('results')
        if not results:
            continue
        name = os.path.basename(f).removesuffix('.json')
        found[name] = results[0]
    return found

def load_baseline(baseline_dir):
    baseline = {}
    for f in glob.glob(os.path.join(baseline_dir, 'lpfeas-*-metal-fp32.json')):
        name = os.path.basename(f).replace('lpfeas-', '').replace('-metal-fp32.json', '')
        try:
            j = json.load(open(f))
        except Exception:
            continue
        if not j.get('results'):
            continue
        generated = j.get('generated_at_utc', '')
        if name not in baseline or generated > baseline[name][0]:
            baseline[name] = (generated, j['results'][0])
    return baseline

def fmt_secs(value):
    if not isinstance(value, (int, float)):
        return 'n/a'
    return f'{value:.1f}'

def fmt_iter(value):
    if not isinstance(value, int):
        return 'n/a'
    return str(value)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('results_dir')
    parser.add_argument('--baseline-dir', default=None)
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    baseline_dir = args.baseline_dir or os.path.join(script_dir, 'results')
    baseline = load_baseline(baseline_dir)
    new = load_new(args.results_dir)

    if not new:
        print(f'no result JSONs found under {args.results_dir}', file=sys.stderr)
        return 2

    rows = []
    for name, (generated, b) in sorted(baseline.items()):
        n = new.get(name)
        if n is None:
            continue
        bs = b['timing_seconds'].get('solve')
        ns = n['timing_seconds'].get('solve')
        rows.append((name, b, n, bs, ns, generated))

    header = (f'{"name":16s} {"ver o/n":9s} {"term o -> n":40s} '
              f'{"iter o -> n":16s} {"solve o -> n (s)":20s} {"delta":>8s}')
    print(header)
    print('-' * len(header))
    improved = regressed = 0
    for name, b, n, bs, ns, generated in rows:
        ver = f'{str(b.get("verified"))[0]}/{str(n.get("verified"))[0]}'
        term = f'{b.get("termination")} -> {n.get("termination")}'
        iters = f'{fmt_iter(b.get("iterations"))} -> {fmt_iter(n.get("iterations"))}'
        solve = f'{fmt_secs(bs)} -> {fmt_secs(ns)}'
        if isinstance(bs, (int, float)) and isinstance(ns, (int, float)) and bs > 0:
            delta = 100.0 * (ns - bs) / bs
            delta_text = f'{delta:+6.1f}%'
            if delta < -5.0:
                improved += 1
            elif delta > 5.0:
                regressed += 1
        else:
            delta_text = '  n/a  '
        print(f'{name:16s} {ver:9s} {term:40s} {iters:16s} {solve:20s} {delta_text:>8s}')

    only_new = sorted(set(new) - set(baseline))
    if only_new:
        print()
        print('instances without a recorded baseline:', ', '.join(only_new))

    print()
    print(f'paired: {len(rows)}  solve-time improved >5%: {improved}  '
          f'regressed >5%: {regressed}')
    print('thermal caveat: solve-time deltas below ~30% are environmental noise '
          'unless a cooled re-run reproduces them; watch iterations and '
          'verified status as the robust axes.')
    return 0

if __name__ == '__main__':
    sys.exit(main())
