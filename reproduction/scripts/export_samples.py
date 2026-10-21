"""
Export every sample the paper's QPD tables were computed from -- the Monte
Carlo draws, the bootstrap resamples and the original datasets -- to Excel,
one workbook per section. Samples only: no fits.

Each sample is regenerated with the SAME function and seed the section's sweep
uses (imported from the section modules, so this cannot drift), and is stored
SORTED ascending, i.e. as the order statistics x_(1) <= ... <= x_(N) the QPDs
were fitted to at p_i = i/(N+1).

Layout of a sample tab: column A = i, column B = p_i = i/(N+1); then one column
per replicate. Row 1 = replicate number, row 2 = seed, rows 3.. = values.

Usage (from reproduction/):
    python scripts/export_samples.py --section all      # or mc, su_boot, bimodal, river, fish, geyser
Output: QPD_Samples/ in the folder that contains the repository (override with --out)
"""
import os, sys, argparse, time, datetime
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..', 'code', 'common')); sys.path.insert(0, HERE)
from openpyxl import Workbook
from openpyxl.styles import Font
from openpyxl.cell import WriteOnlyCell

B = Font(name='Arial', bold=True, size=10)
N_REP = 1000


def sample_sheet(wb, name, samples, seeds, first_rep):
    """samples: list of 1-D arrays (same length), already sorted."""
    ws = wb.create_sheet(name)
    n = len(samples[0])
    M = np.column_stack(samples)
    def hdr(v):
        c = WriteOnlyCell(ws, value=v); c.font = B; return c
    ws.append([hdr('replicate'), hdr('')] + [first_rep + j for j in range(len(samples))])
    ws.append([hdr('seed'), hdr('')] + list(seeds))
    ws.append([hdr('i'), hdr('p_i = i/(N+1)')])
    for i in range(n):
        ws.append([i + 1, (i + 1) / (n + 1)] + M[i].tolist())


def original_sheet(wb, name, x, note):
    ws = wb.create_sheet(name)
    x = np.sort(np.asarray(x, float)); n = len(x)
    ws.append([note])
    ws.append(['i', 'p_i = i/(N+1)', 'x (sorted)'])
    for i, v in enumerate(x):
        ws.append([i + 1, (i + 1) / (n + 1), float(v)])


def readme(wb, title, lines):
    ws = wb.create_sheet('README')
    ws.append([title]); ws.append([])
    for l in lines: ws.append([l])
    ws.append([]); ws.append([f'Generated {datetime.date.today()} by reproduction/scripts/export_samples.py'])


COMMON = ['Each sample tab: column A = i, column B = p_i = i/(N+1); one column per replicate.',
          'Row 1 = replicate number, row 2 = seed used to generate it, rows 3.. = the sample values.',
          'Values are SORTED ascending: the order statistics x_(1) <= ... <= x_(N) that every QPD was fitted to.',
          'Every sample is regenerated with the same function and seed the paper\'s scripts use.']


def mc(out):
    from mc_tables_rebuild import DISTS, DIST_PARAMS, draw, mc_seed
    wb = Workbook(write_only=True)
    readme(wb, 'Monte Carlo samples -- Johnson SU / SL / SB, N = 50 and 200 (Tables 3, 4, A1, A2; Figures 2-5, O1-O4)',
           COMMON + ['Johnson parameters (xi, lambda, gamma, delta) = (0, 1, 0.5, 1.2).',
                     'Seed = 42 + N*1000 + replicate, replicates 1..1000; drawn with jpse rvs (mc_tables_rebuild.draw).',
                     'Figure 2 shows Johnson SB, N = 200, replicate 99.'])
    for short in ('su', 'sl', 'sb'):
        name, cls, _ = DISTS[short]; d = cls(**DIST_PARAMS)
        for n in (50, 200):
            reps = range(1, N_REP + 1)
            sample_sheet(wb, f'{short.upper()}_N{n}', [draw(d, n, r) for r in reps], [mc_seed(n, r) for r in reps], 1)
    wb.save(os.path.join(out, 'Samples_1_MonteCarlo_Johnson.xlsx'))


def su_boot(out):
    from bootstrap_section import load_original_sample_unsorted, bootstrap_resample, SEED
    xu = load_original_sample_unsorted()
    wb = Workbook(write_only=True)
    readme(wb, f'SU bootstrap -- the pinned Johnson SU realization (N = 200, seed {SEED}) and its 1000 resamples (Tables 5, 6; Figures 6-8, O5-O7)',
           COMMON + [f'Resample b (0..999): numpy default_rng({SEED}*10000 + b).choice(original, N, replace=True), then sorted.',
                     'Tab "original" = the realization itself.'])
    original_sheet(wb, 'original', xu, f'Johnson SU realization, N = {len(xu)}, seed {SEED}')
    sample_sheet(wb, 'resamples', [np.sort(bootstrap_resample(xu, b)) for b in range(N_REP)],
                 [SEED * 10_000 + b for b in range(N_REP)], 0)
    wb.save(os.path.join(out, 'Samples_2_SU_Bootstrap.xlsx'))


def bimodal(out):
    from bimodal_common import draw, SEED, BIMODAL_BASE, SAMPLE_SIZE
    wb = Workbook(write_only=True)
    readme(wb, 'Bimodal Monte Carlo -- 1000 independent N = 200 draws from the 60/40 Johnson SU mixture (Table 7; Figure 9)',
           COMMON + [f'Replicate r (0..999): seed = {SEED} + {BIMODAL_BASE} + r (bimodal_common.draw).',
                     'Figure 9 shows replicate 436.'])
    sample_sheet(wb, 'samples', [np.sort(draw(r, SAMPLE_SIZE)) for r in range(N_REP)],
                 [SEED + BIMODAL_BASE + r for r in range(N_REP)], 0)
    wb.save(os.path.join(out, 'Samples_3_Bimodal_MC.xlsx'))


def river(out):
    from hydrology_common import load_hydrology, resample, replicate_seed
    x, _ = load_hydrology()
    wb = Workbook(write_only=True)
    readme(wb, 'River gauge heights (Williamson River, N = 95) and 1000 bootstrap resamples (Table 8; Figures 10-14)',
           COMMON + ['No jitter (continuous measurements). Resample b (0..999): hydrology_common.resample, seed in row 2.'])
    original_sheet(wb, 'original', x, f'Annual peak gauge heights (ft), N = {len(x)}')
    sample_sheet(wb, 'resamples', [np.sort(resample(x, b=b)) for b in range(N_REP)],
                 [replicate_seed(b=b) for b in range(N_REP)], 0)
    wb.save(os.path.join(out, 'Samples_4_RiverGauge_Bootstrap.xlsx'))


def fish(out):
    from fish_common import load_fish, resample, replicate_seed
    x, _ = load_fish()
    wb = Workbook(write_only=True)
    readme(wb, 'Fish weights (Babine River, N = 3474) and 1000 jittered bootstrap resamples at 0.5 and 1.0 lb (Tables 9, 10; Figure 16)',
           COMMON + ['Resample b (0..999) at jitter h: draw N indices with replacement, add Uniform(-h, +h) jitter (fish_common.resample); seed in row 2.',
                     'Tab "original" = the recorded (1-lb heaped) weights, no jitter.'])
    original_sheet(wb, 'original', x, f'Recorded fish weights (lb), N = {len(x)}')
    for j in (0.5, 1.0):
        sample_sheet(wb, f'resamples_jitter{j}', [np.sort(resample(x, j, b)) for b in range(N_REP)],
                     [replicate_seed(j, b) for b in range(N_REP)], 0)
    wb.save(os.path.join(out, 'Samples_5_Fish_Bootstrap.xlsx'))


def geyser(out):
    from geyser_common import load_geyser, resample, replicate_seed, JITTER
    x, _ = load_geyser()
    wb = Workbook(write_only=True)
    readme(wb, f'Old Faithful waiting times (N = {len(x)}) and 1000 jittered bootstrap resamples, jitter +/-{JITTER} min (Tables 12, 13)',
           COMMON + [f'Resample b (0..999): N indices with replacement + Uniform(-{JITTER}, +{JITTER}) jitter (geyser_common.resample); seed in row 2.',
                     'Tab "original" = the recorded (whole-minute) waiting times, no jitter.'])
    original_sheet(wb, 'original', x, f'Recorded waiting times (min), N = {len(x)}')
    sample_sheet(wb, 'resamples', [np.sort(resample(x, JITTER, b)) for b in range(N_REP)],
                 [replicate_seed(JITTER, b) for b in range(N_REP)], 0)
    wb.save(os.path.join(out, 'Samples_6_Geyser_Bootstrap.xlsx'))


SECTIONS = {'mc': mc, 'su_boot': su_boot, 'bimodal': bimodal, 'river': river, 'fish': fish, 'geyser': geyser}

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--section', default='all', choices=['all'] + list(SECTIONS))
    ap.add_argument('--out', default=os.path.abspath(os.path.join(HERE, '..', '..', '..', 'QPD_Samples')))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for k in (SECTIONS if a.section == 'all' else [a.section]):
        t0 = time.time(); SECTIONS[k](a.out); print(f'{k:8s} done in {time.time()-t0:5.1f}s', flush=True)
    print('->', a.out)
