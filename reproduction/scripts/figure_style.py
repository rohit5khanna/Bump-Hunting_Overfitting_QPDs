"""
Colours, line styles and output locations for the paper's figures.

The four colours in QPD are reserved for Metalog, QFlex-U, QFlex-TA+ and
QFlex-A+; scripts draw QFlex-T+ in #e377c2. In the figures listed in RESTYLED
every model is then recoloured on save (see _NEW). Anything that is not a QPD
-- GEV, the population curve, the observed data,
bootstrap bands -- draws from OTHER. Log variants inherit the parent colour
('Log Metalog' is Metalog blue). Model order is carried by line style.

`figure_path(name)` puts a paper figure (a name starting with "Figure") in
reproduction/figures/ and any other figure a script draws in
reproduction/figures/other/. Figures listed in RESTYLED are recoloured with
`restyle()` when they are saved, exactly as they appear in the paper.
"""
import matplotlib as mpl
import numpy as np

QPD = {
    'Metalog':   '#1f77b4',   # blue
    'QFlex-U':   '#9467bd',   # purple
    'QFlex-A+':  '#2ca02c',   # green
    'QFlex-TA+': '#ff7f0e',   # orange
}

OTHER = {
    'GEV':        '#8c564b',   # brown -- a reference model, not a QPD
    'truth':      'black',     # population curve (always dashed)
    'empirical':  'black',     # observed EQF / ECDF (thin solid)
    'data':       '#b0b0b0',   # sample points, histogram bars
    'band':       '#a0a0a0',   # bootstrap intervals, CI fills -- see BAND_ALPHA
    'highlight':  '#444444',   # the one bar/point being singled out
    'kde':        '#d62728',   # kernel density estimate, annotation accents
    'zoom':       '#d62728',   # zoom-inset frames and connectors
    'observed':   '#40556b',   # observed data drawn as MARKERS (an EQF
                               # scatter), as distinct from 'empirical',
                               # the black line used for an empirical
                               # CURVE. Desaturated slate: not one of the
                               # four QPD colours, not the GEV brown, not
                               # the KDE red.
}

#: The three Johnson reference distributions (Figure 1). They are populations,
#: not QPDs, so they must never borrow a reserved colour.
REFERENCE = ['#d62728', '#17becf', '#bcbd22']

#: One alpha for every confidence/bootstrap band (with OTHER['band'] it renders
#: as about #cbcbcb: clearly present, still behind the curves).
BAND_ALPHA = 0.55

#: Model order -> line style. Keeps colour free to mean "which model".
KSTYLE = {4: '-', 7: '--', 10: ':', 13: '-.'}
#: ...and marker, where points are drawn.
KMARKER = {4: 'o', 7: 's', 10: '^', 13: 'D'}

#: The paper's standard order sets.
K_STANDARD = [4, 7, 10, 13]
K_GEYSER = list(range(4, 14))
K_FISH_BIMODAL = [9, 10, 11]
K_BIMODAL = list(range(4, 15))


def color(label):
    """Colour for a model label, tolerating the 'Log ' prefix."""
    base = label[4:] if label.startswith('Log ') else label
    if base in QPD:
        return QPD[base]
    if base in OTHER:
        return OTHER[base]
    raise KeyError(f'{label!r} has no assigned colour -- add it to OTHER, '
                   f'never to QPD')


def rcparams():
    return {
        'font.family': 'sans-serif', 'font.size': 11,
        'axes.titlesize': 11, 'axes.labelsize': 10,
        'xtick.labelsize': 9, 'ytick.labelsize': 9,
        'legend.fontsize': 8.5, 'figure.dpi': 150, 'savefig.dpi': 300,
        'axes.grid': True, 'grid.alpha': 0.3,
    }


def apply():
    mpl.rcParams.update(rcparams())


def check_no_reuse(used):
    """Assert a figure has not spent a reserved colour on a non-QPD series.

    `used` maps series label -> colour. Raises on any non-QPD label drawn in
    one of the four reserved colours.
    """
    reserved = {v: k for k, v in QPD.items()}
    for lbl, c in used.items():
        base = lbl[4:] if lbl.startswith('Log ') else lbl
        if c in reserved and base != reserved[c]:
            raise AssertionError(
                f'{lbl!r} is drawn in {c}, which is reserved for '
                f'{reserved[c]!r}')
    return True


# ---------------------------------------------------------------------------
# DRAWING -- the paper's evaluation framework (code/common/eval_framework.py).
#
# Every fitted curve is drawn on the evaluation grid p = 0.001..0.999 ONLY,
# with the density taken from differences of the quantile between neighbouring
# grid points -- the same density the modes are read from. Nothing is drawn
# for p < 0.001 or p > 0.999 and nothing is extrapolated.
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                                  '..', '..', 'code', 'common'))
import eval_framework as EF


def plot_grid(p_grid=None, n=None):
    """The drawing grid: the evaluation grid itself (arguments ignored)."""
    return EF.GRID


def density_curve(quantile, x_view=None, support=None, p_core=None, n_pad=None, p_min=None):
    """(p, x, f) to DRAW a fitted distribution: the evaluation grid, its
    quantile, and the grid-difference density. `quantile` is a fitted QPD's
    bound .quantile method (its model is then used, so Log/Logit fits get the
    framework's cancellation-free differences) or any callable p -> x (e.g. a
    scipy .ppf). For an INVALID fit the stretches where Q decreases (negative
    differences) are returned as NaN, so they are left blank when drawn; the other arguments are accepted for compatibility and ignored."""
    m = getattr(quantile, '__self__', None)
    if m is not None and (hasattr(m, 'metalog') or hasattr(m, 'qflex')
                          or type(m).__name__ in ('Metalog', 'QFlex')):
        if EF.is_valid(m):
            q, f = EF.density(m)
            return EF.GRID, q, f
        q = EF.quantile(m)
    else:
        q = np.asarray(quantile(EF.GRID), float)
    f = EF.density_from_quantile(q)
    return EF.GRID, q, np.where(f > 0, f, np.nan)   # blank where Q decreases (invalid fit)



# ---------------------------------------------------------------------------
# OUTPUT LOCATIONS
FIGURES = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                                         '..', 'figures'))


def figure_path(name):
    """Path (without extension) for a figure: reproduction/figures/<name> when
    `name` is a paper figure ("Figure..."), reproduction/figures/other/<name>
    otherwise."""
    d = FIGURES if name.startswith('Figure') else _os.path.join(FIGURES, 'other')
    _os.makedirs(d, exist_ok=True)
    return _os.path.join(d, name)


# ---------------------------------------------------------------------------
# RESTYLE -- applied on save to the figures listed in RESTYLED.
RESTYLED = {
    'Figure03_MC_JohnsonSU_N50', 'Figure04_MC_JohnsonSU_N200',
    'Figure06_SU_bootstrap_QF_interval', 'Figure07_Metalog_bootstrap_modes',
    'Figure08_QFlexAplus_bootstrap_modes', 'Figure09_bimodal_reference',
    'Figure10_hydrology_EQF', 'Figure11_hydrology_histogram_pdfs',
    'Figure12_hydrology_EQF_bootstrap_interval', 'Figure13_hydrology_mode_scatter',
    'Figure14_hydrology_GEV_Metalog_QFlex', 'Figure15_fish_histogram',
    'Figure16_fish_EQF_bootstrap_interval', 'Figure17_fish_fits_K7_10_13',
    'Figure18_geyser_data', 'Figure19_geyser_fits',
}
_NEW = {   # model -> (colour, line style)
    'Metalog':   ('#005db7', '-'),
    'QFlex-U':   ('#d93562', (0, (5, 1.8))),
    'QFlex-A+':  ('#38b980', (0, (6, 1.6, 1.2, 1.6))),
    'QFlex-TA+': ('#fda031', (0, (9, 2.5))),
    'QFlex-T+':  ('#50b4e9', (0, (1, 1.3))),
    'GEV':       ('#985e3c', (0, (7, 1.8, 1.2, 1.8, 1.2, 1.8))),   # drawn solid and thick, see _recolor_line
}
_OLD = {'Metalog': '#1f77b4', 'QFlex-U': '#9467bd', 'QFlex-A+': '#2ca02c',
        'QFlex-TA+': '#ff7f0e', 'QFlex-T+': '#e377c2', 'GEV': '#8c564b'}
_GRAYS = {'#d62728': '#333333',    # zoom frames, KDE annotations
          '#b0b0b0': '#d4d4d4',    # data (histogram bars, sample points)
          '#a0a0a0': '#bababa'}    # bands
_MIN_LW = {'QFlex-TA+': 2.0, 'QFlex-T+': 2.0}
_OLD2M = {v: k for k, v in _OLD.items()}


def _hex(c):
    from matplotlib.colors import to_hex
    try:
        return to_hex(c, keep_alpha=False).lower()
    except Exception:
        return None


def _newcol(c):
    h = _hex(c)
    if h in _OLD2M:
        return _NEW[_OLD2M[h]][0], _OLD2M[h]
    if h in _GRAYS:
        return _GRAYS[h], None
    return None, None


def _recolor_line(ln, apply_style):
    from matplotlib.colors import to_rgba
    import matplotlib.patheffects as pe
    nc, model = _newcol(ln.get_color())
    if nc is not None:
        ln.set_color(to_rgba(nc, to_rgba(ln.get_color())[3]))
    for getter, setter in ((ln.get_markerfacecolor, ln.set_markerfacecolor),
                           (ln.get_markeredgecolor, ln.set_markeredgecolor)):
        c = getter()
        if isinstance(c, str) and c in ('none', 'None'):
            continue
        n2, m2 = _newcol(c)
        if n2 is not None:
            setter(n2)
            model = model or m2
    # a Metalog line drawn dashed by its script keeps its dashes
    keep_dash = (_NEW.get(model, (None, None))[1] == '-' and ln.get_linestyle() not in ('-', 'solid'))
    if model and apply_style and not keep_dash and ln.get_linestyle() not in ('None', 'none', ' ', ''):
        ln.set_linestyle(_NEW[model][1])
        if model in _MIN_LW:
            ln.set_linewidth(max(ln.get_linewidth(), _MIN_LW[model]))
        if model == 'GEV':        # wide, opaque, solid, white halo
            ln.set_linestyle('-'); ln.set_linewidth(3.4); ln.set_alpha(1.0)
            ln.set_path_effects([pe.Stroke(linewidth=5.2, foreground='white'), pe.Normal()])
    return model


def _recolor_coll(c):
    from matplotlib.colors import to_rgba
    for get, set_ in ((c.get_facecolor, c.set_facecolor), (c.get_edgecolor, c.set_edgecolor)):
        try:
            cols = get()
        except Exception:
            continue
        if cols is None or len(cols) == 0:
            continue
        out, changed = [], False
        for rgba in cols:
            n, _ = _newcol(rgba)
            if n is not None:
                out.append(to_rgba(n, rgba[3])); changed = True
            else:
                out.append(tuple(rgba))
        if changed:
            set_(out)


def _recolor_patch(p):
    from matplotlib.colors import to_rgba
    for get, set_ in ((p.get_facecolor, p.set_facecolor), (p.get_edgecolor, p.set_edgecolor)):
        rgba = get()
        n, _ = _newcol(rgba)
        if n is not None:
            set_(to_rgba(n, rgba[3]))


def restyle(fig):
    """Recolour a finished figure in place (model colours, model line styles,
    dark-gray accents, lighter data/band grays). Where one model colour appears with more than one
    line style in an axes, the styles encode model order K and are kept."""
    from matplotlib.collections import Collection
    for ax in fig.axes:
        lines = list(ax.get_lines())
        styles = {}
        for l in lines:
            _, m = _newcol(l.get_color())
            if m and l.get_linestyle() not in ('None', 'none', ' ', ''):
                styles.setdefault(m, set()).add(str(l.get_linestyle()) + str(l._dash_pattern))
        k_enc = {m for m, st in styles.items() if len(st) > 1}
        for l in lines:
            _, m = _newcol(l.get_color())
            _recolor_line(l, apply_style=(m not in k_enc))
        leg = ax.get_legend()
        for lg in ([leg] if leg else []):
            for l in lg.get_lines():
                _, m = _newcol(l.get_color())
                _recolor_line(l, apply_style=(m not in k_enc))
            for p in lg.get_patches():
                _recolor_patch(p)
            for h in getattr(lg, 'legend_handles', []) or []:
                if isinstance(h, Collection):
                    _recolor_coll(h)
        for c in ax.collections:
            _recolor_coll(c)
        for p in ax.patches:
            _recolor_patch(p)
    for lg in fig.legends:
        for l in lg.get_lines():
            _recolor_line(l, apply_style=True)
        for p in lg.get_patches():
            _recolor_patch(p)
        for h in getattr(lg, 'legend_handles', []) or []:
            if isinstance(h, Collection):
                _recolor_coll(h)


def _install_restyle_on_save():
    import matplotlib.figure as _mf
    if getattr(_mf.Figure.savefig, '_qpd_restyle', False):
        return
    _orig = _mf.Figure.savefig
    _done = set()

    def savefig(self, fname, *a, **k):
        if isinstance(fname, (str, _os.PathLike)):
            stem = _os.path.splitext(_os.path.basename(str(fname)))[0]
            if stem in RESTYLED and id(self) not in _done:
                restyle(self); _done.add(id(self))
        return _orig(self, fname, *a, **k)
    savefig._qpd_restyle = True
    _mf.Figure.savefig = savefig


_install_restyle_on_save()

if __name__ == '__main__':
    for k, v in QPD.items():
        print(f'  {k:<10} {v}')
    print('  ---')
    for k, v in OTHER.items():
        print(f'  {k:<10} {v}')
    assert color('Log Metalog') == QPD['Metalog']
    assert color('Log QFlex-TA+') == QPD['QFlex-TA+']
    assert color('GEV') == OTHER['GEV']
    assert len(set(QPD.values())) == 4
    assert not (set(QPD.values()) & set(OTHER.values()))
    try:
        check_no_reuse({'GEV': QPD['Metalog']})
    except AssertionError as e:
        print('  guard works:', e)
    print('OK')
