import pyqtgraph as pg

def apply_global_styles():
    pg.setConfigOptions(antialias=True)
    pg.setConfigOption('background', 'w')
    pg.setConfigOption('foreground', 'k')

PLOT_COLORS = [
    '#1f77b4', # Bleu
    '#ff7f0e', # Orange
    '#2ca02c', # Vert
    '#d62728', # Rouge
    '#9467bd', # Violet
    '#8c564b', # Marron
    '#e377c2', # Rose
    '#7f7f7f', # Gris
    '#bcbd22', # Jaune-Vert
    '#17becf'  # Cyan
]