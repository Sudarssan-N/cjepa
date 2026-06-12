# Paper

`causal_lewm.tex` — the Causal-LeWM paper draft, updated with results through
Run 13b (collapse-dissection ablations, 3-seed SAVi headline, end-to-end
DINOv2, qualitative slot-binding figures, planning protocol + random baseline).

Figures referenced by the paper live in `figures/` (copied from
`../notes/figures/`, which holds the full 4-window set for all three models).

Compile (no local LaTeX needed — use Overleaf, or in Colab):

```bash
sudo apt-get install -y texlive-latex-extra texlive-science
pdflatex causal_lewm.tex && pdflatex causal_lewm.tex
```

Open slot: §5.6 planning table — success rates for the trained variants
(frozen SAVi-on, propagation-off, end-to-end) under the LeWM-matched CEM
protocol; random baseline (2.0%) already recorded.
