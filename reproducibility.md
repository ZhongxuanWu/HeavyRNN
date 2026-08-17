# Reproducibility of the Flagship Sweep

## Summary

We evaluated whether the completed sweep in `runs/flagship/` reproduces Figure
2 of *Slow Transition to Low-Dimensional Chaos in Heavy-Tailed Recurrent Neural
Networks*. The answer depends on which Lyapunov analysis is used:

1. **The published numerical transition points are closely reproduced when we
   emulate the original study's historical analysis errors.** This analysis
   propagated tangent vectors with a transposed Jacobian, $J^\top Q$, and
   evaluated the Jacobian one state transition too late. It reproduces 7 of the
   paper's 9 transition grid points exactly; the other 2 differ by one adjacent
   gain value.
2. **The results change when the Lyapunov analysis is corrected.** The canonical
   update, $JQ$, with the Jacobian evaluated at the state that generates the
   transition produces substantially earlier transitions in the most
   heavy-tailed conditions. It exactly matches only 5 of the 9 reported grid
   points and accumulates 11 grid steps of absolute transition error, compared
   with 2 for the historical analysis.

Thus, the paper's qualitative conclusion—a slower, broader transition to chaos
for heavier-tailed networks—remains visible under the corrected analysis. The
paper's reported quantitative transition locations, however, appear to reflect
the historical, erroneous Lyapunov calculation.

## Target experiment

The comparison targets Figure 2 of the archived [paper](original/paper.pdf),
which reports the maximum Lyapunov exponent as a function of recurrent gain for
autonomous networks with:

- $N\in\{1000,3000,10000\}$;
- $\alpha\in\{1.0,1.5,2.0\}$;
- 50 logarithmically spaced gains from $0.01$ to $10$;
- 10 random trials per condition;
- 2,900 warmup steps followed by 100 Lyapunov accumulation steps; and
- the leading 100 Lyapunov exponents.

These settings are reproduced by
[`configs/flagship.yaml`](configs/flagship.yaml). The completed run contains all
4,500 expected condition/trial artifacts. Every recorded metric is finite, no
QR stretch floors were used, and the 52 non-CUDA tests pass.

The transition estimator follows the paper's plotting code: for each
$(N,\alpha)$, trial MLEs are averaged at every sampled gain, and $g^*$ is the
first gain where the mean MLE changes from negative to nonnegative. Because the
gain grid is logarithmic, adjacent values differ by a factor of approximately
1.151. A one-grid-step disagreement therefore corresponds to about a 15%
difference in the reported discrete threshold.

## Correct and historical Lyapunov analyses

For the autonomous recurrence

$$
h_{t+1}=\tanh(W h_t),
$$

the Jacobian for the transition from $h_t$ to $h_{t+1}$ is

$$
J_t=D_tW,
\qquad
D_t=\operatorname{diag}\!\left(\operatorname{sech}^2(W h_t)\right).
$$

The canonical forward tangent update is

$$
Q_{t+1}R_{t+1}=J_tQ_t.
$$

This is the algorithm stated in the paper and implemented by the corrected
code in [`simulation.py`](simulation.py#L134-L141). The state update and its
Jacobian use the same preactivation.

The archived study implementation historically made two changes to this
calculation:

1. **Transposed tangent propagation.** A comment in
   [`original/LypAlgo.py`](original/LypAlgo.py#L9-L12) records that the code used
   $J^\top Q$ before July 8, 2026.
2. **One-step-late Jacobian indexing.** The loop first advances `ht` to
   $h_{t+1}$ and then passes that updated state to `rnn_jac`
   ([update](original/LypAlgo.py#L80-L84),
   [Jacobian call](original/LypAlgo.py#L96-L100)). `rnn_jac` subsequently
   recomputes $W h_{t+1}$ ([construction](original/LypAlgo.py#L138-L162)), so
   the tangent update uses $J_{t+1}$ rather than $J_t$.

The historical update was therefore effectively

$$
Q_{t+1}R_{t+1}=J_{t+1}^{\top}Q_t.
$$

This is not equivalent to the canonical calculation for a time-varying
Jacobian sequence. In particular,

$$
J_T^\top\cdots J_1^\top=(J_1\cdots J_T)^\top,
$$

which does not preserve the chronological product
$J_T\cdots J_1$. The difference is especially consequential for finite-time
Lyapunov estimates in nonlinear, strongly non-normal heavy-tailed networks.

## Finding 1: the historical analysis closely reproduces Figure 2

We regenerated the weights from the saved condition seeds and re-evaluated the
saved trajectories using the historical calculation. The re-analysis used all
100 tangent directions, 100 float32 QR steps, the original $10^{-12}$ QR
floor, the maximum of the resulting 100 exponents per trial, and the mean over
10 trials. This matches the analysis dimensions used for Figure 2 while
holding the networks and trajectories fixed across analysis variants.

| $N$ | $\alpha$ | Paper $g^*$ | Historical $J_{t+1}^\top Q$ | Correct $J_tQ$ |
|---:|---:|---:|---:|---:|
| 1,000 | 1.0 | 0.791 | 0.910 | 0.391 |
| 1,000 | 1.5 | 0.791 | 0.687 | 0.596 |
| 1,000 | 2.0 | 0.910 | 0.910 | 0.910 |
| 3,000 | 1.0 | 0.450 | 0.450 | 0.295 |
| 3,000 | 1.5 | 0.596 | 0.596 | 0.596 |
| 3,000 | 2.0 | 0.910 | 0.910 | 0.910 |
| 10,000 | 1.0 | 0.295 | 0.295 | 0.256 |
| 10,000 | 1.5 | 0.518 | 0.518 | 0.518 |
| 10,000 | 2.0 | 0.791 | 0.791 | 0.791 |

The corresponding transition metrics are:

| Analysis | Exact paper grid matches | Total absolute grid-step error |
|---|---:|---:|
| Correct $J_tQ$ | 5/9 | 11 |
| $J_t^\top Q$, correct indexing | 6/9 | 3 |
| Historical $J_{t+1}^\top Q$ | 7/9 | 2 |

The two remaining historical-analysis differences are both adjacent grid
points at $N=1000$. They are much smaller than the discrepancies produced by
the corrected analysis. At $N=1000,\alpha=1$, for example, the historical
mean MLE is $-0.0062$ at the paper's $g=0.791$ and $+0.0118$ at the next
gain, $g=0.910$. The reported grid point is therefore sensitive to the exact
set of ten random realizations.

## Finding 2: the corrected analysis deviates from Figure 2

At four informative paper boundaries, changing from the corrected analysis to
the correctly indexed transposed analysis changes the mean MLE as follows:

| $N$ | $\alpha$ | $g$ | Correct $J_tQ$ | $J_t^\top Q$ |
|---:|---:|---:|---:|---:|
| 1,000 | 1.0 | 0.791 | +0.1470 | -0.0002 |
| 1,000 | 1.5 | 0.791 | +0.0958 | +0.0167 |
| 3,000 | 1.0 | 0.450 | +0.0985 | +0.0131 |
| 10,000 | 1.0 | 0.295 | +0.0496 | -0.0008 |

The transpose changes the mean MLE by roughly $0.05$ to $0.15$ in these
heavy-tailed conditions. By comparison, shifting the Jacobian index from the
correct $J_t$ to the historical $J_{t+1}$ changes the same means by only
about $0.002$ to $0.006$. The indexing error is nevertheless sufficient to
move $N=10000,\alpha=1$ from $g^*=0.339$ to the paper's $g^*=0.295$,
because the mean lies extremely close to zero.

## Why the largest discrepancies occur for small heavy-tailed networks

The corrected and historical analyses disagree most for $N=1000$ with
$\alpha=1.0$ or $1.5$. Three effects reinforce one another there.

### Nonlinear and heterogeneous Jacobian gating

Writing $J_t=D_tW$ makes the orientation error explicit:

$$
J_tQ=D_tWQ,
\qquad
J_t^\top Q=W^\top D_tQ.
$$

The two updates gate opposite sides of the recurrent matrix. At the paper's
reported gains, the $N=1000$ heavy-tailed networks contain substantially more
saturated activity, so $D_t$ is heterogeneous and far from the identity:

| $N$ | $\alpha$ | Mean $1-h^2$ | Fraction with $|h|\geq0.99$ |
|---:|---:|---:|---:|
| 1,000 | 1.0 | 0.738 | 7.43% |
| 3,000 | 1.0 | 0.932 | 1.66% |
| 10,000 | 1.0 | 0.987 | 0.29% |
| 1,000 | 1.5 | 0.810 | 1.72% |
| 3,000 | 1.5 | 0.955 | 0.31% |
| 10,000 | 1.5 | 0.984 | 0.12% |

At larger $N$, the heavy-tail transition occurs at a lower gain while the
activity remains closer to the quiescent state. Consequently $D_t\approx I$,
and the orientation error has less effect on the detected crossing. Gaussian
networks also lack the rare, dominant connections that make heavy-tailed
Jacobians strongly heterogeneous and non-normal.

### Weak self-averaging

For $\alpha<2$, rare extreme weights dominate individual network
realizations. Transition locations therefore concentrate slowly with network
size. In the corrected sweep, the standard deviation of the per-trial crossing
index was 1.64, 1.55, and 0.46 grid steps for $\alpha=1$ at
$N=1000,3000,10000$, respectively. For $\alpha=1.5$, the corresponding
values were 0.87, 0.75, and 0.50. Gaussian values were only 0.40–0.49 grid
steps.

### Coarse thresholding and different random streams

The paper reports the first nonnegative point on a 50-point gain grid rather
than an interpolated root. Small changes around zero therefore move the
reported result by a full 15% grid step. The original study used a sequential
SciPy/global RNG stream beginning with `seed + trial`
([`fig2_parallel.py`](original/run_figs/fig2_parallel.py#L39-L64)); this
reimplementation uses independently keyed condition streams. Both sample the
same target distributions, but they do not generate the same ten matrices.

This distinction matters most for the two $N=1000$ heavy-tail cases. Under
the historical re-analysis, a bootstrap over the current ten
$N=1000,\alpha=1$ trial curves selected the paper's $g=0.791$ crossing about
20% of the time. The archived submission script also specifies only
$N=1000$ and trial 0
([`submit_fig2_parallel.sh`](original/submit_jobs/submit_fig2_parallel.sh#L5-L9)),
not the three sizes and ten trials reported in the paper, so it does not retain
the exact execution provenance of Figure 2.

## Interpretation

The completed flagship sweep supports two distinct reproducibility claims:

- **Historical numerical reproduction:** yes. When the historical transpose
  and indexing behavior are restored, the reported Figure 2 transition points
  are recovered to within one sampled gain everywhere, with exact agreement in
  7 of 9 conditions.
- **Reproduction under the algorithm stated in the paper:** no, not
  quantitatively. The corrected $J_tQ$ analysis yields materially earlier
  transitions for heavy-tailed networks, most prominently at
  $N=1000,\alpha=1$, $N=1000,\alpha=1.5$, and $N=3000,\alpha=1$.

The corrected sweep still reproduces the qualitative scientific pattern:
heavier tails yield a slower, broader approach to chaos, and increasing
network size shifts the transition toward lower gain. What changes is the
quantitative location and magnitude of that transition. Accordingly, results
from `runs/flagship/` should be described as a **corrected-analysis
replication**, while the historical re-analysis strongly suggests how the
paper's published numerical values arose.
