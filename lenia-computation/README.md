# General computation in Lenia

Question: can Lenia (Bert Chan's continuous cellular automaton) do general computation, and how?
Everything here is plain single-kernel, single-channel Lenia, `A <- clip(A + dt*G(K*A))`,
simulated with `lenia.py`. Patterns come from `animals.json` (from
[Chakazul/Lenia](https://github.com/Chakazul/Lenia), MIT).

## Answer in three levels

**1. Trivially yes, in Lenia's rule space.** Conway's Life is one setting of Lenia's own
update rule (R=1, 3x3 kernel with centre weight 1/2, rectangular growth, dt=1).
`gol_in_lenia.py` checks the two are identical cell-for-cell for 500 steps of random soup.
Life is Turing-complete, so the Lenia family is. This says nothing about the smooth
creatures people mean by "Lenia".

**2. Yes for Boolean circuits, with real Orbium gliders, by collision-based computing.**
A bit is the presence or absence of an Orbium (`R=13k, T=10, m=0.15, s=0.015`) on a
known path. Two head-on reactions are enough:

| reaction (head-on, lateral offset b in units of R/13) | result | logic |
|---|---|---|
| b about 0 to 3 | both gliders vanish | `A AND NOT B` on each path |
| 4 <= abs(b) <= 17 | both survive, both deflected by 20 to 74 degrees (set by b) | on the new paths: `A AND B` twice; on the old paths: `A AND NOT B`, `B AND NOT A` |
| abs(b) >= 24 | pass untouched | wire crossing |

The scatter reaction is the Fredkin/Toffoli billiard-ball interaction gate. With a constant
glider as one input it gives NOT (constant keeps going straight iff the signal is absent),
fan-out (signal and constant are both deflected iff the signal is present) and a turn of
the wire by a chosen angle. Head-on reactions do not depend on arrival time, only on
lateral alignment, which makes them easy to compose.

Demonstrated circuit (`xor_hires.py`, figure `xor_hires.png`): **single-rail XOR from
single-copy inputs using five gliders** (inputs x, y; constants c, c2, c3), three collision
layers, including a collision that consumes the product of an earlier scatter. All four
input cases simulate correctly. The same run is a half adder: when x=y=1 the two
scattered inputs are two copies of the carry.

    x=0 y=0 -> 0     x=0 y=1 -> 1     x=1 y=0 -> 1     x=1 y=1 -> 0

**3. Unbounded (Turing-style) computation: only with an unbounded supply of gliders.**
Every gate consumes fresh constant gliders, and I found no stationary reflector or glider
gun in single-kernel Orbium Lenia. So a finite start state runs a circuit of bounded size.
Any Boolean circuit can be laid out this way in principle, and an infinite (periodic)
start state could unroll an unbounded computation, but neither of those is demonstrated
here beyond the XOR. Treat the universality claim as an engineering argument, not a proof.

## What I measured

* **Resolution matters a lot.** At the standard R=13, collision outcomes are chaotic:
  they flip between annihilation, one glider and two gliders with a one-cell change of
  offset, a change of start distance, or a rotation of the whole scene (`lab.py 1 ...`).
  A freshly rotated axis-aligned Orbium is also metastable: it side-steps about 6 cells
  after roughly 450 steps and then wanders by a few cells.
* **At R=26 (same creature, 2x finer grid) the head-on collision is a smooth function of the
  impact parameter**, repeatable across start phase to under 0.4 degrees and nearly
  unchanged when the scene is rotated by 20 degrees (`data_headon_R26.txt`). Gliders fly
  straight at 1.2303 cells/step with constant mass. Fine table of outgoing headings for a
  right-mover meeting a left-mover (`N[h1,h2]` = N gliders out with headings h1, h2):

      b: -24..-21 pass (deflected 4-14 deg)   -20..-18 annihilate
         -17:[154,334] -14:[156,336] -12:[151,331] -10:[143,323] -8:[126,306] -6:[112,292] -4:[106,286]
         -3,-2 annihilate   -1 scatter [68,248]   0..+3 annihilate
         +4:[71,251] +8:[51,231] +10:[34,214] +12:[25,205] +14:[23,203] +17:[23,203]
         +18,+19 annihilate   +20..+24 pass

  The central annihilation window has a hole at b=-1, so aim for b of about +1.5.
  Non-head-on crossings (60 degrees) are still erratic at R=26; I did not use them.
* **R=13 still works if you calibrate each placement by simulation.** `design.py` builds a
  six-glider XOR from 90-degree annihilations only (NOR logic, inputs supplied in duplicate
  because annihilation cannot copy a signal), searching integer offsets for each new glider.
  Its truth table also passes (`data_xor_R13_design.log`). Margins are 0 to 4 cells, so this
  one is fragile by construction.

## Caveats

* The R=26 XOR was run at one orientation and one set of distances. I did not test it
  under rotation or perturbation, and the output glider's ray was fitted from a
  constants-only run of the same geometry.
* Deeper circuits will need per-stage calibration: scatter angles vary by a few tenths of a
  degree with phase, and the annihilation window is only about 4 units wide.
* Gliders interact out to roughly 22 units of lateral separation, so wires must be kept
  further apart than that in space or time.

## Files

| file | purpose |
|---|---|
| `lenia.py` | FFT Lenia, pattern decoding and placement, blob finder |
| `glider.py` | Orbium rotated to head +x |
| `lab.py` | two-glider collision survey at any angle, offset, rotation and resolution |
| `hires.py`, `scatter.py` | R=26 scenes; fit outgoing rays of a scatter |
| `xor_hires.py`, `render.py` | the five-glider XOR and its figure |
| `scene.py`, `pairmap.py`, `design.py` | R=13 axis-aligned scenes and search-based annihilation circuits |
| `gol_in_lenia.py` | Life as a Lenia rule |

Run: `pip install numpy scipy matplotlib`, then `python3 xor_hires.py` (about a minute on
4 cores), `python3 design.py` (about 6 minutes), `python3 lab.py 2 180` for the survey.
