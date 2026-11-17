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

## Unbounded computation from a finite start: not achieved

Follow-up question: can a finite start state compute without bound? That needs either a
source of new gliders (a gun) or stationary objects that survive being hit (reflectors, so
gliders can circulate). I looked for both and found neither in smooth Lenia. The only
positive answer is the trivial one: a Gosper gun runs under the Lenia-form Life rule
(`gol_in_lenia.py`, +2 gliders per 60 steps), so the *rule family* has finite unbounded
computers, but those are Life patterns, not Lenia creatures.

What the search turned up, in three single-kernel worlds:

| world (R=13, T=10) | what exists | what a collision does |
|---|---|---|
| Orbium, m=0.15 s=0.015 | Orbium glider; a fused double Orbium (mass 143, same speed, found as a collision product); a stationary blob (mass 55) that dies after about 500 steps | 366 two-glider crossings (`products.py`): 0, 1 or 2 gliders out, the fused double, or (3 cases) ignition of a soup that grows without limit. No stable stationary product, no periodic emitter. |
| "fleet", m=0.337 s=0.057 | six glider species, a still ring (Circium, mass 128), a stationary rotator (Helicium, mass 387), all stable together | Glider at the ring (`wsurvey.py m127`): passes untouched beyond 32 cells offset; at 12 to 30 cells both vanish completely; at 10 cells or less the soup ignites. Glider at the rotator (`wsurvey.py m387`): ignition for almost every hit, four merges into a bigger glider. Nothing reflects. |
| chain, m about 0.30 s about 0.043 | Catenopteryx increscens: a moving chain of cells that adds cells for ever (`chain_growth.png`; mass 576 to 3458 in 2000 steps, one connected piece) | not tested against gliders |

So unbounded *growth* from a finite seed is easy to get in two forms: runaway soup, and
the growing chain. Neither is a controllable supply of gliders. The chain is the more
interesting one, because the zoo's 4- to 10-cell gliders are short pieces of the same
chain. I scanned (m, s) around it looking for a regime where the growing chain sheds
pieces, which would make it a glider factory (`chainscan.py`, 135 points coarse plus 61
points along the grow/explode boundary). Every point either died, stayed a fixed length,
grew as one piece, or exploded. None shed.

The ring is a usable one-shot memory cell: a glider aimed 12 to 30 cells off-centre is
destroyed iff the ring is there, and the ring is consumed. I did not find a collision
that creates a ring.

### Multi-kernel Lenia: self-replication yes, a gun no

The cloned repo ships about 60 MB of "found" patterns for Chan's extended Lenia (several
kernels, optionally several channels). `mk.py` is my own implementation of that update
rule; `mkscan.py` runs each pattern for 2400 steps and flags those whose creature count
keeps rising while the occupied area stays small. I scanned 348 patterns: all of
`found/212` (one channel, two kernels; copied here as `found_212.json`) and
`213, 221, 222, 223, 231, 232, 241, 242` plus part of `243` (`data_mkscan_more.log`).
Not scanned: the large `231b`, `233*` and the rest of the `24x` to `27x` files.

Result: 27 patterns grow in number. Every one I looked at is a self-replicator, not a gun.

* **Dividing ring** (`found_212.json` index 40, R=20): a stationary ring divides; cells
  that end up surrounded stop dividing. The colony is an orderly packing of rings whose
  radius grows linearly, about 0.06 cells/step, 2 cells to 281 in 4200 steps
  (`colony.py`, `replicator_colony.png`).
* **Motile replicator** (index 5, R=30): swims, divides and leaves dormant rings that
  later start moving (`replicator_motile.png`). Count roughly doubles every 400 steps.
* Six more I rendered (the two patterns named "replicator" in `212`, and `213` 82 and 85,
  `223` 16, `221` 36) are slow bacteria-like colonies or a pair that divides once and
  then stays put.

So a finite seed can produce an unbounded number of creatures in smooth Lenia. What is
still missing is control: a source that emits non-replicating gliders in a fixed
direction, or a signal that travels through the stationary colony. I found neither.
The one pattern explicitly named "emit" (`233`, 15 kernels, 3 channels) did not reproduce
as an emitter: its file gives T=1 for the first kernel and T=10 for the rest; with T=10
it is a single large cell that slowly grows, with T=1 it explodes.

Not tried: hitting the growing chain with a glider to cut it; three-body collisions;
searching (rather than scanning existing patterns) for a gun, for example by
optimising parameters for "stationary parent, departing child".

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
| `gol_in_lenia.py` | Life as a Lenia rule, plus a Gosper gun under it |
| `products.py` | Orbium world: scan collision products |
| `fleet.py`, `world.py`, `wsurvey.py` | fleet world: extract species, fire gliders at stationary targets (run `fleet.py` first) |
| `chainscan.py` | parameter scan around the growing chain |
| `mk.py`, `mkscan.py`, `montage.py`, `colony.py` | multi-kernel Lenia, scan of found patterns for replication, rendering, colony growth law |

Run: `pip install numpy scipy matplotlib`, then `python3 xor_hires.py` (about a minute on
4 cores), `python3 design.py` (about 6 minutes), `python3 lab.py 2 180` for the survey.
