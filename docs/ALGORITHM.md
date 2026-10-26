# CMHA algorithm

1. Generate a fresh sequence/contact-aware self-avoiding walk for each worker.
2. Select one, two, three or four disjoint sequence intervals. Exterior residues
   and at least one intervening boundary residue remain fixed.
3. Release the entire selected union from occupancy. Extend intervals alternately
   inside one stochastic beam. Each internal interval chooses a growth boundary;
   it does not simultaneously grow from both ends.
4. Reject collisions, invalid links and prefixes failing necessary boundary
   reachability checks. Keep bounded, diverse partial assignments.
5. Score only complete feasible alternatives jointly. Count inter-block contacts
   once and update affected H-neighbor degrees, including fixed residues.
6. Include the current fold in the pool, sample directly with Boltzmann weights,
   and adopt the sampled state without a second Metropolis filter.
7. Update the best *visited* physical contact count, local archive and diagnostics.
8. Continue through progressive cyclic cooling, neutral polar-routing refreshes
   and final terminal cooling until the configured stopping rule is met.

## Scores

The FCC box contains integer sites `(x,y,z)` with even `x+y+z`. Nearest-neighbor
displacements are permutations of `(+-1,+-1,0)`. Chain links have squared length 2.
The FCC graph is not a simple-cubic bipartite lattice; do not impose SC sequence parity.

`C_HH` counts adjacent, nonconsecutive H pairs. Physical energy is `-C_HH`.
If `d_i` is a hydrophobic residue's nonbonded H degree, then
`C_excess = sum_H max(0,d_i-2)`. The complete-candidate score is
`F = -2*C_HH - 0.5*lambda*C_excess`, with stage multipliers `1,0.66,0.33,0`.
The pairwise score has a residue-site quadratic representation; excess-core
shaping is evaluated on decoded conformations. This is classical CPU optimization.

## Selection and reconstruction

- Before ordinary cap 24: 75% single and 25% paired proposals.
- At cap 24: 70% single, 15% paired, 10% triple and 5% four-block proposals.
- Single-block lengths: 70% draw from 2-8; 30% from 2 through the current cap.
- Ordinary caps are 12,16,20,24. Triple/four intervals are 6-8 residues each;
  a four-block union can therefore move up to 32 residues.
- Paired selection mixes spatial/contact-deficit guidance and H-flanked P-run
  windows. Bounded failures are retained as null proposals.
- Multi-block beam width is 24, with 12,000 expansions shared across the union.
- Longer single blocks use target-independent resource allocation among widths
  24,48,96; a wider beam is not automatically an improvement.
- Eligible late paired proposals have a 25% chance of enforcing one selected
  missing H-H contact. This does not guarantee a net gain or an accepted change.

## Cooling and diversification

Requested cycle lengths start at 80k,160k,320k,640k and continue doubling. Two
completed stagnant cycles advance the carried stage. Budget quarters also impose
minimum stages, so shaping can disappear before its nominal quarter.

Ordinary cycles cool exponentially from 8.0 to 0.02. Throughput estimates cap
actual cycle lengths, including a `W/8` duration allowance. A terminal pass cools
over the remaining wall budget when another requested cycle will not fit.
Requested proposal counts are not a guarantee that every cycle completes.

Each worker has its own near-best contact-map archive. Late stagnation can
restore a local alternative or the worker best. P-only refreshes leave all H
positions and both implemented energy terms unchanged; recently vacated sites
briefly guide later selection. Workers do not exchange optimized archives.

Finite beams, biased candidate pools, finite boxes, archives and wall-time control
make this an optimization algorithm, not an exact Gibbs sampler, exhaustive
neighborhood solver or certified ground-state method.
