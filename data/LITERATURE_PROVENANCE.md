# Published reference values and sequence identity

All energy comparisons use the same binary-HP FCC convention: physical energy
is minus the number of nonconsecutive nearest-neighbor H-H contacts. The JSON
comparison file stores positive contact counts, not negative energies.

| Source | Exact numerical source | Role |
| --- | --- | --- |
| Liu et al., 2013, ELP+, DOI 10.1103/PhysRevE.88.052704 | Table V, article page 052704-7 | Standard-run best energies for F90, S, F180, R and CASP-derived sets |
| Liu et al., 2014, IWL, DOI 10.1103/PhysRevE.90.042715 | Table IV, article page 042715-6 | IWL and published Native/HPstruct reference energies |
| Do Duc et al., 2018, K-ACO, DOI 10.1109/ISMICT.2018.8573710 | Table III, PDF page 4 | Standard comparison, not the separate extended-budget Table V |
| Dotu et al., 2011, DOI 10.1109/TCBB.2011.41 | Author supplement at https://bioinformatics.bc.edu/clotelab/FCCproteinStructure/ | Sequence identity and constraint-programming context |
| Backofen and Will, 2006, DOI 10.1007/s10601-006-6848-8; Mann et al., 2008, DOI 10.1186/1471-2105-9-230 | Original method/tool papers | Meaning and scope of CPSP/HPstruct, not a new run performed for this study |

Sequence strings, residue counts, H counts and SHA-256 hashes are preserved in
`paper_data.json`. The 25 non-CASP strings were cross-checked against the author
supplement. The six CASP-derived strings were also checked against the sequence
table in the locally supplied Shatabda thesis (Table 3.7).

## Important exclusions and distinctions

- S3 is the literal 161-residue, 100-H benchmark. The K-ACO 162-residue label is
  not silently assumed sequence-identical; its S3 value is omitted from the
  numerical table. No residue was inserted or removed to fit an energy value.
- F90 length-label and historical F180 filename inconsistencies were resolved
  using literal strings, not filenames alone.
- H1-H10 is the ten-instance Harvard 48-residue collection, not the separate
  twelve-instance table in ELP+/IWL.
- Standard K-ACO Table III and extended Table V are not mixed. The extended
  values differ on R2 (-348 rather than -347) and 3no6 (-411 rather than -410).
- Native denotes a published reference value attributed to HPstruct. A copied
  value is not itself an optimality proof; earlier papers discuss unresolved
  threading bounds on some long sequences, including F180_1 and F180_2.
- No Native value is assumed for 3on7. Its 575-contact fold improves the compared
  stochastic table entries but is not claimed to be globally optimal.
- The 2021 REWL seq_58 through seq_136 collection is distinct and is not
  relabeled as CMHA evidence in this article.
- Transfer-learning PSO's reported F180 averages are not inserted into a column
  of best energies. MJ interaction results are not compared numerically with HP.
- Published budgets and hardware differ; no cross-study wall-time ranking is
  inferred from these tables.
