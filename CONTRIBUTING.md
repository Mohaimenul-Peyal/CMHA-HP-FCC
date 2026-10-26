# Contributing

Use Python 3.12 and the pinned requirements. Run the unit tests, archived-fold
validator and source-integrity checks before proposing a change. Include a small
reproducer and state whether a modification changes candidate generation, scoring,
transition, initialization, random calls, stopping or temperature control.

Keep the reported profiles and archived results immutable. New algorithms or
budgets should use new named profiles/output directories, not silently overwrite
the baseline. A favorable single seed is not proof of a general improvement.

Bug reports should include software versions, command, profile and a minimal
nonprivate example. Do not attach credentials or gigabytes of generated traces.
