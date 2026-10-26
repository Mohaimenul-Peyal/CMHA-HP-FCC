# Repository Maintenance

The public repository is
[Mohaimenul-Peyal/CMHA-HP-FCC](https://github.com/Mohaimenul-Peyal/CMHA-HP-FCC).
Its source and original accompanying materials use the MIT license.

## Publish updates

From the local repository, inspect changes and commit the intended files:

```bash
git status
git add .
git commit -m "Update CMHA documentation and reproducibility materials"
git push origin main
```

`git add .` includes the dotfiles. Generated experiments under `runs/` are ignored.
Do not commit credentials, local environments, caches or duplicated raw traces.
Keep the numerical search snapshots unchanged; a changed method needs a separate
profile or implementation version and corresponding validation.

If Git requests an identity, configure `user.name` and `user.email` in this
repository using your own details. Do not change another user's global identity.
Regenerate `MANIFEST.sha256` after changing distributed files and check it with
`python tools/verify_repository.py --manifest` before pushing.

## Citation and archived releases

The GitHub Actions workflow checks Python 3.12 on Linux and Windows.
`CITATION.cff` supplies the software citation, and `.zenodo.json` supplies archive
metadata when the repository is linked to Zenodo and a tagged release is archived.
After an actual archive DOI exists, add it to the citation metadata and manuscript.
Do not reuse an earlier Q-RBHA DOI for this different software release.

The short description is in `GITHUB_DESCRIPTION.txt`. Repository topics cover
protein folding, FCC lattices, the HP model, heat-bath annealing and reproducible
combinatorial optimization.
