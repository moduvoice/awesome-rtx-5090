# Contributing

Thank you for helping improve Awesome RTX 5090.

## Before submitting

Please make sure the project has public, useful RTX 5090-specific evidence. At least one of these should be present in the repository:

- benchmark results with enough environment and workload detail to interpret them;
- setup instructions or patches for Blackwell `sm_120`/`sm_120a`;
- an application workflow explicitly tested on the RTX 5090;
- hardware, power, thermal, or virtualization tooling for the RTX 5090.

A passing mention of “RTX 5090,” a generated hardware list, or a topic tag alone is not sufficient.

## Pull request format

Submit one project per pull request. Add it to the most specific existing section, keep the description factual, and use this format:

```markdown
- [Project name](https://github.com/owner/repository) — What it does and what was tested specifically on the RTX 5090.
```

In the pull request description, include:

1. A permalink to the README, result file, or documentation that contains the RTX 5090 evidence.
2. GPU type (desktop or laptop), VRAM, operating system, driver/CUDA version, and backend when available.
3. For benchmark claims: model, quantization or precision, context/workload, and measurement method.
4. Any relationship you have with the project.

Please do not rank a project with vague superlatives. Report measured values with their conditions and attribute all claims to the source project.

The English [README.md](README.md) is canonical. Please update the matching entry in [README.ko.md](README.ko.md) in the same pull request when possible. Maintainers can help with the Korean translation.

## Updating benchmark rows

The model matrix is an index of published measurements. When adding or updating a row, keep the source's units and link directly to the supporting result. Do not compare rows tested with different backends or workloads as if they formed one leaderboard.

## Review

Maintainers may edit descriptions for consistency, request stronger evidence, move entries, or decline projects that substantially duplicate an existing entry. A project can be removed if its links break, its RTX 5090 material disappears, or its claims become misleading.
