# Security Policy

## Supported versions

This project is pre-release (v0, milestone M0). Only the latest commit on the
default branch receives security fixes.

## Reporting a vulnerability

Please do **not** open a public issue for a security problem.

Use GitHub private vulnerability reporting: open the repository's **Security**
tab and choose **Report a vulnerability**. Include:

- what the issue is and where it lives (file, function, input type),
- steps or an input file that reproduces it,
- the impact you expect.

You can expect an acknowledgement within 5 business days and a status update
within 30 days. Fixes are released on the default branch, and reporters are
credited unless they ask not to be.

## Scope

In scope:

- Code execution or file access triggered by a crafted input file (YAML, CSV,
  Parquet) or scenario override.
- Path traversal through `*_ref` fields that point at curve or profile files.
- Dependency vulnerabilities that affect how this package is used.

Out of scope: numerical accuracy of model results, and findings that require
an attacker who can already edit the repository or run code on your machine.

## Handling sensitive data

The model consumes inputs that are often commercially confidential, such as
utility tariff sheets, interconnection study results, gas contracts and load
forecasts.

- Do not commit confidential or licensed data. `data/raw/` and
  `data/processed/` are for non-sensitive or public inputs only.
- Do not commit credentials, API keys or tokens. Run `git diff --cached`
  before committing.
- Generated outputs (`outputs/`) are git-ignored. Review them before sharing,
  because they can reproduce confidential inputs.

## Secure development rules

- YAML is read with `yaml.safe_load` only. Never use `yaml.load` on input files.
- No `pickle`, `eval`, `exec`, or shell calls built from input values.
- Inputs are validated through pydantic models before use.
- Reference fields (`*_ref`) resolve only inside the configured data directory.
- Dependencies are declared in `pyproject.toml`. Keep them current and review
  security advisories before upgrading or adding any.
