# Security Policy

## Supported Version

Security fixes are applied to the latest commit on `main`.

## Reporting a Vulnerability

Report vulnerabilities through the repository's private GitHub security advisory form. Do not include credentials, W&B API keys, private datasets, or exploit details in a public issue.

Include the affected commit, reproduction steps, expected impact, and whether the issue requires an NVIDIA GPU or Isaac Gym installation.

## Secrets

W&B and GitHub credentials must be stored in local credential stores or GitHub Actions secrets. They must not be committed to configuration files, logs, checkpoints, metadata, or experiment artifacts.
