#!/usr/bin/env python3

import pathlib
import re
import subprocess
import sys

import yaml


ROOT = pathlib.Path(__file__).resolve().parents[2]
WORKFLOW_DIR = ROOT / ".github" / "workflows"
PINNED_ACTION = re.compile(r"^[0-9a-f]{40}$")
USES_LINE = re.compile(r"^\s*-?\s*uses:\s*([^\s#]+)\s*$")
FORBIDDEN_ARTIFACTS = (
    re.compile(r"^runs/"),
    re.compile(r"^logs/"),
    re.compile(r"(^|/)wandb/(run|offline-run)-"),
    re.compile(r"\.(ckpt|pid|pt|pth)$"),
)


def tracked_files():
    output = subprocess.check_output(
        ["git", "ls-files", "-z"], cwd=ROOT
    )
    return [path.decode() for path in output.split(b"\0") if path]


def validate_yaml(path, violations):
    try:
        with path.open("r", encoding="utf-8") as stream:
            yaml.safe_load(stream)
    except (OSError, yaml.YAMLError) as exc:
        violations.append(f"{path.relative_to(ROOT)}: invalid YAML: {exc}")


def validate_workflow(path, violations):
    text = path.read_text(encoding="utf-8")
    if "pull_request_target:" in text:
        violations.append(f"{path.relative_to(ROOT)}: pull_request_target is forbidden")
    if "permissions: write-all" in text:
        violations.append(f"{path.relative_to(ROOT)}: write-all permissions are forbidden")

    for line_number, line in enumerate(text.splitlines(), start=1):
        match = USES_LINE.match(line)
        if match is None:
            continue
        action = match.group(1)
        if action.startswith("./"):
            continue
        if "@" not in action:
            violations.append(
                f"{path.relative_to(ROOT)}:{line_number}: action is not pinned"
            )
            continue
        reference = action.rsplit("@", 1)[1]
        if PINNED_ACTION.fullmatch(reference) is None:
            violations.append(
                f"{path.relative_to(ROOT)}:{line_number}: action must use a full commit SHA"
            )


def main():
    violations = []
    yaml_paths = sorted((ROOT / ".github").rglob("*.yml"))
    yaml_paths.extend(sorted((ROOT / ".github").rglob("*.yaml")))
    for path in yaml_paths:
        validate_yaml(path, violations)
    for path in sorted(WORKFLOW_DIR.glob("*.yml")):
        validate_workflow(path, violations)

    for path in tracked_files():
        if any(pattern.search(path) for pattern in FORBIDDEN_ARTIFACTS):
            violations.append(f"{path}: generated experiment artifact is tracked")

    if violations:
        print("Repository policy violations:", file=sys.stderr)
        for violation in violations:
            print(f"- {violation}", file=sys.stderr)
        return 1

    print(
        f"Validated {len(yaml_paths)} YAML files, "
        f"{len(list(WORKFLOW_DIR.glob('*.yml')))} workflows, and tracked artifacts"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
