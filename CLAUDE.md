# Project: project13

Empty workspace; ORBIT Codex skills installed project-locally under
`.agents/skills/`. Do not infer this project's research topic, claims, status,
or constraints from other projects.

## Current State

```yaml
stage: workspace_scaffolded
domain: unspecified
orbit_codex_skills: installed_project_local
last_updated: "2026-06-04"
```

No experiment, data job, external call, or paper/proposal claim is authorized by
this file.

## Reuse Policy

Adopt only cross-project operational practice (workflow structure, verification,
file-change safety, environment recording, review protocol, skill boundaries).
Do not import another project's domain facts, results, method names, frozen-file
lists, decisions, approvals, datasets, or metrics unless the user says they apply
here.

## Working Rules

- Inspect local files before assuming.
- Keep edits scoped to the request and the smallest relevant file set; never
  revert unrelated user changes.
- Prefer project-local instructions and skills over global defaults.
- Distinguish proposal / plan / diagnostic evidence / completed experiment /
  claimed result; never present planned, hypothetical, or borrowed results as
  obtained.
- Record assumptions that affect downstream work; verify with the narrowest
  useful check and state what stays unverified.

## Hard Boundaries

- No GPU jobs, long-running experiments, paid API calls, human studies, large
  downloads, or external integrations without explicit user approval.
- Do not modify `.agents/skills/` unless asked to update installed skills.
- Do not write global config (`~/.codex`, shell startup, credential stores) or
  external directories unless the user names that target.

## Environment

No runtime environment declared yet. Identify it from local files (`README`,
`pyproject.toml`, `requirements.txt`, `environment.yml`, `Makefile`) before
running code, and document any environment you create.

Declared GPU resources: SSH nodes `an17` and `an29`, each 8×A800.

Availability does not authorize launching GPU jobs without explicit user
approval.

## ORBIT Skill Scope

ORBIT Codex skills installed project-locally in `.agents/skills/` (78 top-level
entries incl. `shared-references`; no `.aris` manifest). Prefer these over global
skills.

## Codex CLI Review

Call Codex for an independent review/help pass proactively — on long, multi-step,
ambiguous, high-impact, or complex tasks; when uncertain; when changes touch
shared behavior; or when results surprise. Use narrow non-interactive prompts;
split broad tasks and merge the findings.

```bash
codex exec \
  -c 'sandbox_mode="danger-full-access"' \
  -c 'approval_policy="never"' \
  -c 'model_reasoning_effort="xhigh"' \
  -c 'service_tier="fast"' \
  -c 'features.fast_mode=true' \
  -c 'shell_environment_policy.inherit="all"' \
  --skip-git-repo-check \
  -C /dir/to/your/project \
  "your review prompt"
```

Treat Codex output as independent advice, not authority; verify important claims
before changing course. Do not use it to bypass the hard boundaries above.
