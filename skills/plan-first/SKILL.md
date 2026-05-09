---
name: plan-first
description: Structured planning workflow for coding tasks. Use at the start of new features, bug fixes, refactors, or implementation requests; analyze first, ask critical questions once, create TODO.md, get approval, then execute task by task.
---

# Plan-First Workflow

## Rules

- Never write code, create files, or run implementation commands before a `TODO.md` is approved.
- Never assume missing critical information. Ask instead.
- Never skip phases.
- Never go off-plan. If new work is discovered, add it to `TODO.md` and ask for approval before doing it.

## Phase 1 - Analyze the Project

Read the project silently before asking anything. Check:

1. Directory structure, top two levels.
2. Project manifests such as `package.json`, `pubspec.yaml`, `go.mod`, `requirements.txt`, `Cargo.toml`, `pom.xml`, or equivalents.
3. Existing dependencies and versions.
4. Build system and scripts such as `Makefile`, `scripts/`, and CI config.
5. `README.md` or `README.*`.
6. Existing `TODO.md`, `TASKS.md`, `.todo`, or open issue files.

Do not output analysis results unless directly relevant to the questions.

## Phase 2 - Ask Clarifying Questions

After analysis, identify gaps that would block correct implementation.

- Ask at most five questions in one message.
- Only ask what is critical and cannot be inferred from project files.
- Number the questions.
- Do not ask about things already answerable from the project files.
- Do not split into multiple rounds unless the user's answer creates a new blocker.

Use this format:

```text
Before I create the plan, I need a few things clarified:

1. Should the new endpoint require authentication?
2. Is there a preferred database?
3. Should existing tests be updated, or only new ones added?
```

Wait for the user's response before proceeding.

## Phase 3 - Create TODO.md

Using the analysis and the user's answers, write `TODO.md` in the project root:

```markdown
# TODO

## Goal
One sentence describing what will be built or fixed.

## Tasks

### 1. <Phase Name>
- [ ] <Concrete, measurable action>
- [ ] <Concrete, measurable action>

### 2. <Phase Name>
- [ ] <Concrete, measurable action>
- [ ] <Concrete, measurable action>

## Notes
Any constraints, decisions, or known risks recorded here.
```

Tasks must be small, independently verifiable, dependency ordered, and checkable as done or not done.

After writing the file, show the full contents and ask:

```text
I've created TODO.md. Does this plan look correct?
Reply YES to start, or tell me what to change.
```

## Phase 4 - Revision Loop

If the user requests changes:

1. Ask targeted follow-up questions to resolve the disagreement.
2. Rewrite `TODO.md`.
3. Show the updated plan and ask for approval again.

Repeat until approved.

## Phase 5 - Execute the Plan

Once approved:

1. Work through tasks in order, one at a time.
2. After completing each task, mark it done in `TODO.md`.
3. State which task is starting before beginning it.
4. Do not start the next task until the current one is complete.
5. Do not perform work not listed in `TODO.md`.

If an unlisted task is required, stop, add it under `## Discovered Tasks`, explain why it is needed, and ask for approval before continuing.

When all tasks are marked complete, write:

```text
All tasks in TODO.md are complete.
```
