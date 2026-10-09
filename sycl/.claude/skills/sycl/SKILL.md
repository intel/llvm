---
name: sycl
description: "Repo-specific guidance for the sycl/ tree of this repo: the SYCL runtime, headers, tools and their tests. Use it for any task that touches code under sycl/, including PR reviews, bug fixes, extensions. It routes to a per-topic guide and carries the conventions every sycl/ change must follow."
---

# Working in `sycl/`

Refer necessary guides for the task; Also apply the Common rules below.

| Task | Guide |
|---|---|
| Add, extend, fix or review a test (`sycl/test`, `sycl/test-e2e`, `sycl/unittests`) | `guides/testing.md` |

## Common rules for changes in `sycl/`

- Run clang-format on your changes before commiting.
- Comment only what the code does not make obvious; never restate the code.
- Never write absolute paths or Intel-internal links into the repo. A JIRA id may appear as
  plain text (no link) only when the user explicitly asks for it.
