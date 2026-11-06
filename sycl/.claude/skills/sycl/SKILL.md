---
name: sycl
description: "Repo-specific guidance for the sycl/ tree of this repo: the SYCL runtime, headers, tools and their tests. Use it for any task that touches code under sycl/, including PR reviews, bug fixes, extensions. It routes to a per-topic guide and carries the conventions every sycl/ change must follow."
---

# Working in `sycl/`

Read the guide for the task; also apply the Common rules below.

| Task | Guide |
|---|---|
| Add, extend, fix or review a test (`sycl/test`, `sycl/test-e2e`, `sycl/unittests`) | `guides/testing.md` |
| Write, update or review an extension specification (`sycl/doc/extensions`) | `guides/extension-specification.md` (plus `guides/extension-specification-properties.md` when the spec defines or accepts properties) |

## Common rules for changes in `sycl/`

- Run clang-format on your changes before committing.
- Comment only what the code does not make obvious; never restate the code.
- Never write absolute paths or Intel-internal links into the repo. A JIRA id may appear as
  plain text (no link) only when the user explicitly asks for it.
