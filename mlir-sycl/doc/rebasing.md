# Rebasing `sycl` Branch

The SYCL-MLIR project is developed in the [`sycl-mlir` branch of the `intel/llvm`
repository](https://github.com/intel/llvm/tree/sycl-mlir). As it is intended to be a SYCL front-end,
it should be tracking the [`sycl` branch of that
repository](https://github.com/intel/llvm/tree/sycl). Therefore, establishing a process to keep up
with `sycl` progress in `sycl-mlir` is key to success.

## Pulldown

The following steps will create a pulldown PR:

```bash
# Fetch remotes
git fetch --all

# Update sycl-mlir
git checkout sycl-mlir
git pull

# Create new merge branch
git branch -D merge
git checkout -b merge

# Perform merge
git merge upstream/sycl -m "Merge remote-tracking branch 'upstream/sycl' into 'sycl-mlir'"

# Resolve conflicts and test failures (see next section)

# In case the maintainer has installed the 'gh' tool, otherwise push and create PR manually
gh pr create --assignee @me \
    --base sycl-mlir \
    --label sycl-mlir,ignore-lint \
    --title "[SYCL-MLIR] Merge from intel/llvm sycl branch" \
    --body "Please do not squash and merge this PR."

```

After this, the maintainer should update the PR description stating what commits/files require
reviews.

In order to handle test failures, [see next section](#error-solving).

## Error solving

We can identify two error kinds when running:

```bash
cmake --build <build-dir> --target check-cgeist check-mlir-sycl check-polygeist check-clang check-mlir
```

### Build Errors

Maintainer should take action and fix build errors before creatign merge PR.

### Test Errors

If a new E2E test is failing, the maintainer should add it to the `sycl/test-e2e/xfail_tests.txt`
list.

If another kind of test is failing, maintainer should take action as per case basis, trying to solve
the error if it is an easy fix or creating an issue and `XFAIL`ing the test otherwise.
