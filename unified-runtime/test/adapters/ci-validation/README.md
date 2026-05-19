# CI Validation Test Suite

This test suite is designed to validate CI logging and test categorization in GitHub Actions workflows.

## Purpose

These tests intentionally generate different test outcomes to verify that our CI workflow correctly:
- Extracts and displays test statistics
- Categorizes tests into appropriate groups (Passed, Failed, Skipped, etc.)
- Properly displays collapsed sections in GitHub Actions Step Summary

## Test Scenarios

| Test File | Expected Outcome | CI Category |
|-----------|------------------|-------------|
| `test_pass.test` | Pass | Passed Tests |
| `test_fail.test` | Fail | Failed Tests |
| `test_unsupported.test` | Skip | Skipped/Unsupported Tests |
| `test_xfail.test` | Expected Fail | Expected Failures |
| `test_unexpected_pass.test` | Unexpected Pass | Unexpectedly Passed Tests |
| `test_timeout.test` | Timeout | Timed Out Tests |
| `test_unresolved.test` | Unresolved | Unresolved Tests |

## Running Locally

```bash
# Run all validation tests
cd build
cmake --build . --target check-unified-runtime-adapter-ci-validation

# Or use LIT directly with timeout
python3 llvm/utils/lit/lit.py \
  --show-pass --show-unsupported --show-xfail --succinct \
  --timeout 5 \
  unified-runtime/test/adapters/ci-validation
```

## Expected Statistics

When run on a Linux system, you should see approximately:
- Total Discovered Tests: 7
- Passed: 1
- Failed: 1
- Skipped/Unsupported: 1
- Expected Failures: 1
- Unexpectedly Passed: 1
- Timed Out: 1
- Unresolved: 1

## Notes

- `test_timeout.test` requires LIT timeout to be set (e.g., `--timeout 5`)
- `test_unsupported.test` is marked UNSUPPORTED on linux/windows (i.e., all platforms)
- These tests should NOT be run in production CI on every commit (too slow due to timeout test)
