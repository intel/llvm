// E2E tests use a modified expression parser that allows for a third "unknown" 
// boolean state to handle missing run-time features in REQUIRES/UNSUPPORTED 
// statements. This test runs the unit tests related to these expressions.
//
// REQUIRES: linux
// DEFINE: %{E2EExpr}=%S/../../test-e2e/E2EExpr.py
// DEFINE: %{lit_source}=%S/../../../llvm/utils/lit
// RUN: env PYTHONPATH=%{lit_source} python %{E2EExpr}
