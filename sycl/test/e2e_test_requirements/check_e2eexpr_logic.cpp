// E2E tests use a modified expression parser that allows for a third "unknown" 
// boolean state to handle missing run-time features in REQUIRES/UNSUPPORTED 
// statements. This test runs the unit tests related to these expressions.
//
// REQUIRES: linux
// DEFINE: %{e2e_folder}=%S/../../test-e2e
// DEFINE: %{lit_source}=%S/../../../llvm/utils/lit
// RUN: env PYTHONPATH=%{lit_source}:%{e2e_folder} python %S/E2EExpr_tests.py
