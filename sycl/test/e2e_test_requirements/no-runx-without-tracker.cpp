// This test is intended to ensure that we have no tests marked as RUNx.
// If it fails - please remove RUNx and provide a TODO with a tracker.
//
// REQUIRES: linux
//
// RUN: not grep -rI "RUNx:" %S/../../test-e2e
