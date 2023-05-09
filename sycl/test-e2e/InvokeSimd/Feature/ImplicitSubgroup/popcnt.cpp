// TODO: enable on Windows once driver is ready
// REQUIRES: gpu && linux
// UNSUPPORTED: cuda || hip
//
// NOTE: The feature is not yet supported, there is a discussion on the
// feasibility of tests
// REQUIRES: TEMPORARY_DISABLED
//
// Check that full compilation works:
// RUN: %clangxx -DIMPL_SUBGROUP -fsycl -fno-sycl-device-code-split-esimd -Xclang -fsycl-allow-func-ptr %S/../popcnt.cpp -o %t.out
// RUN: env IGC_VCSaveStackCallLinkage=1 IGC_VCDirectCallsOnly=1 %{run} %t.out
//
// VISALTO enable run
// RUN: env IGC_VISALTO=63 IGC_VCSaveStackCallLinkage=1 IGC_VCDirectCallsOnly=1 %{run} %t.out

/*
 * This tests is the same as InvokeSimd/feature/popcnt.cpp, but compiles without
 * optional subgroup attribute specified and intended to check that compiler is
 * able to choose subgroup size correctly.
 */
