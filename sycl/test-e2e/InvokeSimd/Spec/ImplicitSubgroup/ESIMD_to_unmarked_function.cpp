// Check that full compilation works:
// RUN: %clangxx -DIMPL_SUBGROUP -fsycl -fno-sycl-device-code-split-esimd -Xclang -fsycl-allow-func-ptr %S/../ESIMD_to_unmarked_function.cpp -o %t.out
// RUN: env IGC_VCSaveStackCallLinkage=1 IGC_VCDirectCallsOnly=1 %{run} %t.out
//
// VISALTO enable run
// RUN: env IGC_VISALTO=63 IGC_VCSaveStackCallLinkage=1 IGC_VCDirectCallsOnly=1 %{run} %t.out

/*
 * This tests is the same as InvokeSimd/spec/ESIMD_to_unmarked_function.cpp,
 * but compiles without optional subgroup attribute specified and intended to
 * check that compiler is able to choose subgroup size correctly.
 */
