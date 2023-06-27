// Check that full compilation works:
// RUN: %clangxx -DIMPL_SUBGROUP -fsycl -fno-sycl-device-code-split-esimd -Xclang -fsycl-allow-func-ptr %S/../matrix_multiply_accessor_get_pointer.cpp -o %t.out
// RUN: env IGC_VCSaveStackCallLinkage=1 IGC_VCDirectCallsOnly=1 %{run} %t.out
//
// VISALTO enable run
// RUN: env IGC_VISALTO=63 IGC_VCSaveStackCallLinkage=1 IGC_VCDirectCallsOnly=1 %{run} %t.out

/*
 * This tests is the same as
 * InvokeSimd/regression/matrix_multiply_accessor_get_pointer.cpp, but compiles
 * without optional subgroup attribute specified and intended to check that
 * compiler is able to choose subgroup size correctly.
 */
