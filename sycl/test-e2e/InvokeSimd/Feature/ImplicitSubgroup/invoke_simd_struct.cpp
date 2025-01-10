// TODO: Passing/returning structures via invoke_simd() API is not implemented
// in GPU driver yet. Enable the test when GPU RT supports it.
// XFAIL: gpu && run-mode
// XFAIL-TRACKER: https://github.com/intel/llvm/issues/14543
//
// RUN: %{build} -DIMPL_SUBGROUP -fno-sycl-device-code-split-esimd -Xclang -fsycl-allow-func-ptr -o %t.out
// RUN: env IGC_VCSaveStackCallLinkage=1 IGC_VCDirectCallsOnly=1 %{run} %t.out
//
// VISALTO enable run
// RUN: env IGC_VISALTO=63 IGC_VCSaveStackCallLinkage=1 IGC_VCDirectCallsOnly=1 %{run} %t.out

/*
 * This tests is the same as InvokeSimd/feature/invoke_simd_struct.cpp, but
 * compiles without optional subgroup attribute specified and intended to check
 * that compiler is able to choose subgroup size correctly.
 */

#include "../invoke_simd_struct.cpp"
