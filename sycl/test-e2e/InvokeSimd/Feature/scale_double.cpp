// REQUIRES: aspect-fp64
// Check that full compilation works:
// RUN: %{build} -fno-sycl-device-code-split-esimd -Xclang -fsycl-allow-func-ptr -o %t.out
// RUN: env IGC_VCSaveStackCallLinkage=1 IGC_VCDirectCallsOnly=1 %{run} %t.out
//
// VISALTO enable run
// RUN: env IGC_VISALTO=63 IGC_VCSaveStackCallLinkage=1 IGC_VCDirectCallsOnly=1 %{run} %t.out

/*
 * Tests invoke_simd support in the compiler/headers
 * Test case purpose:
 * -----------------
 * To verify that the simple scale example from the invoke_simd spec
 * https://github.com/intel/llvm/blob/sycl/sycl/doc/extensions/experimental/sycl_ext_oneapi_invoke_simd.asciidoc
 * works.
 *
 * Test case description:
 * ---------------------
 * Invoke a simple SIMD function that scales all elements of a SIMD type X by a
 * scalar value n with double.
 *
 * This test also runs with all types of VISA link time optimizations enabled.
 */

#define TEST_DOUBLE_TYPE

#include "scale.cpp"
