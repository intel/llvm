// TODO: enable on Windows once driver is ready
// REQUIRES: gpu && linux
//
// Test not intended to run on PVC
// UNSUPPORTED: cuda || hip || gpu-intel-pvc
//
// TODO: enable when Jira ticket resolved
// XFAIL: gpu
//
// Check that full compilation works:
// RUN: %clangxx -DIMPL_SUBGROUP -fsycl -fno-sycl-device-code-split-esimd -Xclang -fsycl-allow-func-ptr %S/simd8.cpp -o %t.out
// RUN: env IGC_VCSaveStackCallLinkage=1 IGC_VCDirectCallsOnly=1 %GPU_RUN_PLACEHOLDER %t.out
//
// VISALTO enable run
// RUN: env IGC_VISALTO=63 IGC_VCSaveStackCallLinkage=1 IGC_VCDirectCallsOnly=1 %GPU_RUN_PLACEHOLDER %t.out

/*
 * This tests is the same as InvokeSimd/spec/simd_size/simd8.cpp, but
 * compiles without optional subgroup attribute specified and intended to check
 * that compiler is able to choose subgroup size correctly.
 */
