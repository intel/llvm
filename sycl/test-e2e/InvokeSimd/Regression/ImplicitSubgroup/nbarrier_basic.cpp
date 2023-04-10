// TODO: enable on Windows once driver is ready
// NOTE: named barrier supported only since PVC
// REQUIRES: gpu-intel-pvc && linux
// UNSUPPORTED: cuda || hip
//
// TODO: enable when Jira issue resolved, currently fail with VISALTO enable
// XFAIL: gpu-intel-pvc
//
// RUN: %clangxx -DIMPL_SUBGROUP -fsycl -fno-sycl-device-code-split-esimd -Xclang -fsycl-allow-func-ptr %S/../nbarrier_basic.cpp -o %t.out
// RUN: env IGC_VCSaveStackCallLinkage=1 IGC_VCDirectCallsOnly=1 %GPU_RUN_PLACEHOLDER %t.out
//
// VISALTO enable run
// RUN: env IGC_VISALTO=63 IGC_VCSaveStackCallLinkage=1 IGC_VCDirectCallsOnly=1 %GPU_RUN_PLACEHOLDER %t.out

/*
 * This tests is the same as InvokeSimd/Regression/slm_load_store.cpp, but
 * compiles without optional subgroup attribute specified and intended to check
 * that compiler is able to choose subgroup size correctly.
 */
