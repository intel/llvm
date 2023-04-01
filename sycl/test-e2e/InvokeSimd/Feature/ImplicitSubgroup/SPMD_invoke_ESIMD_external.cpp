// TODO: enable on Windows once driver is ready
// REQUIRES: gpu && linux
// UNSUPPORTED: cuda || hip
//
// TODO: enable when Jira ticket resolved
// XFAIL: gpu
//
// RUN: %clangxx -DIMPL_SUBGROUP -fsycl -fno-sycl-device-code-split-esimd -Xclang -fsycl-allow-func-ptr %S/../SPMD_invoke_ESIMD_external.cpp -o %t.out
// RUN: env IGC_VCSaveStackCallLinkage=1 IGC_VCDirectCallsOnly=1 %GPU_RUN_PLACEHOLDER %t.out
//
// VISALTO enable run
// RUN: env IGC_VCSaveStackCallLinkage=1 IGC_VCDirectCallsOnly=1 %GPU_RUN_PLACEHOLDER %t.out

/*
 * This tests is the same as InvokeSimd/feature/SPMD_invoke_ESIMD_external.cpp,
 * but compiles without optional subgroup attribute specified and intended to
 * check that compiler is able to choose subgroup size correctly.
 */
