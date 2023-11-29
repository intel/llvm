// GPU driver had an error in handling of SLM aligned block_loads/stores,
// which has been fixed only in "1.3.26816", and in win/opencl version going
// _after_ 101.4575.
// REQUIRES-INTEL-DRIVER: lin: 26816, win: 101.4576
//
// RUN: %clangxx -DIMPL_SUBGROUP -fsycl -fno-sycl-device-code-split-esimd -Xclang -fsycl-allow-func-ptr %S/../slm_load_store.cpp -o %t.out
// RUN: env IGC_VCSaveStackCallLinkage=1 IGC_VCDirectCallsOnly=1 %{run} %t.out
//
// VISALTO enable run
// RUN: env IGC_VISALTO=63 IGC_VCSaveStackCallLinkage=1 IGC_VCDirectCallsOnly=1 %{run} %t.out

/*
 * This tests is the same as InvokeSimd/Regression/slm_load_store.cpp, but
 * compiles without optional subgroup attribute specified and intended to check
 * that compiler is able to choose subgroup size correctly.
 */
