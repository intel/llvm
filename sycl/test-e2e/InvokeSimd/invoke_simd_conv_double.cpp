// The test checks that invoke_simd implementation performs proper conversions
// on the actual arguments of 'double' type.

// RUN: %{build} -fno-sycl-device-code-split-esimd -Xclang -fsycl-allow-func-ptr -o %t.out
// RUN: env IGC_VCSaveStackCallLinkage=1 IGC_VCDirectCallsOnly=1 %{run} %t.out

#define TEST_DOUBLE_TYPE
#include "invoke_simd_conv.cpp"
