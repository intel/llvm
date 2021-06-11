#pragma once

struct AssertHappened {
  int Flag = 0;
};

#ifndef __SYCL_GLOBAL_VAR__
#define __SYCL_GLOBAL_VAR__ __attribute__((sycl_global_var))
#endif

// declaration
extern __SYCL_GLOBAL_VAR__ __SYCL_GLOBAL__ const AssertHappened
    __SYCL_AssertHappenedMem;

