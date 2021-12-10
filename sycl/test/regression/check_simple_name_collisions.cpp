// RUN: %clangxx -fsycl -fsyntax-only -Xclang -verify %s
// expected-no-diagnostics

#define NONE
#define KERNEL
#define COPY_ACC_TO_PTR
#define COPY_PTR_TO_ACC
#define COPY_ACC_TO_ACC
#define BARRIER
#define BARRIER_WAITLIST
#define FILL
#define UPDATE_HOST
#define RUN_ON_HOST_INTEL
#define COPY_USM
#define FILL_USM
#define PREFETCH_USM
#define CODEPLAY_INTEROP_TASK
#define CODEPLAY_HOST_TASK
#define ADVISE_USM
#define BUFFER
#define IMAGE 
#define UNDEFINED

#include <CL/sycl.hpp>

int main() { return 0; }
