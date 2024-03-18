// RUN: %{build} -o %t.out
// RUN: echo 1  | %{run} %t.out
// RUN: echo 10 | %{run} %t.out
// RUN: echo 20 | %{run} %t.out
// RUN: echo 30 | %{run} %t.out
// UNSUPPORTED: cuda || hip

// Simple test filling a SYCL private alloca and copying it back to an output
// accessor using a decorated multi_ptr.

#include "Inputs/private_alloca_test.hpp"

constexpr sycl::specialization_id<int> size(10);

int main() { test<float, size, sycl::access::decorated::yes>(); }
