// RUN: %clangxx -fsycl -fsycl-rdc    -fsycl-targets=%sycl_triple -fsycl-device-only -fsyntax-only %s
// RUN: %clangxx -fsycl -fno-sycl-rdc -fsycl-targets=%sycl_triple -fsycl-device-only -fsyntax-only %s

#include <cmath>

int main() {}
