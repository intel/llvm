// REQUIRES: linux
// RUN: find %sycl_include/sycl/ext/oneapi -name '*.hpp' -exec %clangxx -fsycl -fsyntax-only -include {} %s ';'
