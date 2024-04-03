// REQUIRES: linux
// RUN: find %sycl_include/sycl/detail -name '*.hpp' -exec %clangxx -fsycl -fsyntax-only -include {} %s ';'
