// REQUIRES: linux
// RUN: find %sycl_include/sycl/detail -maxdepth 1 -name '*.hpp' -exec %clangxx -fsycl -fsyntax-only -include {} %s ';'
