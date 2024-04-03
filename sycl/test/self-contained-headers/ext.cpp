// REQUIRES: linux
// RUN: find %sycl_include/sycl/ext -maxdepth 1 -name '*.hpp' -exec %clangxx -fsycl -fsyntax-only -include {} %s ';'
