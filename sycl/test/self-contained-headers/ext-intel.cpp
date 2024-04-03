// REQUIRES: linux
// RUN: find %sycl_include/sycl/ext/intel -name '*.hpp' -exec %clangxx -fsycl -fsyntax-only -include {} %s ';'
