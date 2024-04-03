// REQUIRES: linux
// RUN: find %sycl_include/sycl/ext/codeplay -name '*.hpp' -exec %clangxx -fsycl -fsyntax-only -include {} %s ';'
