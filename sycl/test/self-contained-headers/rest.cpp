// REQUIRES: linux
// RUN: find %sycl_include/sycl/access -maxdepth 1 -name '*.hpp' -exec %clangxx -fsycl -fsyntax-only -include {} %s ';'
// RUN: find %sycl_include/sycl/backend -maxdepth 1 -name '*.hpp' -exec %clangxx -fsycl -fsyntax-only -include {} %s ';'
// RUN: find %sycl_include/sycl/CL -maxdepth 1 -name '*.hpp' -exec %clangxx -fsycl -fsyntax-only -include {} %s ';'
// RUN: find %sycl_include/sycl/info -maxdepth 1 -name '*.hpp' -exec %clangxx -fsycl -fsyntax-only -include {} %s ';'
// RUN: find %sycl_include/sycl/properties -maxdepth 1 -name '*.hpp' -exec %clangxx -fsycl -fsyntax-only -include {} %s ';'
// RUN: find %sycl_include/sycl/stl_wrappers -maxdepth 1 -name '*.hpp' -exec %clangxx -fsycl -fsyntax-only -include {} %s ';'
// RUN: find %sycl_include/sycl/usm -maxdepth 1 -name '*.hpp' -exec %clangxx -fsycl -fsyntax-only -include {} %s ';'
