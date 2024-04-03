// REQUIRES: linux
// RUN: find %sycl_include/sycl/access -name '*.hpp' -exec %clangxx -fsycl -fsyntax-only -include {} %s ';'
// RUN: find %sycl_include/sycl/backend -name '*.hpp' -exec %clangxx -fsycl -fsyntax-only -include {} %s ';'
// RUN: find %sycl_include/sycl/CL -name '*.hpp' -exec %clangxx -fsycl -fsyntax-only -include {} %s ';'
// RUN: find %sycl_include/sycl/info -name '*.hpp' -exec %clangxx -fsycl -fsyntax-only -include {} %s ';'
// RUN: find %sycl_include/sycl/properties -name '*.hpp' -exec %clangxx -fsycl -fsyntax-only -include {} %s ';'
// RUN: find %sycl_include/sycl/stl_wrappers -name '*.hpp' -exec %clangxx -fsycl -fsyntax-only -include {} %s ';'
// RUN: find %sycl_include/sycl/usm -name '*.hpp' -exec %clangxx -fsycl -fsyntax-only -include {} %s ';'
// RUN: find %sycl_include/sycl -mindepth 1 -maxdepth 1 -type d | wc -l | FileCheck %s
// If the check below failed, it means that list of directories within
// %sycl_include/sycl has changed and RUN lines above should be updated to make
// sure that any new folders (if they've been added) are also checked by the
// test
// CHECK: 9
