// This test checks if DeviceConfigFile.td and aspects.def are in sync.
// RUN: %clangxx -fsycl %s -o %t.out -I %llvm_main_include_dir
// RUN: %t.out

#include <iostream>
#include <map>

#include <llvm/ADT/StringRef.h>
#include <llvm/SYCLLowerIR/DeviceConfigFile.hpp>
#include <sycl/sycl.hpp>

void check(const char *aspect_name, int aspect_val, int &n_fail) {
  const auto &aspectTable = DeviceConfigFile::AspectTable;
  auto res = aspectTable.find(aspect_name);
  if (res == aspectTable.end()) {
    std::cout << "Aspect " << aspect_name
              << " was not found in the device config file!\n";
    ++n_fail;
    return;
  }
  if (res->second != aspect_val) {
    std::cout << "Aspect " << aspect_name << " has value " << res->second
              << " in the device config file but has value " << aspect_val
              << " in aspects.def!\n";
    ++n_fail;
    return;
  }
}

int main() {
  int n_fail = 0;

#define __SYCL_ASPECT(ASPECT, ASPECT_VAL) check(ASPECT, ASPECT_VAL, n_fail)
#define __SYCL_ASPECT_DEPRECATED(ASPECT, ASPECT_VAL, MSG)                      \
  __SYCL_ASPECT(ASPECT, ASPECT_VAL)
#define __SYCL_ASPECT_DEPRECATED_ALIAS(ASPECT, ID, MESSAGE)
#include <sycl/info/aspects.def>
#include <sycl/info/aspects_deprecated.def>
#undef __SYCL_ASPECT_DEPRECATED_ALIAS
#undef __SYCL_ASPECT_DEPRECATED
#undef __SYCL_ASPECT

  if (n_fail > 0) {
    std::cout << "Errors detected, DeviceConfigFile.td and aspects.def are out "
                 "of sync!\n";
  }
  return n_fail;
}