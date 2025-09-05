// RUN: %clangxx -fsycl %s -o %t.out -I %llvm_main_include_dir
// RUN: %t.out
//
#include <map>

#include <llvm/SYCLLowerIR/DeviceConfigFile.hpp>
#include <sycl/sycl.hpp>

#define __SYCL_ASPECT_DEPRECATED_ALIAS(ASPECT, ID, MESSAGE)                    \
  __SYCL_ASPECT_DEPRECATED(ASPECT, ID, MESSAGE)

int main() {
  auto testAspects = DeviceConfigFile::TargetTable.find("__TestAspectList");
  assert(testAspects != DeviceConfigFile::TargetTable.end());
  auto aspectsList = testAspects->second.aspects;

#define __SYCL_ASPECT(ASPECT, ASPECT_VAL)                                      \
  std::string_view s##ASPECT(#ASPECT);                                         \
  assert(std::find(aspectsList.begin(), aspectsList.end(), s##ASPECT) !=       \
         aspectsList.end());

#include <sycl/info/aspects.def>

#undef __SYCL_ASPECT

  auto testDeprecatedAspects =
      DeviceConfigFile::TargetTable.find("__TestDeprecatedAspectList");
  assert(testDeprecatedAspects != DeviceConfigFile::TargetTable.end());
  auto deprecatedAspectsList = testDeprecatedAspects->second.aspects;
#define __SYCL_ASPECT_DEPRECATED(ASPECT, ASPECT_VAL, MSG)                      \
  std::string_view s##ASPECT(#ASPECT);                                         \
  assert(std::find(deprecatedAspectsList.begin(), deprecatedAspectsList.end(), \
                   s##ASPECT) != deprecatedAspectsList.end());

#include <sycl/info/aspects_deprecated.def>

#undef __SYCL_ASPECT_DEPRECATED
}

#undef __SYCL_ASPECT_DEPRECATED_ALIAS
