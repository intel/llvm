// RUN: %clangxx -fsycl-device-only -S -emit-llvm -o - %s | FileCheck %s

#include <sycl/detail/core.hpp>

SYCL_EXTERNAL size_t GlobalInvocationId(int dim) {
  // CHECK: call spir_func {{.*}} i64 @_Z33__spirv_BuiltInGlobalInvocationIdi({{.*}} [[ATTR:#[0-9]+]]
  return __spirv_BuiltInGlobalInvocationId(dim);
}

SYCL_EXTERNAL size_t GlobalSize(int dim) {
  // CHECK: call spir_func {{.*}} i64 @_Z25__spirv_BuiltInGlobalSizei({{.*}} [[ATTR]]
  return __spirv_BuiltInGlobalSize(dim);
}

SYCL_EXTERNAL size_t GlobalOffset(int dim) {
  // CHECK: call spir_func {{.*}} i64 @_Z27__spirv_BuiltInGlobalOffseti({{.*}} [[ATTR]]
  return __spirv_BuiltInGlobalOffset(dim);
}

SYCL_EXTERNAL size_t NumWorkgroups(int dim) {
  // CHECK: call spir_func {{.*}} i64 @_Z28__spirv_BuiltInNumWorkgroupsi({{.*}} [[ATTR]]
  return __spirv_BuiltInNumWorkgroups(dim);
}

SYCL_EXTERNAL size_t WorkgroupSize(int dim) {
  // CHECK: call spir_func {{.*}} i64 @_Z28__spirv_BuiltInWorkgroupSizei({{.*}} [[ATTR]]
  return __spirv_BuiltInWorkgroupSize(dim);
}

SYCL_EXTERNAL size_t WorkgroupId(int dim) {
  // CHECK: call spir_func {{.*}} i64 @_Z26__spirv_BuiltInWorkgroupIdi({{.*}} [[ATTR]]
  return __spirv_BuiltInWorkgroupId(dim);
}

SYCL_EXTERNAL size_t LocalInvocationId(int dim) {
  // CHECK: call spir_func {{.*}} i64 @_Z32__spirv_BuiltInLocalInvocationIdi({{.*}} [[ATTR]]
  return __spirv_BuiltInLocalInvocationId(dim);
}

SYCL_EXTERNAL uint32_t SubgroupSize() {
  // CHECK: call spir_func {{.*}} i32 @_Z27__spirv_BuiltInSubgroupSizev() [[ATTR]]
  return __spirv_BuiltInSubgroupSize();
}

SYCL_EXTERNAL uint32_t SubgroupMaxSize() {
  // CHECK: call spir_func {{.*}} i32 @_Z30__spirv_BuiltInSubgroupMaxSizev() [[ATTR]]
  return __spirv_BuiltInSubgroupMaxSize();
}

SYCL_EXTERNAL uint32_t NumSubgroups() {
  // CHECK: call spir_func {{.*}} i32 @_Z27__spirv_BuiltInNumSubgroupsv() [[ATTR]]
  return __spirv_BuiltInNumSubgroups();
}

SYCL_EXTERNAL uint32_t SubgroupId() {
  // CHECK: call spir_func {{.*}} i32 @_Z25__spirv_BuiltInSubgroupIdv() [[ATTR]]
  return __spirv_BuiltInSubgroupId();
}

SYCL_EXTERNAL uint32_t SubgroupLocalInvocationId() {
  // CHECK: call spir_func {{.*}} i32 @_Z40__spirv_BuiltInSubgroupLocalInvocationIdv() [[ATTR]]
  return __spirv_BuiltInSubgroupLocalInvocationId();
}

// CHECK: attributes [[ATTR]] = {{.*}} memory(none)
