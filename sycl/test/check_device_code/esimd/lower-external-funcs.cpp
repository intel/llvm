// RUN: %clangxx -fsycl -fsycl-device-only -S -emit-llvm -x c++ %s -o %t
// RUN: sycl-post-link -properties -split-esimd -lower-esimd -O2 -S %t -o %t.table
// RUN: FileCheck %s -input-file=%t_esimd_0.ll

// This test checks that unreferenced SYCL_EXTERNAL functions are not dropped
// from the module and go through sycl-post-link. This test also checks that
// ESIMD lowering happens for such functions as well.

#include <sycl/ext/intel/esimd.hpp>
#include <sycl/sycl.hpp>

constexpr unsigned VL = 8;
using namespace sycl;
using namespace sycl::ext::intel::esimd;
extern "C" SYCL_EXTERNAL SYCL_ESIMD_FUNCTION void foo() { barrier(); }

// CHECK: define dso_local spir_func void @foo
// CHECK: call void @llvm.genx.fence(i8 33)
// CHECK: call void @llvm.genx.barrier()
// CHECK:   ret void
