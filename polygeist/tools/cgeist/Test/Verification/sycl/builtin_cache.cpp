// RUN: clang++  -fsycl -fsycl-device-only -O0 -w -emit-mlir -o - %s | FileCheck %s

#include <sycl/sycl.hpp>
using namespace sycl;

// CHECK-LABEL:     func.func @_Z6callee38__spirv_SampledImage__image1d_array_ro
// CHECK-SAME:          (%[[VAL_151:.*]]: !llvm.target<"spirv.SampledImage", !llvm.void, 0, 0, 1, 0, 0, 0, 0>) 
// CHECK-NEXT:        return
// CHECK-NEXT:      }

// CHECK-LABEL:     func.func @_Z6caller38__spirv_SampledImage__image1d_array_ro
// CHECK-SAME:          (%[[VAL_152:.*]]: !llvm.target<"spirv.SampledImage", !llvm.void, 0, 0, 1, 0, 0, 0, 0>) 
// CHECK-NEXT:        call @_Z6callee38__spirv_SampledImage__image1d_array_ro(%[[VAL_152]]) : (!llvm.target<"spirv.SampledImage", !llvm.void, 0, 0, 1, 0, 0, 0, 0>) -> ()
// CHECK-NEXT:        return
// CHECK-NEXT:      }

SYCL_EXTERNAL void callee(__ocl_sampled_image1d_array_ro_t var) {}
SYCL_EXTERNAL void caller(__ocl_sampled_image1d_array_ro_t var) {
  callee(var);
}
