//===--- SYCLAttributes.h -------------------------------------------------===//
//
// MLIR-SYCL is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_SYCL_IR_SYCLATTRIBUTES_H
#define MLIR_DIALECT_SYCL_IR_SYCLATTRIBUTES_H

#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"

#include "mlir/Dialect/SYCL/IR/SYCLEnums.h.inc"

#define GET_ATTRDEF_CLASSES
#include "mlir/Dialect/SYCL/IR/SYCLAttributes.h.inc"
#undef GET_ATTRDEF_CLASSES

namespace mlir {
namespace sycl {

// TODO: Replace with TargetAttr -> AccessAddrSpaceAttr mapping
inline unsigned targetToAddressSpace(Target target) {
  switch (target) {
  case Target::ConstantBuffer:
  case Target::GlobalBuffer:
    return 1;
  case Target::Local:
    return 3;
  default:
    llvm_unreachable("Invalid Target for an accessor");
  }
}

class ReqdWorkGroupSize {
public:
  ReqdWorkGroupSize(ArrayRef<gpu::GPUFuncOp> kernels) {
    init(kernels.front(), wgSizes);
    for (gpu::GPUFuncOp kernel : drop_begin(kernels)) {
      SmallVector<unsigned> wgSizes2;
      init(kernel, wgSizes2);
      assert(wgSizes == wgSizes2 && "Expecting same reqd_work_group_size");
    }
  }
  bool empty() const { return wgSizes.empty(); }
  unsigned operator[](unsigned dim) const {
    assert(dim < wgSizes.size() && "Expecting valid dim");
    return wgSizes[wgSizes.size() - 1 - dim];
  }

private:
  /// Populate \p wgSizes from kernel \p kernel.
  static void init(gpu::GPUFuncOp kernel, SmallVector<unsigned> &wgSizes) {
    if (ArrayAttr wgSizesAttr = dyn_cast_or_null<ArrayAttr>(
            kernel->getAttr("reqd_work_group_size"))) {
      for (IntegerAttr wgSizeAttr : wgSizesAttr.getAsRange<IntegerAttr>())
        wgSizes.push_back(wgSizeAttr.getInt());
      assert(!wgSizes.empty() && wgSizes.size() <= 3 &&
             "Expecting non-empty wgSizes of size less than 3");
    }
  }

  SmallVector<unsigned> wgSizes;
};

} // namespace sycl
} // namespace mlir

#endif // MLIR_DIALECT_SYCL_IR_SYCLATTRIBUTES_H
