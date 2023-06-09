//===--- SYCLOpsAttributes.cpp --------------------------------------------===//
//
// MLIR-SYCL is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/SYCL/IR/SYCLAttributes.h"

#include "mlir/Dialect/SYCL/IR/SYCLAttributes.cpp.inc"
#include "mlir/Dialect/SYCL/IR/SYCLEnums.cpp.inc"

using namespace mlir;
using namespace mlir::sycl;

ReqdWorkGroupSize::ReqdWorkGroupSize(ArrayRef<gpu::GPUFuncOp> kernels) {
  init(kernels.front(), wgSizes);
  for (gpu::GPUFuncOp kernel : drop_begin(kernels)) {
    SmallVector<unsigned> wgSizes2;
    init(kernel, wgSizes2);
    assert(wgSizes == wgSizes2 && "Expecting same reqd_work_group_size");
  }
}

bool ReqdWorkGroupSize::empty() const { return wgSizes.empty(); }

unsigned ReqdWorkGroupSize::size() const { return wgSizes.size(); }

unsigned ReqdWorkGroupSize::operator[](unsigned dim) const {
  assert(dim < wgSizes.size() && "Expecting valid dim");
  return wgSizes[wgSizes.size() - 1 - dim];
}

void ReqdWorkGroupSize::init(gpu::GPUFuncOp kernel,
                             SmallVectorImpl<unsigned> &wgSizes) {
  assert(wgSizes.empty() && "Expecting empty wgSizes");
  if (ArrayAttr wgSizesAttr = dyn_cast_or_null<ArrayAttr>(
          kernel->getAttr("reqd_work_group_size"))) {
    for (IntegerAttr wgSizeAttr : wgSizesAttr.getAsRange<IntegerAttr>())
      wgSizes.push_back(wgSizeAttr.getInt());
    assert(!wgSizes.empty() && wgSizes.size() <= 3 &&
           "Expecting non-empty wgSizes of size less than 3");
  }
}
