//===--- SYCLMethodOpInterface.h ------------------------------------------===//
//
// MLIR-SYCL is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_SYCL_METHOD_OP_INTERFACE_H_
#define MLIR_SYCL_METHOD_OP_INTERFACE_H_

#include "mlir/Dialect/SYCL/IR/SYCLMethodOpInterface.h.inc"

#include <type_traits>

namespace mlir {
namespace sycl {
template <typename T>
using isSYCLMethod = std::is_base_of<SYCLMethodOpInterface::Trait<T>, T>;
} // namespace sycl
} // namespace mlir

#endif // MLIR_SYCL_METHOD_OP_INTERFACE_H_
