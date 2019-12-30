//===- Value.cpp - MLIR Value Classes -------------------------------------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Value.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Operation.h"
using namespace mlir;

/// If this value is the result of an Operation, return the operation that
/// defines it.
Operation *Value::getDefiningOp() const {
  if (auto result = dyn_cast<OpResult>())
    return result->getOwner();
  return nullptr;
}

Location Value::getLoc() {
  if (auto *op = getDefiningOp())
    return op->getLoc();
  return UnknownLoc::get(getContext());
}

/// Return the Region in which this Value is defined.
Region *Value::getParentRegion() {
  if (auto *op = getDefiningOp())
    return op->getParentRegion();
  return cast<BlockArgument>()->getOwner()->getParent();
}

//===----------------------------------------------------------------------===//
// BlockOperand
//===----------------------------------------------------------------------===//

/// Return the current block being used by this operand.
Block *BlockOperand::get() { return static_cast<Block *>(IROperand::get()); }

/// Set the current value being used by this operand.
void BlockOperand::set(Block *block) { IROperand::set(block); }

/// Return which operand this is in the operand list.
unsigned BlockOperand::getOperandNumber() {
  return this - &getOwner()->getBlockOperands()[0];
}

//===----------------------------------------------------------------------===//
// OpOperand
//===----------------------------------------------------------------------===//

OpOperand::OpOperand(Operation *owner, Value value)
    : IROperand(owner, value.impl) {}

/// Return the current value being used by this operand.
Value OpOperand::get() { return (detail::ValueImpl *)IROperand::get(); }

/// Set the current value being used by this operand.
void OpOperand::set(Value newValue) { IROperand::set(newValue.impl); }

/// Return which operand this is in the operand list.
unsigned OpOperand::getOperandNumber() {
  return this - &getOwner()->getOpOperands()[0];
}

//===----------------------------------------------------------------------===//
// IRObjectWithUseList implementation.
//===----------------------------------------------------------------------===//

/// Replace all uses of 'this' value with the new value, updating anything in
/// the IR that uses 'this' to use the other value instead.  When this returns
/// there are zero uses of 'this'.
void IRObjectWithUseList::replaceAllUsesWith(IRObjectWithUseList *newValue) {
  assert(this != newValue && "cannot RAUW a value with itself");
  while (!use_empty()) {
    use_begin()->set(newValue);
  }
}

/// Drop all uses of this object from their respective owners.
void IRObjectWithUseList::dropAllUses() {
  while (!use_empty()) {
    use_begin()->drop();
  }
}
