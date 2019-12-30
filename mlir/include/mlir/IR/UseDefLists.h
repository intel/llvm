//===- UseDefLists.h --------------------------------------------*- C++ -*-===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines generic use/def list machinery and manipulation utilities.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_IR_USEDEFLISTS_H
#define MLIR_IR_USEDEFLISTS_H

#include "mlir/IR/Location.h"
#include "llvm/ADT/PointerIntPair.h"
#include "llvm/ADT/iterator_range.h"

namespace mlir {

class Block;
class IROperand;
class Operation;
class Value;
template <typename OperandType> class ValueUseIterator;
template <typename OperandType> class ValueUserIterator;

//===----------------------------------------------------------------------===//
// IRObjectWithUseList
//===----------------------------------------------------------------------===//

class IRObjectWithUseList {
public:
  ~IRObjectWithUseList() {
    assert(use_empty() && "Cannot destroy a value that still has uses!");
  }

  /// Returns true if this value has no uses.
  bool use_empty() const { return firstUse == nullptr; }

  /// Returns true if this value has exactly one use.
  inline bool hasOneUse() const;

  using use_iterator = ValueUseIterator<IROperand>;
  using use_range = iterator_range<use_iterator>;

  inline use_iterator use_begin() const;
  inline use_iterator use_end() const;

  /// Returns a range of all uses, which is useful for iterating over all uses.
  inline use_range getUses() const;

  using user_iterator = ValueUserIterator<IROperand>;
  using user_range = iterator_range<user_iterator>;

  inline user_iterator user_begin() const;
  inline user_iterator user_end() const;

  /// Returns a range of all users.
  inline user_range getUsers() const;

  /// Replace all uses of 'this' value with the new value, updating anything in
  /// the IR that uses 'this' to use the other value instead.  When this returns
  /// there are zero uses of 'this'.
  void replaceAllUsesWith(IRObjectWithUseList *newValue);

  /// Drop all uses of this object from their respective owners.
  void dropAllUses();

protected:
  IRObjectWithUseList() {}

  /// Return the first IROperand that is using this value, for use by custom
  /// use/def iterators.
  IROperand *getFirstUse() { return firstUse; }
  const IROperand *getFirstUse() const { return firstUse; }

private:
  friend class IROperand;
  IROperand *firstUse = nullptr;
};

//===----------------------------------------------------------------------===//
// IROperand
//===----------------------------------------------------------------------===//

/// A reference to a value, suitable for use as an operand of an operation.
class IROperand {
public:
  IROperand(Operation *owner) : owner(owner) {}
  IROperand(Operation *owner, IRObjectWithUseList *value)
      : value(value), owner(owner) {
    insertIntoCurrent();
  }

  /// Return the current value being used by this operand.
  IRObjectWithUseList *get() const { return value; }

  /// Set the current value being used by this operand.
  void set(IRObjectWithUseList *newValue) {
    // It isn't worth optimizing for the case of switching operands on a single
    // value.
    removeFromCurrent();
    value = newValue;
    insertIntoCurrent();
  }

  /// Return the owner of this operand.
  Operation *getOwner() { return owner; }
  Operation *getOwner() const { return owner; }

  /// \brief Remove this use of the operand.
  void drop() {
    removeFromCurrent();
    value = nullptr;
    nextUse = nullptr;
    back = nullptr;
  }

  ~IROperand() { removeFromCurrent(); }

  /// Return the next operand on the use-list of the value we are referring to.
  /// This should generally only be used by the internal implementation details
  /// of the SSA machinery.
  IROperand *getNextOperandUsingThisValue() { return nextUse; }

  /// We support a move constructor so IROperand's can be in vectors, but this
  /// shouldn't be used by general clients.
  IROperand(IROperand &&other) : owner(other.owner) {
    *this = std::move(other);
  }
  IROperand &operator=(IROperand &&other) {
    removeFromCurrent();
    other.removeFromCurrent();
    value = other.value;
    other.value = nullptr;
    other.back = nullptr;
    nextUse = nullptr;
    back = nullptr;
    insertIntoCurrent();
    return *this;
  }

private:
  /// The value used as this operand.  This can be null when in a
  /// "dropAllUses" state.
  IRObjectWithUseList *value = nullptr;

  /// The next operand in the use-chain.
  IROperand *nextUse = nullptr;

  /// This points to the previous link in the use-chain.
  IROperand **back = nullptr;

  /// The operation owner of this operand.
  Operation *const owner;

  /// Operands are not copyable or assignable.
  IROperand(const IROperand &use) = delete;
  IROperand &operator=(const IROperand &use) = delete;

  void removeFromCurrent() {
    if (!back)
      return;
    *back = nextUse;
    if (nextUse)
      nextUse->back = back;
  }

  void insertIntoCurrent() {
    back = &value->firstUse;
    nextUse = value->firstUse;
    if (nextUse)
      nextUse->back = &nextUse;
    value->firstUse = this;
  }
};

//===----------------------------------------------------------------------===//
// BlockOperand
//===----------------------------------------------------------------------===//

/// Terminator operations can have Block operands to represent successors.
class BlockOperand : public IROperand {
public:
  using IROperand::IROperand;

  /// Return the current block being used by this operand.
  Block *get();

  /// Set the current value being used by this operand.
  void set(Block *block);

  /// Return which operand this is in the operand list of the User.
  unsigned getOperandNumber();

private:
  /// The number of OpOperands that correspond with this block operand.
  unsigned numSuccessorOperands = 0;

  /// Allow access to 'numSuccessorOperands'.
  friend Operation;
};

//===----------------------------------------------------------------------===//
// OpOperand
//===----------------------------------------------------------------------===//

/// A reference to a value, suitable for use as an operand of an operation.
class OpOperand : public IROperand {
public:
  OpOperand(Operation *owner) : IROperand(owner) {}
  OpOperand(Operation *owner, Value value);

  /// Return the current value being used by this operand.
  Value get();

  /// Set the current value being used by this operand.
  void set(Value newValue);

  /// Return which operand this is in the operand list of the User.
  unsigned getOperandNumber();
};

//===----------------------------------------------------------------------===//
// ValueUseIterator
//===----------------------------------------------------------------------===//

/// An iterator over all uses of a ValueBase.
template <typename OperandType>
class ValueUseIterator
    : public std::iterator<std::forward_iterator_tag, OperandType> {
public:
  ValueUseIterator() = default;
  explicit ValueUseIterator(OperandType *current) : current(current) {}
  OperandType *operator->() const { return current; }
  OperandType &operator*() const { return *current; }

  Operation *getUser() const { return current->getOwner(); }

  ValueUseIterator &operator++() {
    assert(current && "incrementing past end()!");
    current = (OperandType *)current->getNextOperandUsingThisValue();
    return *this;
  }

  ValueUseIterator operator++(int unused) {
    ValueUseIterator copy = *this;
    ++*this;
    return copy;
  }

  friend bool operator==(ValueUseIterator lhs, ValueUseIterator rhs) {
    return lhs.current == rhs.current;
  }

  friend bool operator!=(ValueUseIterator lhs, ValueUseIterator rhs) {
    return !(lhs == rhs);
  }

private:
  OperandType *current;
};

inline auto IRObjectWithUseList::use_begin() const -> use_iterator {
  return use_iterator(firstUse);
}

inline auto IRObjectWithUseList::use_end() const -> use_iterator {
  return use_iterator(nullptr);
}

inline auto IRObjectWithUseList::getUses() const -> use_range {
  return {use_begin(), use_end()};
}

/// Returns true if this value has exactly one use.
inline bool IRObjectWithUseList::hasOneUse() const {
  return firstUse && firstUse->getNextOperandUsingThisValue() == nullptr;
}

//===----------------------------------------------------------------------===//
// ValueUserIterator
//===----------------------------------------------------------------------===//

/// An iterator over all users of a ValueBase.
template <typename OperandType>
class ValueUserIterator final
    : public llvm::mapped_iterator<ValueUseIterator<OperandType>,
                                   Operation *(*)(OperandType &)> {
  static Operation *unwrap(OperandType &value) { return value.getOwner(); }

public:
  using pointer = Operation *;
  using reference = Operation *;

  /// Initializes the result type iterator to the specified result iterator.
  ValueUserIterator(ValueUseIterator<OperandType> it)
      : llvm::mapped_iterator<ValueUseIterator<OperandType>,
                              Operation *(*)(OperandType &)>(it, &unwrap) {}
  Operation *operator->() { return **this; }
};

inline auto IRObjectWithUseList::user_begin() const -> user_iterator {
  return user_iterator(use_begin());
}

inline auto IRObjectWithUseList::user_end() const -> user_iterator {
  return user_iterator(use_end());
}

inline auto IRObjectWithUseList::getUsers() const -> user_range {
  return {user_begin(), user_end()};
}

} // namespace mlir

#endif
