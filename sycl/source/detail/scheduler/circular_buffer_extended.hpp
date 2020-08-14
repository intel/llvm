//==-- circular_buffer_extended.hpp - Circular buffer with host accessor ---==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <detail/circular_buffer.hpp>
#include <detail/scheduler/commands.hpp>

#include <cstddef>
#include <list>
#include <map>
#include <unordered_map>
#include <utility>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace detail {

class MemObjRecord;

/// A wrapper for CircularBuffer class along with collection for host accessor's
/// EmptyCommands. This class is introduced to overcome the problem with a lot
/// of host accessors for the same memory object. The problem arises even when
/// all the host accessors are read-only and their ranges intersect somehow.
///
/// Inside the class there is a classical CircularBuffer for generic commands.
/// Also, it contains proper data structures for storing host accessors'
/// EmptyCommands. Cross-referencing data structures are employed for
/// quick enough navigation amongst stored EmptyCommands.
/// IteratorT subclass allows for iterating and dereferencing. Though, it's not
/// guaranteed to work with std::remove as host accessors' commands are stored
/// in a map. Hence, the CircularBufferExtended class provides a viable solution
/// with its own remove method.
class CircularBufferExtended {
public:
  using GenericCommandsT = CircularBuffer<Command *>;
  using HostAccessorCommandListT = std::list<EmptyCommand *>;
  using HostAccessorCommandsT =
      std::unordered_map<MemObjRecord *, HostAccessorCommandListT>;

  using IfGenericIsFullF =
      std::function<void(Command *, MemObjRecord *, GenericCommandsT &)>;
  // Make first command depend on the second
  using AllocateDependencyF =
      std::function<void(Command *, Command *, MemObjRecord *)>;

  template <bool IsConst>
  struct IteratorT;

  using value_type = Command *;
  using pointer = value_type *;
  using const_pointer = const value_type *;
  using reference = value_type &;
  using const_reference = const value_type &;

  using iterator = IteratorT<false>;
  using const_iterator = IteratorT<true>;

  CircularBufferExtended(std::size_t GenericCommandsCapacity,
                         IfGenericIsFullF IfGenericIsFull,
                         AllocateDependencyF AllocateDependency)
      : MGenericCommands{GenericCommandsCapacity},
        MIfGenericIsFull{std::move(IfGenericIsFull)},
        MAllocateDependency{std::move(AllocateDependency)}
  {}

  iterator begin() {
    return iterator{*this, MGenericCommands.begin()};
  }

  iterator end() {
    return iterator{*this, endHostAccessor()};
  }

  const_iterator cbegin() const {
    return const_iterator{*this, MGenericCommands.begin()};
  }

  const_iterator cend() const {
    return const_iterator{*this, endHostAccessor()};
  }

  void push_back(value_type Cmd, MemObjRecord *Record);

  /// Replacement for std::remove with subsequent call to erase(newEnd, end()).
  /// This function is introduced here due to complexity of iterator.
  /// \returns number of removed elements
  size_t remove(value_type Cmd);

  std::vector<value_type> toVector() const;

  size_t genericCommandsCapacity() const {
    return MGenericCommands.capacity();
  };

  const GenericCommandsT &getGenericCommands() const {
    return MGenericCommands;
  }

private:
  using HostAccessorCommandSingleXRefT = HostAccessorCommandListT::iterator;
  // Unordered map would suit here better. Though, its iterators are invalidated
  // upon rehash. Ordered map's iterators, here however, are not invalidated
  // upon rebalance operation.
  using HostAccessorCommandXRefT =
      std::map<EmptyCommand *, HostAccessorCommandSingleXRefT>;
  template <bool IsConst>
  struct HostAccessorIt;


  GenericCommandsT MGenericCommands;
  // main storage for host accessor comands:
  // mem object record -> list of EmptyCommands
  HostAccessorCommandsT MHostAccessorCommands;
  // cross-reference map: EmptyCommand -> iterator in list in main storage
  HostAccessorCommandXRefT MHostAccessorCommandsXRef;

  IfGenericIsFullF MIfGenericIsFull;
  AllocateDependencyF MAllocateDependency;

  void addGenericCommand(value_type Cmd, MemObjRecord *Record);
  void addHostAccessorCommand(EmptyCommand *Cmd, MemObjRecord *Record);

  // inserts a command to the end of list for its mem object
  void insertHostAccessorCommand(EmptyCommand *Cmd);
  void eraseHostAccessorCommand(EmptyCommand *Cmd);

  HostAccessorIt<false> endHostAccessor() {
    return HostAccessorIt<false>(
        &MHostAccessorCommands, MHostAccessorCommands.end());
  }

  HostAccessorIt<true> endHostAccessor() const {
    return HostAccessorIt<true>(
        &MHostAccessorCommands, MHostAccessorCommands.end());
  }

  // for access to struct ConstRef.
  friend struct IteratorT<true>;
  friend struct IteratorT<false>;

  template <bool IsConst, typename T>
  struct ConstRef;

  template<typename T> struct ConstRef<true, T> {
    using type = const T &;
  };

  template<typename T> struct ConstRef<false, T> {
    using type = T &;
  };


  template <bool IsConst, typename T>
  struct ConstPtr;

  template<typename T> struct ConstPtr<true, T> {
    using type = const T *;
  };

  template<typename T> struct ConstPtr<false, T> {
    using type = T *;
  };


  template <bool IsConst, typename T>
  struct ConstIterator;

  template<typename T> struct ConstIterator<true, T> {
    using type = typename T::const_iterator;
  };

  template<typename T> struct ConstIterator<false, T> {
    using type = typename T::iterator;
  };


  template <bool IsConst>
  struct HostAccessorIt {
    using HostT = typename ConstPtr<IsConst, HostAccessorCommandsT>::type;
    using MapItT = typename ConstIterator<IsConst, HostAccessorCommandsT>::type;
    using ValueItT =
        typename ConstIterator<IsConst,
                               HostAccessorCommandsT::mapped_type>::type;

    HostT MHost;
    MapItT MMapIt;
    ValueItT MValueIt;

    HostAccessorIt() : MHost{nullptr} {}

    HostAccessorIt(HostT Host, MapItT MapIt) : MHost{Host}, MMapIt{MapIt} {
      if (MMapIt != MHost->end())
        MValueIt = MMapIt->second.begin();
    }

    HostAccessorIt(const HostAccessorIt<IsConst> &Other)
      : MHost{Other.MHost}, MMapIt{Other.MMapIt}, MValueIt{Other.MValueIt} {}

    HostAccessorIt(HostAccessorIt<IsConst> &&Other)
      : MHost{Other.MHost}, MMapIt{Other.MMapIt}, MValueIt{Other.MValueIt} {}

    value_type operator*() {
      assert(MHost);
      return *MValueIt;
    }

    value_type operator*() const {
      assert(MHost);
      return *MValueIt;
    }

    // pre-increment
    HostAccessorIt<IsConst> &operator++() {
      assert(MHost);
      increment();
      return *this;
    }

    // post-increment
    HostAccessorIt<IsConst> operator++(int) {
      assert(MHost);
      HostAccessorIt<IsConst> Result = *this;
      increment();

      return Result;
    }

    bool operator==(const HostAccessorIt<IsConst> &Rhs) const {
      return MHost && MHost == Rhs.MHost && MMapIt == Rhs.MMapIt &&
          (MMapIt == MHost->end() || MValueIt == Rhs.MValueIt);
    }

    bool operator!=(const HostAccessorIt<IsConst> &Rhs) const {
      return !MHost || MHost != Rhs.MHost || MMapIt != Rhs.MMapIt ||
          (MMapIt != MHost->end() && Rhs.MMapIt != Rhs.MHost->end() && 
           MValueIt != Rhs.MValueIt);
    }

    HostAccessorIt<IsConst> &operator=(const HostAccessorIt<IsConst> &Other) {
      MHost = Other.MHost;
      MMapIt = Other.MMapIt;

      if (MMapIt != MHost->end())
        MValueIt = Other.MValueIt;

      return *this;
    }

    HostAccessorIt<IsConst> &operator=(HostAccessorIt<IsConst> &&Other) {
      MHost = Other.MHost;
      MMapIt = std::move(Other.MMapIt);

      if (MMapIt != MHost->end())
        MValueIt = std::move(Other.MValueIt);

      return *this;
    }

  private:
    // return true if in the very end or the increment happened
    inline bool incrementMapItIfNeeded() {
      if (MValueIt == MMapIt->second.end()) {
        ++MMapIt;

        if (MMapIt != MHost->end())
          MValueIt = MMapIt->second.begin();

        return true;
      }

      return false;
    }

    void increment() {
      if (MMapIt == MHost->end())
        return;

      if (incrementMapItIfNeeded())
        return;

      ++MValueIt;

      incrementMapItIfNeeded();
    }
  };

public:
  // iterate over generic commands in the first place and over host accessors
  // later on
  template <bool IsConst>
  struct IteratorT {
    using HostT = typename ConstRef<IsConst, CircularBufferExtended>::type;
    using GCItT = typename ConstIterator<IsConst, GenericCommandsT>::type;
    using HACItT = HostAccessorIt<IsConst>;

    HostT MHost;
    GCItT MGCIt;
    HACItT MHACIt;

    bool MGenericIsActive;

    IteratorT(HostT Host, GCItT GCIt)
        : MHost(Host), MGCIt{std::move(GCIt)}, MGenericIsActive{true} {}

    IteratorT(HostT Host, HACItT HACIt)
        : MHost(Host), MHACIt{std::move(HACIt)}, MGenericIsActive{false} {}

    IteratorT(const IteratorT<IsConst> &Other)
        : MHost{Other.MHost}, MGCIt(Other.MGCIt), MHACIt(Other.MHACIt),
          MGenericIsActive(Other.MGenericIsActive) {}

    IteratorT(IteratorT<IsConst> &&Other)
        : MHost{Other.MHost}, MGCIt(std::move(Other.MGCIt)),
          MHACIt(std::move(Other.MHACIt)),
          MGenericIsActive(Other.MGenericIsActive) {}

    bool operator==(const IteratorT<IsConst> &Rhs) const {
      return &MHost == &Rhs.MHost && MGenericIsActive == Rhs.MGenericIsActive &&
          ((MGenericIsActive && MGCIt == Rhs.MGCIt) ||
           (!MGenericIsActive && MHACIt == Rhs.MHACIt));
    }

    bool operator!=(const IteratorT<IsConst> &Rhs) const {
      return &MHost != &Rhs.MHost || MGenericIsActive != Rhs.MGenericIsActive ||
          ((MGenericIsActive && MGCIt != Rhs.MGCIt) ||
           (!MGenericIsActive && MHACIt != Rhs.MHACIt));
    }

    // pre-increment
    IteratorT<IsConst> &operator++() {
      increment();
      return *this;
    }

    // post-increment
    IteratorT<IsConst> operator++(int) {
      IteratorT<IsConst> Other(*this);
      increment();

      return Other;
    }

    value_type operator*() const {
      if (MGenericIsActive && MGCIt != MHost.MGenericCommands.end())
        return *MGCIt;

      if (!MGenericIsActive && MHACIt != MHost.endHostAccessor())
        return *MHACIt;

      assert(false);

      return nullptr;
    }

  private:
    void increment() {
      if (MGenericIsActive) {
        if (MGCIt == MHost.MGenericCommands.end()) {
          MGenericIsActive = false;
          MHACIt = HACItT{&MHost.MHostAccessorCommands,
                          MHost.MHostAccessorCommands.begin()};
          return;
        }

        ++MGCIt;
        return;
      }

      assert(MGCIt == MHost.MGenericCommands.end());

      if (MHACIt == MHost.endHostAccessor())
        return;

      ++MHACIt;
    }
  };
};

} // namespace detail
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
