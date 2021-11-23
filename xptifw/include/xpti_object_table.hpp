//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
#pragma once

#include "spin_lock.hpp"

#include <array>
#include <functional>
#include <mutex>
#include <shared_mutex>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <variant>
#include <vector>

namespace xpti {
/// A thread-safe caching table for arbitrary objects in XPTI framework.
///
/// This class enables registration of arbitrary objects within XPTI framework
/// to allow passing them as metadata. If an object being added already exists,
/// an existing ID will be returned.
///
/// @tparam KeyType is the data type of the returned key.
/// @tparam SmallSize is the size of an object, that will fit within table
/// without allocation of additional memory (i.e. small size optimization).
/// The default value of 224 is carefully chosen for Value struct to take
/// 4 cache lines on x86.
template <typename KeyType = uint64_t, size_t SmallSize = 224>
class ObjectTable {
public:
  using HashFunction = std::function<uint64_t(std::string_view)>;

  constexpr static auto DefaultHash = [](std::string_view Data) -> uint64_t {
    // This is an implementation of FNV hash function.
    constexpr uint64_t Prime = 1099511628211;
    uint64_t Hash = 14695981039346656037UL;

    for (char C : Data) {
      Hash *= Prime;
      Hash ^= C;
    }

    return Hash;
  };

  /// Constructs empty object table.
  ///
  /// @param InitialSize is the number of pre-allocated values in the table.
  /// @param HashFunc is a callable object, that given raw bytes returns some
  /// hash value. This is only used upon insertion to quickly scan through the
  /// table and return an existing ID, if any.
  ObjectTable(size_t InitialSize = 4096,
              const HashFunction &HashFunc = DefaultHash)
      : MHashFunction(HashFunc) {
    MValues.reserve(InitialSize);
  }

  /// Inserts an object into a table or retrieves an existing object ID.
  KeyType insert(std::string_view Data, uint8_t Type) {
    uint64_t Hash = MHashFunction(Data);

    SharedLock Lock(MMutex);
    // Check if this data object already exists
    if (MCache.count(Hash) > 0) {
      for (KeyType Key : MCache[Hash]) {
        // Avoid collisions
        if (getValue(MValues[Key]).first == Data) {
#ifdef XPTI_STATISTICS
          MCacheHits++;
#endif
          return Key;
        }
      }
    }

    const Value &V = makeValue(Data, Hash, Type);

    Lock.upgrade_to_writer();

    MValues.push_back(std::move(V));
    KeyType Key = MValues.size() - 1;
    MCache[MValues.back().MHash].push_back(Key);

    return Key;
  }

  /// @returns a pair of raw data bytes and registered data type.
  std::pair<std::string_view, uint8_t> lookup(KeyType Key) {
    SharedLock Lock(MMutex);

    return getValue(MValues[Key]);
  }

#ifdef XPTI_STATISTICS
  size_t getCacheHits() const noexcept { return MCacheHits; }
  size_t getSmallObjectsCount() const noexcept { return MSmallObjects; }
  size_t getLargeObjectsCount() const noexcept { return MLargeObjects; }
#endif

private:
  using Item = std::variant<std::array<char, SmallSize>, std::vector<char>>;

  struct Value {
    uint64_t MSize = 0;
    uint64_t MHash = 0;
    Item MItem;
    uint8_t MType = 0;
  };

  Value makeValue(std::string_view Data, uint64_t Hash, uint8_t Type) {
    Value V;
    V.MSize = Data.size();
    V.MHash = Hash;
    V.MType = Type;

    char *Dest = nullptr;

    if (V.MSize > SmallSize) {
      V.MItem = std::vector<char>(V.MSize, 0);
      Dest = std::get<1>(V.MItem).data();
#ifdef XPTI_STATISTICS
      MLargeObjects++;
#endif
    } else {
      V.MItem = std::array<char, SmallSize>();
      Dest = std::get<0>(V.MItem).data();
#ifdef XPTI_STATISTICS
      MSmallObjects++;
#endif
    }

    std::uninitialized_copy(Data.begin(), Data.end(), Dest);

    return V;
  }

  std::pair<std::string_view, uint8_t> getValue(const Value &V) {
    return std::visit(
        [&V](auto &&Data) {
          return std::make_pair(std::string_view(Data.data(), V.MSize),
                                V.MType);
        },
        V.MItem);
  }

  HashFunction MHashFunction;
  std::vector<Value> MValues;
  std::unordered_map<uint64_t, std::vector<KeyType>> MCache;
  mutable xpti::SharedSpinLock MMutex;

#ifdef XPTI_STATISTICS
  size_t MCacheHits = 0;
  size_t MSmallObjects = 0;
  size_t MLargeObjects = 0;
#endif
};
} // namespace xpti
