//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
#pragma once
#include "xpti_data_types.h"

#include <atomic>
#include <mutex>
#include <unordered_map>

#ifdef XPTI_STATISTICS
#include <cstdio>
#endif

#ifdef XPTI_USE_TBB
#include <tbb/concurrent_hash_map.h>
#include <tbb/spin_mutex.h>

namespace xpti {
/// \brief A class for mapping one 64-bit value to another 64-bit value
/// \details With each payload, a kernel/function name and the source file name
/// may be passed and we need to ensure that the payload can be cached in a hash
/// map that maps a unique value from the payload to a universal ID. We could
/// use the payload hash for this purpose, but the numbers are non-monotonic and
/// can be harder to debug. This implementation of the hash table uses Threading
/// Building Blocks concurrent containers for multi-threaded efficiency.
class Hash64x64Table {
public:
  using ht_lut_t = tbb::concurrent_hash_map<int64_t, int64_t>;

  Hash64x64Table(int size = 1024)
      : MForward(size), MReverse(size), MTableSize(size) {
#ifdef XPTI_STATISTICS
    MInsertions = 0;
    MRetrievals = 0;
#endif
  }

  ~Hash64x64Table() {
    MForward.clear();
    MReverse.clear();
  }

  //  Clear all the contents of this hash table and get it ready for re-use
  void clear() {
    MForward.clear();
    MReverse.clear();
    MForward.rehash(MTableSize);
    MReverse.rehash(MTableSize);
#ifdef XPTI_STATISTICS
    MInsertions = 0;
    MRetrievals = 0;
#endif
  }

  //  Check to see if a particular Key is already present in the table;
  //
  //  On success, the value for the Key will be returned. If not,
  //  xpti::invalid_id will be returned.
  int64_t find(int64_t Key) {
    //  Try to read it, if already present
    ht_lut_t::const_accessor e;
    if (MForward.find(e, Key)) {
#ifdef XPTI_STATISTICS
      MRetrievals++;
#endif
      return e->second; // We found it, so we return the value
    } else
      return xpti::invalid_id;
  }

  //  Add a <Key, Value> pair to the hash table. If the Key already exists, this
  //  call returns even if the value happens to be different this time.
  //
  //  If the Key does not exist, then the Key is inserted into the hash map and
  //  the reverse lookup populated with the <Value, Key> pair.
  void add(int64_t Key, int64_t Value) {
    //  Try to read it, if already present
    ht_lut_t::const_accessor e;
    if (MForward.find(e, Key)) {
#ifdef XPTI_STATISTICS
      MRetrievals++;
#endif
    } else { // Multiple threads could fall through here
      // Release the reader lock held;
      e.release();
      {
        // Employ a double-check pattern here
        tbb::spin_mutex::scoped_lock dc(MMutex);
        ht_lut_t::accessor f;
        if (MForward.insert(f, Key)) {
          // The Key does not exist, so we will add the Key-Value pair to the
          // hash map
          f->second = Value;
#ifdef XPTI_STATISTICS
          MInsertions++;
#endif
          // When we insert a new entry into the table, we also need to build
          // the reverse lookup;
          {
            ht_lut_t::accessor r;
            if (MReverse.insert(r, Value)) {
              // An entry does not exist, so we will add it to the reverse
              // lookup.
              r->second = Key;
              f.release();
              r.release();
            }
          }
        }
        // else, we do not add the Key-Value pair as the Key already exists in
        // the table!
      }
    }
  }

  //  The reverse query allows one to get the Value from the Key that may have
  //  been cached somewhere.
  int64_t reverseFind(int64_t Value) {
    ht_lut_t::const_accessor e;
    if (MReverse.find(e, Value)) {
#ifdef XPTI_STATISTICS
      MRetrievals++;
#endif
      return e->second;
    } else
      return xpti::invalid_id;
  }

  void printStatistics() {
#ifdef XPTI_STATISTICS
    printf("Hash table inserts : [%llu]\n", MInsertions.load());
    printf("Hash table lookups : [%llu]\n", MRetrievals.load());
#endif
  }

private:
  ht_lut_t MForward;  ///< Forward lookup hash map
  ht_lut_t MReverse;  ///< Reverse lookup hash map
  int32_t MTableSize; ///< Initial size of the hash map
  tbb::spin_mutex
      MMutex; ///< Mutex required to implement a double-check pattern
#ifdef XPTI_STATISTICS
  safe_uint64_t MInsertions, ///< Thread-safe tracking of insertions
      MRetrievals;           ///< Thread-safe tracking of lookups
#endif
};

#else
namespace xpti {
/// \brief A class for mapping one 64-bit value to another 64-bit value
/// \details With each payload, a kernel/function name and the source file name
/// may be passed and we need to ensure that the payload can be cached in a hash
/// map that maps a unique value from the payload to a universal ID. We could
/// use the payload hash for this purpose, but the numbers are non-monotonic and
/// can be harder to debug. This implementation of the hash table uses std
/// library containers.
class Hash64x64Table {
public:
  using ht_lut_t = std::unordered_map<int64_t, int64_t>;

  Hash64x64Table(int size = 1024) : MForward(size), MReverse(size) {
#ifdef XPTI_STATISTICS
    MInsertions = 0;
    MRetrievals = 0;
#endif
  }

  ~Hash64x64Table() {
    MForward.clear();
    MReverse.clear();
  }

  //  Clear all the contents of this hash table and get it ready for re-use
  void clear() {
    MForward.clear();
    MReverse.clear();
#ifdef XPTI_STATISTICS
    MInsertions = 0;
    MRetrievals = 0;
#endif
  }

  //  Check to see if a particular Key is already present in the table;
  //
  //  On success, the Value for the Key will be returned. If not,
  //  xpti::invalid_id will be returned.
  int64_t find(int64_t Key) {
    std::lock_guard<std::mutex> Lock(MMutex);
    //  Try to read it, if already present
    auto KeyLoc = MForward.find(Key);
    if (KeyLoc != MForward.end()) {
#ifdef XPTI_STATISTICS
      MRetrievals++;
#endif
      return KeyLoc->second; // We found it, so we return the Value
    } else
      return xpti::invalid_id;
  }

  //  Add a <Key, Value> pair to the hash table. If the Key already exists, this
  //  call returns even if the Value happens to be different this time.
  //
  //  If the Key does not exist, then the Key is inserted into the hash map and
  //  the reverse lookup populated with the <Value, Key> pair.
  void add(int64_t Key, int64_t Value) {
    //  Try to read it, if already present
    auto KeyLoc = MForward.find(Key);
    if (KeyLoc != MForward.end()) {
#ifdef XPTI_STATISTICS
      MRetrievals++;
#endif
    } else { // Multiple threads could fall through here
      {
        // Employ a double-check pattern here
        std::lock_guard<std::mutex> Lock(MMutex);
        auto KeyLoc = MForward.find(Key);
        if (KeyLoc == MForward.end()) {
          // The Key does not exist, so we will add the Key-Value pair to the
          // hash map
          MForward[Key] = Value;
          KeyLoc = MForward.find(Key);
#ifdef XPTI_STATISTICS
          MInsertions++;
#endif
          // When we insert a new entry into the table, we also need to build
          // the reverse lookup;
          {
            auto ValLoc = MReverse.find(Value);
            if (ValLoc == MReverse.end()) {
              // An entry does not exist, so we will add it to the reverse
              // lookup.
              MReverse[Value] = Key;
            } else {
              MForward.erase(KeyLoc);
            }
          }
        }
        // else, we do not add the Key-Value pair as the Key already exists in
        // the table!
      }
    }
  }

  //  The reverse query allows one to get the Value from the Key that may have
  //  been cached somewhere.
  int64_t reverseFind(int64_t Value) {
    std::lock_guard<std::mutex> Lock(MMutex);
    auto ValLoc = MReverse.find(Value);
    if (ValLoc != MReverse.end()) {
#ifdef XPTI_STATISTICS
      MRetrievals++;
#endif
      return ValLoc->second;
    } else
      return xpti::invalid_id;
  }

  void printStatistics() {
#ifdef XPTI_STATISTICS
    printf("Hash table inserts : [%llu]\n", MInsertions.load());
    printf("Hash table lookups : [%llu]\n", MRetrievals.load());
#endif
  }

private:
  ht_lut_t MForward; ///< Forward lookup hash map
  ht_lut_t MReverse; ///< Reverse lookup hash map
  std::mutex MMutex; ///< Mutex required to implement a double-check pattern
#ifdef XPTI_STATISTICS
  safe_uint64_t MInsertions, ///< Thread-safe tracking of insertions
      MRetrievals;           ///< Thread-safe tracking of lookups
#endif
};
#endif
} // namespace xpti
