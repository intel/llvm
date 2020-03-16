//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
#pragma once
#include "xpti_data_types.h"

#include <atomic>

#ifdef XPTI_STATISTICS
#include <stdio.h>
#endif

#include <tbb/concurrent_hash_map.h>
#include <tbb/spin_mutex.h>

namespace xpti {
/// @brief A class for mapping one 64-bit value to another 64-bit value
/// @details With each payload, a kernel/function name and the
/// source file name may be passed and we need to ensure that the
/// payload can be cached in a hash map that maps a unique value
/// from the payload to a universal ID. We could use the payload
/// hash for this purpose, but the numbers are non -monotonic and
/// can be harder to debug.
///
class Hash64x64Table {
public:
  typedef tbb::concurrent_hash_map<int64_t, int64_t> ht_lut_t;

  Hash64x64Table(int size = 1024)
      : m_forward(size), m_reverse(size), m_table_size(size) {
#ifdef XPTI_STATISTICS
    m_insert = 0;
    m_lookup = 0;
#endif
  }

  ~Hash64x64Table() {
    m_forward.clear();
    m_reverse.clear();
  }

  //  Clear all the contents of this hash table and get it ready for re-use
  void clear() {
    m_forward.clear();
    m_reverse.clear();
    m_forward.rehash(m_table_size);
    m_reverse.rehash(m_table_size);
#ifdef XPTI_STATISTICS
    m_insert = 0;
    m_lookup = 0;
#endif
  }

  //  Check to see if a particular key is already present in the table;
  //
  //  On success, the value for the key will be returned. If not,
  //  xpti::invalid_id will be returned.
  int64_t find(int64_t key) {
    //  Try to read it, if already present
    ht_lut_t::const_accessor e;
    if (m_forward.find(e, key)) {
#ifdef XPTI_STATISTICS
      m_lookup++;
#endif
      return e->second; // We found it, so we return the value
    } else
      return xpti::invalid_id;
  }

  //  Add a <key, value> pair to the hash table. If the key already exists, this
  //  call returns even if the value happens to be different this time.
  //
  //  If the key does not exists, then the key is inserted into the hash map and
  //  the reverse lookup populated with the <value, key> pair.
  void add(int64_t key, int64_t value) {
    //  Try to read it, if already present
    ht_lut_t::const_accessor e;
    if (m_forward.find(e, key)) {
#ifdef XPTI_STATISTICS
      m_lookup++;
#endif
    } else { // Multiple threads could fall through here
      // Release the reader lock held;
      e.release();
      {
        // Employ a double-check pattern here
        tbb::spin_mutex::scoped_lock dc(m_mutex);
        ht_lut_t::accessor f;
        if (m_forward.insert(f, key)) {
          // The key does not exist, so we will add the key-value pair to the
          // hash map
          f->second = value;
#ifdef XPTI_STATISTICS
          m_insert++;
#endif
          // When we insert a new entry into the table, we also need to build
          // the reverse lookup;
          {
            ht_lut_t::accessor r;
            if (m_reverse.insert(r, value)) {
              // An entry does not exist, so we will add it to the reverse
              // lookup.
              r->second = key;
              f.release();
              r.release();
            }
          }
        }
        // else, we do not add the key-value pair as the key already exists in
        // the table!
      }
    }
  }

  //  The reverse query allows one to get the value from the key that may have
  //  been cached somewhere.
  int64_t reverseFind(int64_t value) {
    ht_lut_t::const_accessor e;
    if (m_reverse.find(e, value)) {
#ifdef XPTI_STATISTICS
      m_lookup++;
#endif
      return e->second;
    } else
      return xpti::invalid_id;
  }

  void printStatistics() {
#ifdef XPTI_STATISTICS
    printf("Hash table inserts : [%llu]\n", m_insert.load());
    printf("Hash table lookups : [%llu]\n", m_lookup.load());
#endif
  }

private:
  ht_lut_t m_forward;   ///< Forward lookup hash map
  ht_lut_t m_reverse;   ///< Reverse lookup hash map
  int32_t m_table_size; ///< Initial size of the hash map
  tbb::spin_mutex
      m_mutex; ///< Mutex required to implement a double-check pattern
#ifdef XPTI_STATISTICS
  safe_uint64_t m_insert, ///< Thread-safe tracking of insertions
      m_lookup;           ///< Thread-safe tracking of lookups
#endif
};
} // namespace xpti