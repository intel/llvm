//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
#pragma once
#include "xpti_data_types.h"

#include <atomic>
#include <iostream>
#include <mutex>
#include <unordered_map>

#ifdef XPTI_STATISTICS
#include <stdio.h>
#endif

#ifdef XPTI_USE_TBB

#include <tbb/concurrent_hash_map.h>
#include <tbb/spin_mutex.h>

#pragma message("Using TBB concurrent containers...")
namespace xpti {
/// \brief A string table class to support the payload handling
/// \details With each payload, a kernel/function name and the source file name
/// may be passed and we need to ensure that the incoming strings are copied and
/// represented in a string table as the incoming strings are guaranteed to be
/// valid only for the duration of the call that handles the payload. This
/// implementation used Threading Building Blocks concurrent containers.
class StringTable {
public:
  using st_forward_t = tbb::concurrent_hash_map<std::string, int32_t>;
  using st_reverse_t = tbb::concurrent_hash_map<int32_t, const char *>;

  StringTable(int size = 4096)
      : m_str2id(size), m_id2str(size), m_table_size(size) {
    m_ids = 1;
#ifdef XPTI_STATISTICS
    m_insert = 0;
    m_lookup = 0;
#endif
  }

  //  Clear all the contents of this string table and get it ready for re-use
  void clear() {
    m_ids = {1};
    m_id2str.clear();
    m_str2id.clear();

    m_id2str.rehash(m_table_size);
    m_str2id.rehash(m_table_size);
#ifdef XPTI_STATISTICS
    m_insert = 0;
    m_lookup = 0;
#endif
  }

  // If the string being added to the string table is empty or invalid, then
  // the returned string id = invalid_id;
  //
  // On success, the string will be inserted into two tables - one that maps
  // string to string ID and another that maps from string ID to string. If a
  // reference string pointer is made available, then the address of the string
  // in the string table is returned through the default argument
  xpti::string_id_t add(const char *str, const char **ref_str = nullptr) {
    if (!str)
      return xpti::invalid_id;

    std::string local_str = str;
    return add(local_str, ref_str);
  }

  xpti::string_id_t add(std::string str, const char **ref_str = nullptr) {
    if (str.empty())
      return xpti::invalid_id;

    // Try to see if the string is already present in the string table
    st_forward_t::const_accessor e;
    if (m_str2id.find(e, str)) {
#ifdef XPTI_STATISTICS
      m_lookup++;
#endif
      if (ref_str)
        *ref_str = e->first.c_str();

      // We found it, so we return the string ID
      return e->second;
    } else {
      // Multiple threads could fall through here Release the reader lock held
      e.release();
      string_id_t id;
      {
        // Employ a double-check pattern here
        tbb::spin_mutex::scoped_lock dc(m_mutex);
        st_forward_t::accessor f;
        if (m_str2id.insert(f, str)) {
          // If the string does not exist, then insert() returns true. Here we
          // create an ID for it
          id = m_ids++;
          f->second = id;
#ifdef XPTI_STATISTICS
          m_insert++;
#endif
          //  When we insert a new entry into the table, we also need to build
          //  the reverse lookup;
          {
            st_reverse_t::accessor r;
            if (m_id2str.insert(r, id)) {
              //  An entry does not exist, so we will add it to the reverse
              //  lookup.
              r->second = f->first.c_str();
              // Cache the saved string address and send it to the caller
              if (ref_str)
                *ref_str = r->second;
              f.release();
              r.release();
              m_strings++;
              return id;
            } else {
              // We cannot have a case where a string is not present in the
              // forward lookup and present in the reverse lookup
              m_str2id.erase(f);
              if (ref_str)
                *ref_str = nullptr;

              return xpti::invalid_id;
            }
          }

        } else {
          // The string has already been added, so we return the stored ID
          id = f->second;
#ifdef XPTI_STATISTICS
          m_lookup++;
#endif
          if (ref_str)
            *ref_str = f->first.c_str();
          return id;
        }
        // Both the accessor and m_mutex will be released here!
      }
    }
    return xpti::invalid_id;
  }

  //  The reverse query allows one to get the string from the string_id_t that
  //  may have been cached somewhere.
  const char *query(xpti::string_id_t id) {
    st_reverse_t::const_accessor e;
    if (m_id2str.find(e, id)) {
#ifdef XPTI_STATISTICS
      m_lookup++;
#endif
      return e->second;
    } else
      return nullptr;
  }

  int32_t count() { return (int32_t)m_strings; }

  const st_reverse_t &table() { return m_id2str; }

  void printStatistics() {
#ifdef XPTI_STATISTICS
    printf("String table inserts: [%llu]\n", m_insert.load());
    printf("String table lookups: [%llu]\n", m_lookup.load());
#endif
  }

private:
  safe_int32_t m_ids;      ///< Thread-safe ID generator
  st_forward_t m_str2id;   ///< Forward lookup hash map
  st_reverse_t m_id2str;   ///< Reverse lookup hash map
  int32_t m_table_size;    ///< Initial table size of the hash-map
  tbb::spin_mutex m_mutex; ///< Mutex required for double-check pattern
  safe_int32_t m_strings;  ///< The count of strings in the table
#ifdef XPTI_STATISTICS
  safe_uint64_t m_insert, ///< Thread-safe tracking of insertions
      m_lookup;           ///< Thread-safe tracking of lookups
#endif
};
} // namespace xpti
#else
namespace xpti {
/// \brief A string table class to support the payload handling
/// \details With each payload, a kernel/function name and the source file name
/// may be passed and we need to ensure that the incoming strings are copied and
/// represented in a string table as the incoming strings are guaranteed to be
/// valid only for the duration of the call that handles the payload. This
/// implementation used STL containers protected with std::mutex.
class StringTable {
public:
  using st_forward_t = std::unordered_map<std::string, int32_t>;
  using st_reverse_t = std::unordered_map<int32_t, const char *>;

  StringTable(int size = 4096)
      : m_str2id(size), m_id2str(size), m_table_size(size) {
    m_ids = 1;
#ifdef XPTI_STATISTICS
    m_insert = 0;
    m_lookup = 0;
#endif
  }

  //  Clear all the contents of this string table and get it ready for re-use
  void clear() {
    m_ids = {1};
    m_id2str.clear();
    m_str2id.clear();

#ifdef XPTI_STATISTICS
    m_insert = 0;
    m_lookup = 0;
#endif
  }

  // If the string being added to the string table is empty or invalid, then
  // the returned string id = invalid_id;
  //
  // On success, the string will be inserted into two tables - one that maps
  // string to string ID and another that maps from string ID to string. If a
  // reference string pointer is made available, then the address of the string
  // in the string table is returned through the default argument
  xpti::string_id_t add(const char *str, const char **ref_str = nullptr) {
    if (!str)
      return xpti::invalid_id;

    std::string local_str = str;
    return add(local_str, ref_str);
  }

  xpti::string_id_t add(std::string str, const char **ref_str = nullptr) {
    if (str.empty())
      return xpti::invalid_id;

    // Try to see if the string is already present in the string table
    auto loc = m_str2id.find(str);
    if (loc != m_str2id.end()) {
#ifdef XPTI_STATISTICS
      m_lookup++;
#endif
      if (ref_str)
        *ref_str = loc->first.c_str();

      // We found it, so we return the string ID
      return loc->second;
    } else {
      // Multiple threads could fall through here
      string_id_t id;
      {
        // Employ a double-check pattern here
        std::lock_guard<std::mutex> lock(m_mutex);
        auto loc = m_str2id.find(str);
        // String not present in the table
        if (loc == m_str2id.end()) {
          // Add it
          id = m_ids++;
          m_str2id[str] = id;
          loc = m_str2id.find(str);
          if (ref_str)
            *ref_str = loc->first.c_str();
#ifdef XPTI_STATISTICS
          m_insert++;
#endif
          //  When we insert a new entry into the table, we also need to build
          //  the reverse lookup;
          {
            auto id_loc = m_id2str.find(id);
            if (id_loc == m_id2str.end()) {
              //  An entry does not exist, so we will add it to the reverse
              //  lookup.
              m_id2str[id] = loc->first.c_str();
              // Cache the saved string address and send it to the caller
              m_strings++;
              return id;
            } else {
              // We cannot have a case where a string is not present in the
              // forward lookup and present in the reverse lookup
              m_str2id.erase(loc);
              if (ref_str)
                *ref_str = nullptr;

              return xpti::invalid_id;
            }
          }

        } else {
          // The string has already been added, so we return the stored ID
          id = loc->second;
#ifdef XPTI_STATISTICS
          m_lookup++;
#endif
          if (ref_str)
            *ref_str = loc->first.c_str();
          return id;
        }
        // The m_mutex will be released here!
      }
    }
    return xpti::invalid_id;
  }

  //  The reverse query allows one to get the string from the string_id_t that
  //  may have been cached somewhere.
  const char *query(xpti::string_id_t id) {
    std::lock_guard<std::mutex> lock(m_mutex);
    auto loc = m_id2str.find(id);
    if (loc != m_id2str.end()) {
#ifdef XPTI_STATISTICS
      m_lookup++;
#endif
      return loc->second;
    } else
      return nullptr;
  }

  int32_t count() { return (int32_t)m_strings; }

  const st_reverse_t &table() { return m_id2str; }

  void printStatistics() {
#ifdef XPTI_STATISTICS
    printf("String table inserts: [%llu]\n", m_insert.load());
    printf("String table lookups: [%llu]\n", m_lookup.load());
#endif
  }

private:
  safe_int32_t m_ids;     ///< Thread-safe ID generator
  st_forward_t m_str2id;  ///< Forward lookup hash map
  st_reverse_t m_id2str;  ///< Reverse lookup hash map
  int32_t m_table_size;   ///< Initial table size of the hash-map
  std::mutex m_mutex;     ///< Mutex required for double-check pattern
                          ///< Replace with reader-writer lock in C++14
  safe_int32_t m_strings; ///< The count of strings in the table
#ifdef XPTI_STATISTICS
  safe_uint64_t m_insert, ///< Thread-safe tracking of insertions
      m_lookup;           ///< Thread-safe tracking of lookups
#endif
};
} // namespace xpti
#endif
