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
      : MStringToID(size), MIDToString(size), MTableSize(size) {
    MIds = 1;
#ifdef XPTI_STATISTICS
    MInsertions = 0;
    MRetrievals = 0;
#endif
  }

  //  Clear all the contents of this string table and get it ready for re-use
  void clear() {
    MIds = {1};
    MIDToString.clear();
    MStringToID.clear();

    MIDToString.rehash(MTableSize);
    MStringToID.rehash(MTableSize);
#ifdef XPTI_STATISTICS
    MInsertions = 0;
    MRetrievals = 0;
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

    std::string LocalStr = str;
    return add(LocalStr, ref_str);
  }

  xpti::string_id_t add(std::string str, const char **ref_str = nullptr) {
    if (str.empty())
      return xpti::invalid_id;

    // Try to see if the string is already present in the string table
    st_forward_t::const_accessor e;
    if (MStringToID.find(e, str)) {
#ifdef XPTI_STATISTICS
      MRetrievals++;
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
        tbb::spin_mutex::scoped_lock dc(MMutex);
        st_forward_t::accessor f;
        if (MStringToID.insert(f, str)) {
          // If the string does not exist, then insert() returns true. Here we
          // create an ID for it
          id = MIds++;
          f->second = id;
#ifdef XPTI_STATISTICS
          MInsertions++;
#endif
          //  When we insert a new entry into the table, we also need to build
          //  the reverse lookup;
          {
            st_reverse_t::accessor r;
            if (MIDToString.insert(r, id)) {
              //  An entry does not exist, so we will add it to the reverse
              //  lookup.
              r->second = f->first.c_str();
              // Cache the saved string address and send it to the caller
              if (ref_str)
                *ref_str = r->second;
              f.release();
              r.release();
              MStrings++;
              return id;
            } else {
              // We cannot have a case where a string is not present in the
              // forward lookup and present in the reverse lookup
              MStringToID.erase(f);
              if (ref_str)
                *ref_str = nullptr;

              return xpti::invalid_id;
            }
          }

        } else {
          // The string has already been added, so we return the stored ID
          id = f->second;
#ifdef XPTI_STATISTICS
          MRetrievals++;
#endif
          if (ref_str)
            *ref_str = f->first.c_str();
          return id;
        }
        // Both the accessor and MMutex will be released here!
      }
    }
    return xpti::invalid_id;
  }

  //  The reverse query allows one to get the string from the string_id_t that
  //  may have been cached somewhere.
  const char *query(xpti::string_id_t id) {
    st_reverse_t::const_accessor e;
    if (MIDToString.find(e, id)) {
#ifdef XPTI_STATISTICS
      MRetrievals++;
#endif
      return e->second;
    } else
      return nullptr;
  }

  int32_t count() { return (int32_t)MStrings; }

  const st_reverse_t &table() { return MIDToString; }

  void printStatistics() {
#ifdef XPTI_STATISTICS
    printf("String table inserts: [%llu]\n", MInsertions.load());
    printf("String table lookups: [%llu]\n", MRetrievals.load());
#endif
  }

private:
  safe_int32_t MIds;        ///< Thread-safe ID generator
  st_forward_t MStringToID; ///< Forward lookup hash map
  st_reverse_t MIDToString; ///< Reverse lookup hash map
  int32_t MTableSize;       ///< Initial table size of the hash-map
  tbb::spin_mutex MMutex;   ///< Mutex required for double-check pattern
  safe_int32_t MStrings;    ///< The count of strings in the table
#ifdef XPTI_STATISTICS
  safe_uint64_t MInsertions, ///< Thread-safe tracking of insertions
      MRetrievals;           ///< Thread-safe tracking of lookups
#endif
};
} // namespace xpti
#else // Non-TBB implementation follows

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
      : MStringToID(size), MIDToString(size), MTableSize(size) {
    MIds = 1;
#ifdef XPTI_STATISTICS
    MInsertions = 0;
    MRetrievals = 0;
#endif
  }

  //  Clear all the contents of this string table and get it ready for re-use
  void clear() {
    MIds = 1;
    MIDToString.clear();
    MStringToID.clear();

#ifdef XPTI_STATISTICS
    MInsertions = 0;
    MRetrievals = 0;
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

    std::string LocalStr = str;
    return add(LocalStr, ref_str);
  }

  xpti::string_id_t add(std::string str, const char **ref_str = nullptr) {
    if (str.empty())
      return xpti::invalid_id;

    // Try to see if the string is already present in the string table
    auto Loc = MStringToID.find(str);
    if (Loc != MStringToID.end()) {
#ifdef XPTI_STATISTICS
      MRetrievals++;
#endif
      if (ref_str)
        *ref_str = Loc->first.c_str();

      // We found it, so we return the string ID
      return Loc->second;
    } else {
      // String not in the table
      // Multiple threads could fall through here
      string_id_t StrID;
      {
        // Employ a double-check pattern here
        std::lock_guard<std::mutex> lock(MMutex);
        auto Loc = MStringToID.find(str);
        // String not present in the table
        if (Loc == MStringToID.end()) {
          // Add it
          StrID = MIds++;
          auto Entry = MStringToID.insert(st_forward_t::value_type(str, StrID));
          if (ref_str)
            *ref_str = Entry.first->first.c_str();
#ifdef XPTI_STATISTICS
          MInsertions++;
#endif
          //  When we insert a new entry into the table, we also need to build
          //  the reverse lookup;
          {
            auto IDLoc = MIDToString.find(StrID);
            if (IDLoc == MIDToString.end()) {
              //  An entry does not exist, so we will add it to the reverse
              //  lookup.
              MIDToString[StrID] = Entry.first->first.c_str();
              // Cache the saved string address and send it to the caller
              MStrings++;
              return StrID;
            } else {
              // We cannot have a case where a string is not present in the
              // forward lookup and present in the reverse lookup
              MStringToID.erase(Loc);
              if (ref_str)
                *ref_str = nullptr;

              return xpti::invalid_id;
            }
          }

        } else {
          // The string has already been added, so we return the stored ID
          StrID = Loc->second;
#ifdef XPTI_STATISTICS
          MRetrievals++;
#endif
          if (ref_str)
            *ref_str = Loc->first.c_str();
          return StrID;
        }
        // The MMutex will be released here!
      }
    }
    return xpti::invalid_id;
  }

  //  The reverse query allows one to get the string from the string_id_t that
  //  may have been cached somewhere.
  const char *query(xpti::string_id_t id) {
    std::lock_guard<std::mutex> lock(MMutex);
    auto Loc = MIDToString.find(id);
    if (Loc != MIDToString.end()) {
#ifdef XPTI_STATISTICS
      MRetrievals++;
#endif
      return Loc->second;
    } else
      return nullptr;
  }

  int32_t count() { return (int32_t)MStrings; }

  const st_reverse_t &table() { return MIDToString; }

  void printStatistics() {
#ifdef XPTI_STATISTICS
    printf("String table inserts: [%llu]\n", MInsertions.load());
    printf("String table lookups: [%llu]\n", MRetrievals.load());
#endif
  }

private:
  safe_int32_t MIds;         ///< Thread-safe ID generator
  st_forward_t MStringToID;  ///< Forward lookup hash map
  st_reverse_t MIDToString;  ///< Reverse lookup hash map
  int32_t MTableSize;        ///< Initial table size of the hash-map
  std::mutex MMutex;         ///< Mutex required for double-check pattern
                             ///< Replace with reader-writer lock in C++14
  safe_int32_t MStrings;     ///< The count of strings in the table
#ifdef XPTI_STATISTICS
  safe_uint64_t MInsertions, ///< Thread-safe tracking of insertions
      MRetrievals;           ///< Thread-safe tracking of lookups
#endif
};
} // namespace xpti
#endif
