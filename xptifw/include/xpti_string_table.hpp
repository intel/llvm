//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
#pragma once

#include "hash_table7.hpp"
#include "xpti/xpti_data_types.h"

#include <atomic>
#include <shared_mutex>
#include <unordered_map>

#ifdef XPTI_STATISTICS
#include <cstdio>
#endif

namespace xpti {
/// @brief A string table class to support the payload handling
/// @details With each payload, a kernel/function name and the source file name
/// may be passed and we need to ensure that the incoming strings are copied and
/// represented in a string table as the incoming strings are guaranteed to be
/// valid only for the duration of the call that handles the payload. This
/// implementation used STL containers protected with std::mutex.
class StringTable {
public:
  using st_forward_t = std::unordered_map<std::string, int32_t>;
  using st_reverse_t = emhash7::HashMap<int32_t, const char *>;

  StringTable(int size = 65536) : MStringToID(size), MIDToString(size) {
    MIds = 1;
#ifdef XPTI_STATISTICS
    MInsertions = 0;
    MRetrievals = 0;
#endif
  }
  ~StringTable() {
    MStringToID.clear();
    MIDToString.clear();
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

    //  Lock-free lookup to see if the string exists in the table; XPTI has
    //  always had this as lock-free, but if instability occurs, we can use a
    //  shared_lock here for this scope
    st_forward_t::iterator Loc;
    {
      // If the string is already present in the string table, return the ID
      Loc = MStringToID.find(str);
      if (Loc != MStringToID.end()) {
#ifdef XPTI_STATISTICS
        MRetrievals++;
#endif
        if (ref_str)
          *ref_str = Loc->first.c_str();

        // We found it, so we return the string ID
        return Loc->second;
      }
    }
    //  If we are here, then the string is not present in the table and 'Loc' is
    //  pointing to MStringToID.end()
    // String not in the table
    // Multiple threads could fall through here
    string_id_t StrID;
    {
      // Employ a double-check pattern here
      std::unique_lock<std::shared_mutex> Lock(MMutex);
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
    return xpti::invalid_id;
  }

  //  The reverse query allows one to get the string from the string_id_t that
  //  may have been cached somewhere.
  const char *query(xpti::string_id_t id) {
    std::shared_lock<std::shared_mutex> lock(MMutex);
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

  int getInsertions() const noexcept {
#ifdef XPTI_STATISTICS
    return MInsertions;
#else
    return 0;
#endif
  }

  int getRetrievals() const noexcept {
#ifdef XPTI_STATISTICS
    return MRetrievals;
#else
    return 0;
#endif
  }

private:
  safe_int32_t MIds;                ///< Thread-safe ID generator
  st_forward_t MStringToID;         ///< Forward lookup hash map
  st_reverse_t MIDToString;         ///< Reverse lookup hash map
  mutable std::shared_mutex MMutex; ///< Mutex required for double-check pattern

  safe_int32_t MStrings; ///< The count of strings in the table
#ifdef XPTI_STATISTICS
  safe_uint64_t MInsertions, ///< Thread-safe tracking of insertions
      MRetrievals;           ///< Thread-safe tracking of lookups
#endif
};
} // namespace xpti
