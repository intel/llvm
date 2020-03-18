//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
#include "xpti_trace_framework.hpp"
#include "xpti_int64_hash_table.hpp"
#include "xpti_string_table.hpp"

#include <cassert>
#include <map>
#include <mutex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#ifdef XPTI_STATISTICS
#include <stdio.h>
#endif

#ifdef XPTI_USE_TBB
#include <tbb/concurrent_hash_map.h>
#include <tbb/concurrent_unordered_map.h>
#include <tbb/concurrent_vector.h>
#include <tbb/spin_mutex.h>
#endif

#define XPTI_USER_DEFINED_TRACE_TYPE16(value)                                  \
  ((uint16_t)xpti::trace_point_type_t::user_defined | (uint16_t)value)
#define XPTI_USER_DEFINED_EVENT_TYPE16(value)                                  \
  ((uint16_t)xpti::trace_event_type_t::user_defined | (uint16_t)value)
#define XPTI_EXTRACT_MSB16(val) (val >> 16)
#define XPTI_EXTRACT_LSB16(val) (val & 0x0000ffff)

#define XPTI_VENDOR_DEFINED_TRACE_TYPE16(vendor_id, trace_type)                \
  ((uint16_t)vendor_id << 8 | XPTI_USER_DEFINED_TRACE_TYPE16(trace_type))
#define XPTI_VENDOR_DEFINED_EVENT_TYPE16(vendor_id, event_type)                \
  ((uint16_t)vendor_id << 8 | XPTI_USER_DEFINED_EVENT_TYPE16(event_type))

namespace xpti {
constexpr const char *env_subscribers = "XPTI_SUBSCRIBERS";
xpti::utils::PlatformHelper g_helper;
// This class is a helper class to load all the listed subscribers provided by
// the user in XPTI_SUBSCRIBERS environment variable.
class Subscribers {
public:
  // Data structure to hold the plugin related information, including the
  // initialization and finalization functions
  struct plugin_data_t {
    /// The handle of the loaded shared object
    xpti_plugin_handle_t handle = nullptr;
    /// The initialization entry point
    xpti::plugin_init_t init = nullptr;
    /// The finalization entry point
    xpti::plugin_fini_t fini = nullptr;
    /// The name of the shared object (in UTF8?))
    std::string name;
    /// indicates whether the data structure is valid
    bool valid = false;
  };
  // Data structures defined to hold the plugin data that can be looked up by
  // plugin name or the handle
  //
  using plugin_handle_lut_t = std::map<xpti_plugin_handle_t, plugin_data_t>;
  using plugin_name_lut_t = std::map<std::string, plugin_data_t>;

  // We unload all loaded shared objects in the destructor; Must not be invoked
  // in the DLLMain() function and possibly the __fini() function in Linux
  ~Subscribers() { unloadAllPlugins(); }
  // Method to query the plugin data information using the handle. If there's no
  // information present for the handle provided, the method returns a structure
  // with the valid attribute set to 'false'
  plugin_data_t queryPlugin(xpti_plugin_handle_t h) {
    plugin_data_t p;
#ifdef XPTI_USE_TBB
    tbb::spin_mutex::scoped_lock my_lock(m_mutex);
#else
    std::lock_guard<std::mutex> lock(m_mutex);
#endif
    if (m_handle_lut.count(h))
      return m_handle_lut[h];
    else
      return p; // return invalid plugin data
  }

  // Load the provided shared object file name using the explicit load API. If
  // the load is successful, a test is performed to see if the shared object has
  // the required entry points for it to be considered a trace plugin
  // subscriber. If so, the internal data structures are updated and a valid
  // handle is returned.
  //
  // If not, the shared object is unloaded and a NULL handle is returned.
  xpti_plugin_handle_t loadPlugin(const char *path) {
    xpti_plugin_handle_t handle = 0;
    std::string error;
    // Check to see if the subscriber has already been loaded; if so, return the
    // handle from the previously loaded library
    if (m_name_lut.count(path)) {
#ifdef XPTI_USE_TBB
      tbb::spin_mutex::scoped_lock my_lock(m_mutex);
#else
      std::lock_guard<std::mutex> lock(m_mutex);
#endif
      // This plugin has already been loaded, so let's return previously
      // recorded handle
      printf("Plugin (%s) has already been loaded..\n", path);
      plugin_data_t &d = m_name_lut[path];
      assert(d.valid && "Lookup is invalid!");
      if (d.valid)
        return d.handle;
    }

    handle = g_helper.loadLibrary(path, error);
    if (handle) {
      // The tracing framework requires the tool plugins to implement the
      // xptiTraceInit() and xptiTraceFinish() functions. If these are not
      // present, then the plugin will be ruled an invalid plugin and unloaded
      // from the process.
      xpti::plugin_init_t init =
          (xpti::plugin_init_t)g_helper.findFunction(handle, "xptiTraceInit");
      xpti::plugin_fini_t fini =
          (xpti::plugin_fini_t)g_helper.findFunction(handle, "xptiTraceFinish");
      if (init && fini) {
        //  We appear to have loaded a valid plugin, so we will insert the
        //  plugin information into the two maps guarded by a lock
        plugin_data_t d;
        d.valid = true;
        d.handle = handle;
        d.name = path;
        d.init = init;
        d.fini = fini;
#ifdef XPTI_USE_TBB
        tbb::spin_mutex::scoped_lock my_lock(m_mutex);
#else
        std::lock_guard<std::mutex> lock(m_mutex);
#endif
        m_name_lut[path] = d;
        m_handle_lut[handle] = d;
      } else {
        // We may have loaded another shared object that is not a tool plugin
        // for the tracing framework, so we'll unload it now
        unloadPlugin(handle);
        handle = nullptr;
      }
    } else {
      //  Get error from errno
      if (!error.empty())
        printf("[%s]: %s\n", path, error.c_str());
    }
    return handle;
  }

  //  Unloads the shared object identified by the handle provided. If
  //  successful, returns a success code, else a failure code.
  xpti::result_t unloadPlugin(xpti_plugin_handle_t h) {
    xpti::result_t res = g_helper.unloadLibrary(h);
    if (xpti::result_t::XPTI_RESULT_SUCCESS == res) {
      auto it = m_handle_lut.find(h);
      if (it != m_handle_lut.end()) {
        m_handle_lut.erase(h);
      }
    }
    return res;
  }

  // Quick test to see if there are registered subscribers
  bool hasValidSubscribers() { return (m_handle_lut.size() > 0); }

  void initializeForStream(const char *stream, uint32_t major_revision,
                           uint32_t minor_revision,
                           const char *version_string) {
    //  If there are subscribers registered, then initialize the subscribers
    //  with the new stream information.
    if (m_handle_lut.size()) {
      for (auto &handle : m_handle_lut) {
        handle.second.init(major_revision, minor_revision, version_string,
                           stream);
      }
    }
  }

  void finalizeForStream(const char *stream) {
    //  If there are subscribers registered, then finalize the subscribers for
    //  the stream
    if (m_handle_lut.size()) {
      for (auto &handle : m_handle_lut) {
        handle.second.fini(stream);
      }
    }
  }

  void loadFromEnvironmentVariable() {
    if (!g_helper.checkTraceEnv())
      return;
    //  Load all registered listeners by scanning the environment variable in
    //  "env"; The environment variable, if set, extract the comma separated
    //  tokens into a vector.
    std::string token, env = g_helper.getEnvironmentVariable(env_subscribers);
    std::vector<std::string> listeners;
    std::stringstream stream(env);

    //  Split the environment variable value by ',' and build a vector of the
    //  tokens (subscribers)
    while (std::getline(stream, token, ',')) {
      listeners.push_back(token);
    }

    size_t valid_subscribers = listeners.size();
    if (valid_subscribers) {
      //  Let's go through the subscribers and load these plugins;
      for (auto &path : listeners) {
        // Load the plugins listed in the environment variable
#ifdef XPTI_USE_TBB
        tbb::spin_mutex::scoped_lock my_lock(m_loader);
#else
        std::lock_guard<std::mutex> lock(m_loader);
#endif
        auto subs_handle = loadPlugin(path.c_str());
        if (!subs_handle) {
          valid_subscribers--;
          printf("Failed to load %s successfully...\n", path.c_str());
        }
      }
    }
  }

  void unloadAllPlugins() {
    for (auto &item : m_name_lut) {
      unloadPlugin(item.second.handle);
    }

    m_handle_lut.clear();
    m_name_lut.clear();
  }

private:
  /// Hash map that maps shared object name to the plugin data
  plugin_name_lut_t m_name_lut;
  /// Hash map that maps shared object handle to the plugin data
  plugin_handle_lut_t m_handle_lut;
#ifdef XPTI_USE_TBB
  /// Lock to ensure the operation on these maps are safe
  tbb::spin_mutex m_mutex;
  /// Lock to ensure that only one load happens at a time
  tbb::spin_mutex m_loader;
#else
  /// Lock to ensure the operation on these maps are safe
  std::mutex m_mutex;
  /// Lock to ensure that only one load happens at a time
  std::mutex m_loader;
#endif
};

/// \brief Helper class to create and  manage tracepoints
/// \details The class uses the global string table to register the strings it
/// encounters in various payloads and builds internal hash maps to manage them.
/// This is a single point for managing tracepoints.
class Tracepoints {
public:
#ifdef XPTI_USE_TBB
  using va_uid_t = tbb::concurrent_unordered_map<uint64_t, int64_t>;
  using uid_payload_t = tbb::concurrent_unordered_map<int64_t, xpti::payload_t>;
  using uid_event_t =
      tbb::concurrent_unordered_map<int64_t, xpti::trace_event_data_t>;
#else
  using va_uid_t = std::unordered_map<uint64_t, int64_t>;
  using uid_payload_t = std::unordered_map<int64_t, xpti::payload_t>;
  using uid_event_t = std::unordered_map<int64_t, xpti::trace_event_data_t>;
#endif

  Tracepoints(xpti::StringTable &st)
      : m_uid(1), m_insert(0), m_lookup(0), m_string_table(st) {
    // Nothing requires to be done at construction time
  }

  ~Tracepoints() { clear(); }

  void clear() {
    m_string_table.clear();
    // We will always start our ID
    // stream from 1. 0 is null_id
    // and -1 is invalid_id
    m_uid = {1};
    m_payload_lut.clear();
    m_insert = m_lookup = {0};
    m_payloads.clear();
    m_events.clear();
    m_va_lut.clear();
  }

  inline uint64_t makeUniqueID() { return m_uid++; }

  //  Create an event with the payload information. If one already exists, the
  //  retrieve the previously added event. If not, we register the provided
  //  payload as we are seeing it for the first time. We will register all of
  //  the strings in the payload and used the string ids for generating a hash
  //  for the payload.
  //
  //  In the case the event already exists, the instance_no will return the
  //  instance ID of the event. If the event is created for the first time, the
  //  instance_id will always be 1.
  //
  //  If the string information like the name, source file etc is not available,
  //  we will use the code pointer to generate an universal id.
  //
  //  At the end of the function, the following tasks will be complete:
  //  1. Create a hash for the payload and cache it
  //  2. Create a mapping from hash <--> Universal ID
  //  3. Create a mapping from code_ptr <--> Universal ID
  //  4. Create a mapping from Universal ID <--> Payload
  //  5. Create a mapping from Universal ID <--> Event
  xpti::trace_event_data_t *create(const xpti::payload_t *p,
                                   uint64_t *instance_no) {
    return register_event(p, instance_no);
  }
  // Method to get the payload information from the event structure. This method
  // uses the Universal ID in the event structure to lookup the payload
  // information and returns the payload if available.
  //
  // This method is thread-safe
  const xpti::payload_t *payloadData(xpti::trace_event_data_t *e) {
    if (!e || e->unique_id == xpti::invalid_id)
      return nullptr;
#ifndef XPTI_USE_TBB
    std::lock_guard<std::mutex> lock(m_mutex);
#endif
    if (e->reserved.payload)
      return e->reserved.payload;
    else {
      // Cache it in case it is not already cached
      e->reserved.payload = &m_payloads[e->unique_id];
      return e->reserved.payload;
    }
  }

  const xpti::trace_event_data_t *eventData(int64_t uid) {
    if (uid == xpti::invalid_id)
      return nullptr;

#ifndef XPTI_USE_TBB
    std::lock_guard<std::mutex> lock(m_mutex);
#endif
    auto ev = m_events.find(uid);
    if (ev != m_events.end())
      return &(ev->second);
    else
      return nullptr;
  }

  // Sometimes, the user may want to add key-value pairs as metadata associated
  // with an event; this would be in addition to the source_file, line_no and
  // column_no fields that may already be present. Since we are not sure of the
  // data types, we will allow them to add these pairs as strings. Internally,
  // we will store key-value pairs as a map of string ids.
  xpti::result_t addMetadata(xpti::trace_event_data_t *e, const char *key,
                             const char *value) {
    if (!e || !key || !value)
      return xpti::result_t::XPTI_RESULT_INVALIDARG;

    string_id_t key_id = m_string_table.add(key);
    if (key_id == xpti::invalid_id) {
      return xpti::result_t::XPTI_RESULT_INVALIDARG;
    }
    string_id_t val_id = m_string_table.add(value);
    if (val_id == xpti::invalid_id) {
      return xpti::result_t::XPTI_RESULT_INVALIDARG;
    }
    // Protect simultaneous insert operations on the metadata tables
#ifdef XPTI_USE_TBB
    tbb::spin_mutex::scoped_lock hl(m_metadata_mutex);
#else
    std::lock_guard<std::mutex> lock(m_metadata_mutex);
#endif

    if (e->reserved.metadata.count(key_id)) {
      return xpti::result_t::XPTI_RESULT_DUPLICATE;
    }
    e->reserved.metadata[key_id] = val_id;
    return xpti::result_t::XPTI_RESULT_SUCCESS;
  }

  // Method to get the access statistics of the tracepoints.
  // It will print the number of insertions vs lookups that were
  // performed.
  //
  void printStatistics() {
    printf("Tracepoint inserts : [%lu] \n", m_insert.load());
    printf("Tracepoint lookups : [%lu]\n", m_lookup.load());
    printf("Tracepoint Hashmap :\n");
    m_payload_lut.printStatistics();
  }

private:
  ///  Goals: To create a hash value from payload
  ///  1. Check the payload structure to see if it is valid. If valid, then
  ///  check to see if any strings are provided and add them to the string
  ///  table.
  ///  2. Generate a payload reference using the string information, if
  ///  present or the code pointer information, otherwise
  ///  3. Add the payload and generate a unique ID
  ///  4. Cache the computed hash in the payload
  int64_t make_hash(xpti::payload_t *p) {
    // Initialize to invalid hash value
    int64_t hash = xpti::invalid_id;
    // If no flags are set, then the payload is not valid
    if (p->flags == 0)
      return hash;
    // If the hash value has been cached, return and bail early
    if (p->flags & (uint64_t)payload_flag_t::HashAvailable)
      return p->internal;

    //  Add the string information to the string table and use the string IDs
    //  (in addition to any unique addresses) to create a hash value
    if ((p->flags & (uint64_t)payload_flag_t::NameAvailable)) {
      // Add the kernel name to the string table; if the add() returns the
      // address to the string in the string table, we can avoid a query [TBD]
      p->name_sid = m_string_table.add(p->name, &p->name);
      // p->name = m_string_table.query(p->name_sid);
      if (p->flags & (uint64_t)payload_flag_t::SourceFileAvailable) {
        // Add source file information ot string table
        p->source_file_sid =
            m_string_table.add(p->source_file, &p->source_file);
        // p->source_file = m_string_table.query(p->source_file_sid);
        if (p->flags & (uint64_t)payload_flag_t::CodePointerAvailable) {
          // We have source file, kernel name info and kernel address;
          // so we combine all of them to make it unique:
          //
          // <32-bits of address bits 5-36><16-bit source_file_sid><16-bit
          // kernel name sid>
          //
          // Using the code pointer address works better than using the line
          // number and column number as the column numbers are not set in all
          // compilers that support builtin functions. If two objects are
          // declared on the same line, then the line numbers, function name,
          // source file are all the same and it would be hard to disambiguate
          // them. However, if we use the address, which would be the object
          // address, they both will have different addresses even if they
          // happen to be on the same line.
          uint16_t sname_pack = (uint16_t)(p->name_sid & 0x0000ffff);
          uint16_t sfile_pack = (uint16_t)(p->source_file_sid & 0x0000ffff);
          uint32_t kernel_sid_pack = XPTI_PACK16_RET32(sfile_pack, sname_pack);
          uint32_t addr =
              (uint32_t)(((uint64_t)p->code_ptr_va & 0x0000000ffffffff0) >> 4);
          hash = XPTI_PACK32_RET64(addr, kernel_sid_pack);
          // Cache the hash once it is computed
          p->flags |= (uint64_t)payload_flag_t::HashAvailable;
          p->internal = hash;
          return hash;
        } else {
          // We have both source file and kernel name info
          //
          // If we happen to have the line number, then we will combine all
          // three integer values (22-bits) to form a 64-bit hash. If not, we
          // will use 22 bits of the source file and kernel name ids and form a
          // 64-bit value with the middle 22-bits being zero representing the
          // line number.
          uint64_t left = 0, middle = 0, right = 0, mask22 = 0x00000000003fffff;
          // If line number info is available, extract 22-bits of it
          if (p->flags & (uint64_t)payload_flag_t::LineInfoAvailable) {
            middle = p->line_no & mask22;
            middle = middle << 22;
          }
          // The leftmost 22-bits will represent the file name string id
          left = p->source_file_sid & mask22;
          left = left << 44;
          // The rightmost 22-bits will represent the kernel name string id
          right = p->name_sid & mask22;
          hash = left | middle | right;
          p->flags |= (uint64_t)payload_flag_t::HashAvailable;
          p->internal = hash;
          return hash;
        }
      } else if (p->flags & (uint64_t)payload_flag_t::CodePointerAvailable) {
        // We have both kernel name and kernel address; we use bits 5-36 from
        // the address and combine it with the kernel name string ID
        uint32_t addr =
            (uint32_t)(((uint64_t)p->code_ptr_va & 0x0000000ffffffff0) >> 4);
        hash = XPTI_PACK32_RET64(addr, p->name_sid);
        p->flags |= (uint64_t)payload_flag_t::HashAvailable;
        p->internal = hash;
        return hash;
      } else {
        // We only have kernel name and this is suspect if the kernel names are
        // not unique and will replace any previously stored payload information
        if (p->name_sid != xpti::invalid_id) {
          hash = XPTI_PACK32_RET64(0, p->name_sid);
          p->flags |= (uint64_t)payload_flag_t::HashAvailable;
          p->internal = hash;
          return hash;
        }
      }
    } else if (p->flags & (uint64_t)payload_flag_t::CodePointerAvailable) {
      // We are only going to look at Kernel address when kernel name is not
      // available.
      hash = (uint64_t)p->code_ptr_va;
      p->flags |= (uint64_t)payload_flag_t::HashAvailable;
      p->internal = hash;
      return hash;
    }
    return hash;
  }

  // Register the payload and generate a universal ID for it.
  // Once registered, the payload is accessible through the
  // Universal ID that corresponds to the payload.
  //
  // This method is thread-safe
  //
  xpti::trace_event_data_t *register_event(const xpti::payload_t *payload,
                                           uint64_t *instance_no) {
    xpti::payload_t ptemp = *payload;
    // Initialize to invalid
    // We need an explicit lock for the rest of the operations as the same
    // payload could be registered from multiple-threads.
    //
    // 1. make_hash(p) is invariant, although the hash may be created twice and
    // written to the same field in the structure. If we have a lock guard, we
    // may be spinning and wasting time instead. We will just compute this in
    // parallel.
    // 2. m_payload_lut is queried by two threads and and both queries return
    // "not found"
    // 3. This takes both threads to the else clause both threads will create a
    // unique_id for the payload being registered and add them to the hash table
    // [with DIFFERENT IDs] and m_payloads[unique_id] gets updated twice for the
    // same payload with different IDs
    // 4. ev.unique_id is undefined as it could be one of the two IDs generated
    // for the payload
    //
    int64_t uid = xpti::invalid_id;
    //  Make a hash value from the payload. If the hash value created is
    //  invalid, return immediately
    int64_t hash = make_hash(&ptemp);
    if (hash == xpti::invalid_id)
      return nullptr;
      // If it's valid, we check to see if we can retrieve the previously added
      // event structure; we do this as a critical section
#ifdef XPTI_USE_TBB
    tbb::speculative_spin_mutex::scoped_lock hl(m_hash_lock);
#else
    std::lock_guard<std::mutex> lock(m_hash_lock);
#endif
    uid = m_payload_lut.find(hash);
    if (uid != xpti::invalid_id) {
#ifdef XPTI_STATISTICS
      m_lookup++;
#endif
#ifndef XPTI_USE_TBB
      std::lock_guard<std::mutex> lock(m_mutex);
#endif
      auto ev = m_events.find(uid);
      if (ev != m_events.end()) {
        ev->second.instance_id++;
        // Guarantees that the returned instance ID will be accurate as
        // it is on the stack
        if (instance_no)
          *instance_no = ev->second.instance_id;
        return &(ev->second);
      } else
        return nullptr; // we have a problem!
    } else {
#ifdef XPTI_STATISTICS
      m_insert++;
#endif
      // Create a new unique ID
      //
      uid = m_uid++;
      // And add it as a pair
      //
      m_payload_lut.add(hash, uid);
      // The API allows you to query a Universal ID from the kernel address; so
      // build the necessary data structures for this.
      if (ptemp.flags & (uint64_t)payload_flag_t::HashAvailable) {
#ifndef XPTI_USE_TBB
        std::lock_guard<std::mutex> lock(m_va_mutex);
#endif
        m_va_lut[(uint64_t)ptemp.code_ptr_va] = uid;
      }
      // We also want to query the payload by universal ID that has been
      // generated
#ifndef XPTI_USE_TBB
      std::lock_guard<std::mutex> lock(m_mutex);
#endif
      m_payloads[uid] = ptemp; // when it uses tbb, should be thread-safe
      {
        xpti::trace_event_data_t *ev = &m_events[uid];
        // We are seeing this unique ID for the first time, so we will
        // initialize the event structure with defaults and set the unique_id to
        // the newly generated unique id (uid)
        ev->unique_id = uid;
        ev->unused = 0;
        ev->reserved.payload = &m_payloads[uid];
        ev->data_id = ev->source_id = ev->target_id = 0;
        ev->instance_id = 1;
        ev->user_data = nullptr;
        ev->event_type = (uint16_t)xpti::trace_event_type_t::unknown_event;
        ev->activity_type =
            (uint16_t)xpti::trace_activity_type_t::unknown_activity;
        *instance_no = ev->instance_id;
        return ev;
      }
    }
    return nullptr;
  }

  xpti::safe_int64_t m_uid;
  xpti::Hash64x64Table m_payload_lut;
  xpti::StringTable &m_string_table;
  xpti::safe_uint64_t m_insert, m_lookup;
  uid_payload_t m_payloads;
  uid_event_t m_events;
  va_uid_t m_va_lut;
#ifdef XPTI_USE_TBB
  tbb::spin_mutex m_metadata_mutex;
  tbb::speculative_spin_mutex m_hash_lock;
#else
  std::mutex m_metadata_mutex;
  std::mutex m_hash_lock;
  std::mutex m_mutex;
  std::mutex m_va_mutex;
#endif
};

/// \brief Helper class to manage subscriber callbacks for a given tracepoint
/// \details This class provides a thread-safe way to register and unregister
/// callbacks for a given stream. This will be used by tool plugins.
///
/// The class also provided a way to notify registered callbacks for a given
/// stream and trace point type. This will be used by framework to trigger
/// notifications are instrumentation points.
///
class Notifications {
public:
  using cb_entry_t = std::pair<bool, xpti::tracepoint_callback_api_t>;
#ifdef XPTI_USE_TBB
  using cb_entries_t = tbb::concurrent_vector<cb_entry_t>;
  using cb_t = tbb::concurrent_hash_map<uint16_t, cb_entries_t>;
  using stream_cb_t = tbb::concurrent_unordered_map<uint16_t, cb_t>;
  using statistics_t = tbb::concurrent_unordered_map<uint16_t, uint64_t>;
#else
  using cb_entries_t = std::vector<cb_entry_t>;
  using cb_t = std::unordered_map<uint16_t, cb_entries_t>;
  using stream_cb_t = std::unordered_map<uint16_t, cb_t>;
  using statistics_t = std::unordered_map<uint16_t, uint64_t>;
#endif
  Notifications() = default;
  ~Notifications() = default;

  xpti::result_t registerCallback(uint8_t stream_id, uint16_t trace_type,
                                  xpti::tracepoint_callback_api_t cb) {
    if (!cb)
      return xpti::result_t::XPTI_RESULT_INVALIDARG;

#ifdef XPTI_STATISTICS
    //  Initialize first encountered trace
    //  type statistics counters
    {
#ifdef XPTI_USE_TBB
      tbb::spin_mutex::scoped_lock sl(m_stats_lock);
#else
      std::lock_guard<std::mutex> lock(m_stats_lock);
#endif
      auto instance = m_stats.find(trace_type);
      if (instance == m_stats.end()) {
        m_stats[trace_type] = 0;
      }
    }
#endif
#ifndef XPTI_USE_TBB
    std::lock_guard<std::mutex> lock(m_cb_lock);
#endif
    auto &stream_cbs = m_cbs[stream_id]; // thread-safe
                                         // What we get is a concurrent_hash_map
                                         // of vectors holding the callbacks we
                                         // need access to;
#ifdef XPTI_USE_TBB
    cb_t::accessor a;
    stream_cbs.insert(a, trace_type);
#else
    auto a = stream_cbs.find(trace_type);
    if (a == stream_cbs.end()) {
      auto b = stream_cbs[trace_type];
      a = stream_cbs.find(trace_type);
    }
#endif
    // If the key does not exist, a new entry is created and an accessor to it
    // is returned. If it exists, we have access to the previous entry.
    //
    // Before we add this element, we scan all existing elements to see if it
    // has already been registered. If so, we return XPTI_RESULT_DUPLICATE.
    //
    // If not, we set the first element of new entry to 'true' indicating that
    // it is valid. Unregister will just set this flag to false, indicating that
    // it is no longer valid and is unregistered.
    for (auto &e : a->second) {
      if (e.second == cb) {
        if (e.first) // Already here and active
          return xpti::result_t::XPTI_RESULT_DUPLICATE;
        else { // it has been unregistered before, re-enable
          e.first = true;
          return xpti::result_t::XPTI_RESULT_UNDELETE;
        }
      }
    }
    // If we come here, then we did not find the callback being registered
    // already in the framework. So, we insert it.
    a->second.push_back(std::make_pair(true, cb));
    return xpti::result_t::XPTI_RESULT_SUCCESS;
  }

  xpti::result_t unregisterCallback(uint8_t stream_id, uint16_t trace_type,
                                    xpti::tracepoint_callback_api_t cb) {
    if (!cb)
      return xpti::result_t::XPTI_RESULT_INVALIDARG;

#ifndef XPTI_USE_TBB
    std::lock_guard<std::mutex> lock(m_cb_lock);
#endif
    auto &stream_cbs =
        m_cbs[stream_id]; // thread-safe
                          //  What we get is a concurrent_hash_map of
                          //  vectors holding the callbacks we need
                          //  access to;
#ifdef XPTI_USE_TBB
    cb_t::accessor a;
    bool success = stream_cbs.find(a, trace_type);
#else
    auto a = stream_cbs.find(trace_type);
    bool success = (a != stream_cbs.end());
#endif
    if (success) {
      for (auto &e : a->second) {
        if (e.second == cb) {
          if (e.first) { // Already here and active
                         // unregister, since delete and simultaneous
                         // iterations by other threads are unsafe
            e.first = false;
            // releases the accessor
            return xpti::result_t::XPTI_RESULT_SUCCESS;
          } else {
            // releases the accessor
            return xpti::result_t::XPTI_RESULT_DUPLICATE;
          }
        }
      }
    }
    //  Not here, so nothing to unregister
    return xpti::result_t::XPTI_RESULT_NOTFOUND;
  }

  xpti::result_t unregisterStream(uint8_t stream_id) {
    // If there are no callbacks registered for the requested stream ID, we
    // return not found
#ifndef XPTI_USE_TBB
    std::lock_guard<std::mutex> lock(m_cb_lock);
#endif
    if (m_cbs.count(stream_id) == 0)
      return xpti::result_t::XPTI_RESULT_NOTFOUND;

    auto &stream_cbs = m_cbs[stream_id]; // thread-safe
    // Disable all callbacks registered for the stream represented by stream_id
    for (auto &it : stream_cbs) {
      for (auto &ele : it.second) {
        ele.first = false;
      }
    }
    //  Return success
    return xpti::result_t::XPTI_RESULT_SUCCESS;
  }

  xpti::result_t notifySubscribers(uint16_t stream_id, uint16_t trace_type,
                                   xpti::trace_event_data_t *parent,
                                   xpti::trace_event_data_t *object,
                                   uint64_t instance, const void *user_data) {
    {
#ifndef XPTI_USE_TBB
      std::lock_guard<std::mutex> lock(m_cb_lock);
#endif
      cb_t &stream = m_cbs[stream_id]; // Thread-safe
#ifdef XPTI_USE_TBB
      cb_t::const_accessor a; // read-only accessor
      bool success = stream.find(a, trace_type);
#else
      auto a = stream.find(trace_type);
      bool success = (a != stream.end());
#endif

      if (success) {
        // Go through all registered callbacks and invoke them
        for (auto &e : a->second) {
          if (e.first)
            (e.second)(trace_type, parent, object, instance, user_data);
        }
      }
    }
#ifdef XPTI_STATISTICS
    auto &counter = m_stats[trace_type];
    {
#ifdef XPTI_USE_TBB
      tbb::spin_mutex::scoped_lock sl(m_stats_lock);
#else
      std::lock_guard<std::mutex> lock(m_stats_lock);
#endif
      counter++;
    }
#endif
    return xpti::result_t::XPTI_RESULT_SUCCESS;
  }

  void printStatistics() {
#ifdef XPTI_STATISTICS
    printf("Notification statistics:\n");
    for (auto &s : m_stats) {
      printf("%19s: [%llu] \n",
             stringify_trace_type((xpti_trace_point_type_t)s.first).c_str(),
             s.second);
    }
#endif
  }

private:
#ifdef XPTI_STATISTICS
  std::string stringify_trace_type(xpti_trace_point_type_t trace_type) {
    switch (trace_type) {
    case graph_create:
      return "graph_create";
    case node_create:
      return "node_create";
    case edge_create:
      return "edge_create";
    case region_begin:
      return "region_begin";
    case region_end:
      return "region_end";
    case task_begin:
      return "task_begin";
    case task_end:
      return "task_end";
    case barrier_begin:
      return "barrier_begin";
    case barrier_end:
      return "barrier_end";
    case lock_begin:
      return "lock_begin";
    case lock_end:
      return "lock_end";
    case signal:
      return "signal";
    case transfer_begin:
      return "transfer_begin";
    case transfer_end:
      return "transfer_end";
    case thread_begin:
      return "thread_begin";
    case thread_end:
      return "thread_end";
    case wait_begin:
      return "wait_begin";
    case wait_end:
      return "wait_end";
      break;
    default:
      if (trace_type & user_defined_trace_point) {
        std::string str =
            "user_defined[" +
            std::to_string(XPTI_EXTRACT_USER_DEFINED_ID(trace_type)) + "]";
        return str;
      } else {
        std::string str =
            "unknown[" +
            std::to_string(XPTI_EXTRACT_USER_DEFINED_ID(trace_type)) + "]";
        return str;
      }
    }
  }
#endif
  stream_cb_t m_cbs;
#ifdef XPTI_USE_TBB
  tbb::spin_mutex m_stats_lock;
#else
  std::mutex m_cb_lock;
  std::mutex m_stats_lock;
#endif
  statistics_t m_stats;
};

class Framework {
public:
  Framework()
      : m_tracepoints(m_string_table), m_universal_ids(0),
        m_trace_enabled(false) {
    //  Load all subscribers on construction
    m_subscribers.loadFromEnvironmentVariable();
    m_trace_enabled =
        (g_helper.checkTraceEnv() && m_subscribers.hasValidSubscribers());
  }
  ~Framework() = default;

  void clear() {
    m_universal_ids = {1};
    m_tracepoints.clear();
    m_string_table.clear();
  }

  inline void setTraceEnabled(bool yesOrNo = true) {
    m_trace_enabled = yesOrNo;
  }

  inline bool traceEnabled() { return m_trace_enabled; }

  inline uint64_t makeUniqueID() { return m_tracepoints.makeUniqueID(); }

  xpti::result_t addMetadata(xpti::trace_event_data_t *e, const char *key,
                             const char *value) {
    return m_tracepoints.addMetadata(e, key, value);
  }

  xpti::trace_event_data_t *
  createEvent(const xpti::payload_t *payload, uint16_t event_type,
              xpti::trace_activity_type_t activity_type,
              uint64_t *instance_no) {
    if (!payload || !instance_no)
      return nullptr;

    if (payload->flags == 0)
      return nullptr;

    xpti::trace_event_data_t *e = m_tracepoints.create(payload, instance_no);

    // Event is not managed by anyone. The unique_id that is a part of the event
    // structure can be used to determine the payload that forms the event. The
    // attribute 'ev.user_data' and 'ev.reserved' can be used to store user
    // defined and system defined data respectively. Currently the 'reserved'
    // field is not used, but object lifetime management must be employed once
    // this is active.
    //
    // On the other hand, the 'user_data' field is for user data and should be
    // managed by the user code. The framework will NOT free any memory
    // allocated to this pointer
    e->event_type = event_type;
    e->activity_type = (uint16_t)activity_type;
    return e;
  }

  inline const xpti::trace_event_data_t *findEvent(int64_t universal_id) {
    return m_tracepoints.eventData(universal_id);
  }

  xpti::result_t initializeStream(const char *stream, uint32_t major_revision,
                                  uint32_t minor_revision,
                                  const char *version_string) {
    if (!stream || !version_string)
      return xpti::result_t::XPTI_RESULT_INVALIDARG;

    m_subscribers.initializeForStream(stream, major_revision, minor_revision,
                                      version_string);
    return xpti::result_t::XPTI_RESULT_SUCCESS;
  }

  uint8_t registerStream(const char *stream_name) {
    return (uint8_t)m_stream_string_table.add(stream_name);
  }

  void closeAllStreams() {
    auto table = m_stream_string_table.table();
    StringTable::st_reverse_t::iterator it;
    for (it = table.begin(); it != table.end(); ++it) {
      xptiFinalize(it->second);
    }
  }

  xpti::result_t unregisterStream(const char *stream_name) {
    return finalizeStream(stream_name);
  }

  uint8_t registerVendor(const char *stream_name) {
    return (uint8_t)m_vendor_string_table.add(stream_name);
  }

  string_id_t registerString(const char *string, char **table_string) {
    if (!table_string || !string)
      return xpti::invalid_id;

    *table_string = 0;

    const char *ref_str;
    auto id = m_string_table.add(string, &ref_str);
    *table_string = const_cast<char *>(ref_str);

    return id;
  }

  const char *lookupString(string_id_t id) {
    if (id < 0)
      return nullptr;
    return m_string_table.query(id);
  }

  xpti::result_t registerCallback(uint8_t stream_id, uint16_t trace_type,
                                  xpti::tracepoint_callback_api_t cb) {
    return m_notifier.registerCallback(stream_id, trace_type, cb);
  }

  xpti::result_t unregisterCallback(uint8_t stream_id, uint16_t trace_type,
                                    xpti::tracepoint_callback_api_t cb) {
    return m_notifier.unregisterCallback(stream_id, trace_type, cb);
  }

  xpti::result_t notifySubscribers(uint8_t stream_id, uint16_t trace_type,
                                   xpti::trace_event_data_t *parent,
                                   xpti::trace_event_data_t *object,
                                   uint64_t instance, const void *user_data) {
    if (!m_trace_enabled)
      return xpti::result_t::XPTI_RESULT_FALSE;
    if (!object)
      return xpti::result_t::XPTI_RESULT_INVALIDARG;
    //
    //  Notify all subscribers for the stream 'stream_id'
    //
    return m_notifier.notifySubscribers(stream_id, trace_type, parent, object,
                                        instance, user_data);
  }

  bool hasSubscribers() { return m_subscribers.hasValidSubscribers(); }

  xpti::result_t finalizeStream(const char *stream) {
    if (!stream)
      return xpti::result_t::XPTI_RESULT_INVALIDARG;
    m_subscribers.finalizeForStream(stream);
    return m_notifier.unregisterStream(m_stream_string_table.add(stream));
  }

  const xpti::payload_t *queryPayload(xpti::trace_event_data_t *e) {
    return m_tracepoints.payloadData(e);
  }

  void printStatistics() {
    m_notifier.printStatistics();
    m_string_table.printStatistics();
    m_tracepoints.printStatistics();
  }

private:
  /// Thread-safe counter used for generating universal IDs
  xpti::safe_uint64_t m_universal_ids;
  /// Manages loading the subscribers and calling their init() functions
  xpti::Subscribers m_subscribers;
  /// Used to send event notification to subscribers
  xpti::Notifications m_notifier;
  /// Thread-safe string table
  xpti::StringTable m_string_table;
  /// Thread-safe string table, used for stream IDs
  xpti::StringTable m_stream_string_table;
  /// Thread-safe string table, used for vendor IDs
  xpti::StringTable m_vendor_string_table;
  /// Manages the tracepoints - framework caching
  xpti::Tracepoints m_tracepoints;
  /// Flag indicates whether tracing should be enabled
  bool m_trace_enabled;
};

static Framework g_framework;
} // namespace xpti

extern "C" {
XPTI_EXPORT_API uint16_t xptiRegisterUserDefinedTracePoint(
    const char *tool_name, uint8_t user_defined_tp) {
  uint8_t tool_id = xpti::g_framework.registerVendor(tool_name);
  user_defined_tp |= (uint8_t)xpti::trace_point_type_t::user_defined;
  uint16_t usr_def_tp = XPTI_PACK08_RET16(tool_id, user_defined_tp);

  return usr_def_tp;
}

XPTI_EXPORT_API uint16_t xptiRegisterUserDefinedEventType(
    const char *tool_name, uint8_t user_defined_event) {
  uint8_t tool_id = xpti::g_framework.registerVendor(tool_name);
  user_defined_event |= (uint8_t)xpti::trace_event_type_t::user_defined;
  uint16_t usr_def_ev = XPTI_PACK08_RET16(tool_id, user_defined_event);
  return usr_def_ev;
}

XPTI_EXPORT_API xpti::result_t xptiInitialize(const char *stream, uint32_t maj,
                                              uint32_t min,
                                              const char *version) {
  return xpti::g_framework.initializeStream(stream, maj, min, version);
}

XPTI_EXPORT_API void xptiFinalize(const char *stream) {
  xpti::g_framework.finalizeStream(stream);
}

XPTI_EXPORT_API uint64_t xptiGetUniqueId() {
  return xpti::g_framework.makeUniqueID();
}

XPTI_EXPORT_API xpti::string_id_t xptiRegisterString(const char *string,
                                                     char **table_string) {
  return xpti::g_framework.registerString(string, table_string);
}

XPTI_EXPORT_API const char *xptiLookupString(xpti::string_id_t id) {
  return xpti::g_framework.lookupString(id);
}

XPTI_EXPORT_API uint8_t xptiRegisterStream(const char *stream_name) {
  return xpti::g_framework.registerStream(stream_name);
}

XPTI_EXPORT_API xpti::result_t xptiUnregisterStream(const char *stream_name) {
  return xpti::g_framework.unregisterStream(stream_name);
}
XPTI_EXPORT_API xpti::trace_event_data_t *
xptiMakeEvent(const char *name, xpti::payload_t *payload, uint16_t event,
              xpti::trace_activity_type_t activity, uint64_t *instance_no) {
  return xpti::g_framework.createEvent(payload, event, activity, instance_no);
}

XPTI_EXPORT_API void xptiReset() { xpti::g_framework.clear(); }

XPTI_EXPORT_API const xpti::trace_event_data_t *xptiFindEvent(int64_t uid) {
  return xpti::g_framework.findEvent(uid);
}

XPTI_EXPORT_API const xpti::payload_t *
xptiQueryPayload(xpti::trace_event_data_t *lookup_object) {
  return xpti::g_framework.queryPayload(lookup_object);
}

XPTI_EXPORT_API xpti::result_t
xptiRegisterCallback(uint8_t stream_id, uint16_t trace_type,
                     xpti::tracepoint_callback_api_t cb) {
  return xpti::g_framework.registerCallback(stream_id, trace_type, cb);
}

XPTI_EXPORT_API xpti::result_t
xptiUnregisterCallback(uint8_t stream_id, uint16_t trace_type,
                       xpti::tracepoint_callback_api_t cb) {
  return xpti::g_framework.unregisterCallback(stream_id, trace_type, cb);
}

XPTI_EXPORT_API xpti::result_t
xptiNotifySubscribers(uint8_t stream_id, uint16_t trace_type,
                      xpti::trace_event_data_t *parent,
                      xpti::trace_event_data_t *object, uint64_t instance,
                      const void *temporal_user_data) {
  return xpti::g_framework.notifySubscribers(
      stream_id, trace_type, parent, object, instance, temporal_user_data);
}

XPTI_EXPORT_API bool xptiTraceEnabled() {
  return xpti::g_framework.traceEnabled();
}

XPTI_EXPORT_API xpti::result_t xptiAddMetadata(xpti::trace_event_data_t *e,
                                               const char *key,
                                               const char *value) {
  return xpti::g_framework.addMetadata(e, key, value);
}

XPTI_EXPORT_API xpti::metadata_t *
xptiQueryMetadata(xpti::trace_event_data_t *e) {
  return &e->reserved.metadata;
}

XPTI_EXPORT_API void xptiForceSetTraceEnabled(bool yesOrNo) {
  xpti::g_framework.setTraceEnabled(yesOrNo);
}
} // extern "C"

#if (defined(_WIN32) || defined(_WIN64))

#include <string>
#include <windows.h>

BOOL WINAPI DllMain(HINSTANCE hinstDLL, DWORD fwdReason, LPVOID lpvReserved) {
  switch (fwdReason) {
  case DLL_PROCESS_ATTACH:
    break;
  case DLL_PROCESS_DETACH:
    //
    //  We cannot unload all subscribers here...
    //
#ifdef XPTI_STATISTICS
    __g_framework.printStatistics();
#endif
    break;
  }

  return TRUE;
}

#else // Linux (possibly macOS?)

__attribute__((constructor)) static void framework_init() {}

__attribute__((destructor)) static void framework_fini() {
#ifdef XPTI_STATISTICS
  __g_framework.printStatistics();
#endif
}

#endif
