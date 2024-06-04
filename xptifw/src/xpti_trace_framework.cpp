//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
#include "xpti/xpti_trace_framework.hpp"
#include "xpti_int64_hash_table.hpp"
#include "xpti_object_table.hpp"
#include "xpti_string_table.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <cstdio>
#include <iostream>
#include <map>
#include <mutex>
#include <sstream>
#include <string>
#include <string_view>
#include <tuple>
#include <unordered_map>
#include <vector>

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

static_assert(std::is_trivially_destructible<xpti::utils::SpinLock>::value,
              "SpinLock is not trivial");
static_assert(
    std::is_trivially_destructible<xpti::utils::PlatformHelper>::value,
    "PlatformHelper is not trivial");

// TLS variables to support stashing tupples and universal IDs
/// @file xpti_trace_framework.cpp
/// @brief This file contains the implementation of the XPTI trace framework.

/// @brief Alias for a tuple that is used to stash a key-value pair.
using stash_tuple_t = std::tuple<const char *, uint64_t>;

/// @brief A thread-local variable of type stash_tuple_t, default initialized.
/// This variable is used to stash a key-value pair/thread for use within a
/// scope.
static thread_local stash_tuple_t g_tls_stash_tuple = stash_tuple_t(nullptr, 0);

/// @brief A thread-local variable of type xpti::uid_t, default initialized.
/// This variable is used to store a unique identifier within a program scope
/// for each thread.
static thread_local xpti::uid_t g_tls_uid = xpti::uid_t();

/// @brief A TLS of type xpti::trace_point_data_t, default initialized.
/// This variable is used to store trace point data generated from code location
/// information in a function scope for all top level entry points recorded for
/// each thread.
static thread_local xpti::trace_point_data_t g_tls_tracepoint_scope_data =
    xpti::trace_point_data_t();

namespace xpti {
/// @var env_subscribers
/// @brief A constant character pointer initialized with the string
/// "XPTI_SUBSCRIBERS". This variable represents the environment variable name
/// for subscribers in the XPTI trace framework.
constexpr const char *env_subscribers = "XPTI_SUBSCRIBERS";
/// @var g_helper
/// @brief An instance of the PlatformHelper class from the xpti::utils
/// namespace. This instance is used to perform platform-specific operations.
xpti::utils::PlatformHelper g_helper;

/// @var g_framework_mutex
/// @brief An instance of the SpinLock class from the xpti::utils namespace.
/// This instance is used to ensure thread-safety in the XPTI trace framework.
xpti::utils::SpinLock g_framework_mutex;

/// @var g_default_stream_id
/// @brief A uint8_t variable initialized with 0.
/// This variable represents the default stream ID in the XPTI trace framework.
uint8_t g_default_stream_id = 0;

/// @var g_default_event_type
/// @brief A variable of type xpti::trace_event_type_t, initialized with
/// xpti::trace_event_type_t::algorithm. This variable represents the default
/// event type in the XPTI trace framework.
xpti::trace_event_type_t g_default_event_type =
    xpti::trace_event_type_t::algorithm;

/// @var g_default_trace_type
/// @brief A variable of type xpti::trace_point_type_t, initialized with
/// xpti::trace_point_type_t::function_begin. This variable represents the
/// default trace point type in the XPTI trace framework.
xpti::trace_point_type_t g_default_trace_type =
    xpti::trace_point_type_t::function_begin;

// This class is a helper class to load all the listed subscribers
// provided by the user in XPTI_SUBSCRIBERS environment variable.
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
  plugin_data_t queryPlugin(xpti_plugin_handle_t Handle) {
    plugin_data_t PData;
    {
      std::lock_guard<std::mutex> Lock(MMutex);
      if (MHandleLUT.count(Handle))
        return MHandleLUT[Handle];
      else
        return PData; // return invalid plugin data
    }
  }

  // Load the provided shared object file name using the explicit load API. If
  // the load is successful, a test is performed to see if the shared object has
  // the required entry points for it to be considered a trace plugin
  // subscriber. If so, the internal data structures are updated and a valid
  // handle is returned.
  //
  // If not, the shared object is unloaded and a NULL handle is returned.
  xpti_plugin_handle_t loadPlugin(const char *Path) {
    xpti_plugin_handle_t Handle = 0;
    std::string Error;
    // Check to see if the subscriber has already been loaded; if so, return the
    // handle from the previously loaded library
    if (MNameLUT.count(Path)) {
      {
        std::lock_guard<std::mutex> Lock(MMutex);
        // This plugin has already been loaded, so let's return previously
        // recorded handle
        plugin_data_t &Data = MNameLUT[Path];
        assert(Data.valid && "Lookup is invalid!");
        if (Data.valid)
          return Data.handle;
      }
    }

    Handle = g_helper.loadLibrary(Path, Error);
    if (Handle) {
      // The tracing framework requires the tool plugins to implement the
      // xptiTraceInit() and xptiTraceFinish() functions. If these are not
      // present, then the plugin will be ruled an invalid plugin and unloaded
      // from the process.
      xpti::plugin_init_t InitFunc = reinterpret_cast<xpti::plugin_init_t>(
          g_helper.findFunction(Handle, "xptiTraceInit"));
      xpti::plugin_fini_t FiniFunc = reinterpret_cast<xpti::plugin_fini_t>(
          g_helper.findFunction(Handle, "xptiTraceFinish"));
      if (InitFunc && FiniFunc) {
        //  We appear to have loaded a valid plugin, so we will insert the
        //  plugin information into the two maps guarded by a lock
        plugin_data_t Data;
        Data.valid = true;
        Data.handle = Handle;
        Data.name = Path;
        Data.init = InitFunc;
        Data.fini = FiniFunc;
        {
          std::lock_guard<std::mutex> Lock(MMutex);
          MNameLUT[Path] = Data;
          MHandleLUT[Handle] = Data;
        }
      } else {
        // We may have loaded another shared object that is not a tool plugin
        // for the tracing framework, so we'll unload it now
        unloadPlugin(Handle);
        Handle = nullptr;
      }
    } else {
      //  Get error from errno
      if (!Error.empty())
        std::cout << '[' << Path << "]: " << Error << '\n';
    }
    return Handle;
  }

  //  Unloads the shared object identified by the handle provided. If
  //  successful, returns a success code, else a failure code.
  xpti::result_t unloadPlugin(xpti_plugin_handle_t PluginHandle) {
    xpti::result_t Res = g_helper.unloadLibrary(PluginHandle);
    if (xpti::result_t::XPTI_RESULT_SUCCESS == Res) {
      auto Loc = MHandleLUT.find(PluginHandle);
      if (Loc != MHandleLUT.end()) {
        MHandleLUT.erase(PluginHandle);
      }
    }
    return Res;
  }

  // Quick test to see if there are registered subscribers
  bool hasValidSubscribers() { return (MHandleLUT.size() > 0); }

  void initializeForStream(const char *Stream, uint32_t major_revision,
                           uint32_t minor_revision,
                           const char *version_string) {
    //  If there are subscribers registered, then initialize the subscribers
    //  with the new stream information.
    if (MHandleLUT.size()) {
      for (auto &Handle : MHandleLUT) {
        Handle.second.init(major_revision, minor_revision, version_string,
                           Stream);
      }
    }
  }

  void finalizeForStream(const char *Stream) {
    //  If there are subscribers registered, then finalize the subscribers for
    //  the stream
    if (MHandleLUT.size()) {
      for (auto &Handle : MHandleLUT) {
        Handle.second.fini(Stream);
      }
    }
  }

  void loadFromEnvironmentVariable() {
    if (!g_helper.checkTraceEnv())
      return;
    //  Load all registered Listeners by scanning the environment variable in
    //  "Env"; The environment variable, if set, extract the comma separated
    //  tokens into a vector.
    std::string Token, Env = g_helper.getEnvironmentVariable(env_subscribers);
    std::vector<std::string> Listeners;
    std::stringstream Stream(Env);

    //  Split the environment variable value by ',' and build a vector of the
    //  tokens (subscribers)
    while (std::getline(Stream, Token, ',')) {
      Listeners.push_back(Token);
    }

    size_t ValidSubscribers = Listeners.size();
    if (ValidSubscribers) {
      //  Let's go through the subscribers and load these plugins;
      for (auto &Path : Listeners) {
        // Load the plugins listed in the environment variable
        {
          std::lock_guard<std::mutex> Lock(MLoader);
          auto SubscriberHandle = loadPlugin(Path.c_str());
          if (!SubscriberHandle) {
            ValidSubscribers--;
            std::cout << "Failed to load " << Path << " successfully\n";
          }
        }
      }
    }
  }

  void unloadAllPlugins() {
    for (auto &Item : MNameLUT) {
      unloadPlugin(Item.second.handle);
    }
    MHandleLUT.clear();
    MNameLUT.clear();
  }

private:
  /// Hash map that maps shared object name to the plugin data
  plugin_name_lut_t MNameLUT;
  /// Hash map that maps shared object handle to the plugin data
  plugin_handle_lut_t MHandleLUT;
  /// Lock to ensure the operation on these maps are safe
  std::mutex MMutex;
  /// Lock to ensure that only one load happens at a time
  std::mutex MLoader;
};

/// \brief Helper class to create and  manage tracepoints
/// \details The class uses the global string table to register the strings it
/// encounters in various payloads and builds internal hash maps to manage them.
/// This is a single point for managing tracepoints.
class Tracepoints {
public:
  /// @typedef uid_entry_t
  /// @brief A type alias for a pair containing an xpti::payload_t and an
  /// xpti::safe_uint64_t. This type is used to store a payload and a
  /// thread-safe unsigned 64-bit integer that tracks the instance.
  using uid_entry_t = std::pair<xpti::payload_t, xpti::safe_uint64_t>;

  /// @typedef uid_payload_lut
  /// @brief A type alias for an unordered map from uid_t to uid_entry_t.
  /// This type is used to create a lookup table (lut) that maps unique
  /// identifiers (uids) to their corresponding payload entries.
  using uid_payload_lut = std::unordered_map<uid_t, uid_entry_t>;

  /// @typedef trace_instance_t
  /// @brief A type alias for an unordered map from uint64_t to
  /// xpti::trace_event_data_t. This type is used to map unique identifiers to
  /// their corresponding trace event data vectors that records instance data.
  using trace_instance_t =
      std::unordered_map<uint64_t, xpti::trace_event_data_t>;

  /// @typedef uid_event_lut
  /// @brief A type alias for an unordered map from uid_t to trace_instance_t.
  /// This type is used to create a lookup table (lut) that maps unique
  /// identifiers (uids) to their corresponding trace instances.
  using uid_event_lut = std::unordered_map<uid_t, trace_instance_t>;

  /// @typedef uid_payload_t
  /// @brief A type alias for a pair containing a uid_t and a pointer to an
  /// xpti::payload_t. This type is used to store a unique identifier and a
  /// pointer to its corresponding payload together.
  using uid_payload_t = std::pair<uid_t, xpti::payload_t *>;

  Tracepoints(xpti::StringTable &st)
      : MUId(1), MStringTableRef(st), MInsertions(0), MRetrievals(0) {
    // Nothing requires to be done at construction time
  }

  ~Tracepoints() { clear(); }

  void clear() {
    MStringTableRef.clear();
    // We will always start our ID
    // stream from 1. 0 is null_id
    // and -1 is invalid_id
    MUId = 1;
    MInsertions = MRetrievals = 0;
    MPayloads.clear();
    MEvents.clear();
  }

  inline uint64_t makeUniqueID() { return MUId++; }

  /// @brief Creates/Registers a trace event data object for a given payload.
  ///
  /// This function takes a pointer to a payload and creates a trace event data
  /// object for it. It also updates the instance number for the event.
  ///
  /// @param Payload Pointer to the payload for which the trace event data
  /// object is to be created.
  /// @param InstanceNo Pointer to the instance number to be updated.
  /// @return Pointer to the created trace event data object.
  ///
  xpti::trace_event_data_t *create(const xpti::payload_t *Payload,
                                   uint64_t *InstanceNo) {
    if (!InstanceNo)
      return nullptr;
    auto Event = register_event(Payload);
    // Extract the instance information from the UID reference in the event
    if (Event)
      *InstanceNo = Event->uid.instance;
    return Event;
  }
  // Method to get the payload information from the event structure. This method
  // uses the Universal ID in the event structure to lookup the payload
  // information and returns the payload if available.
  //
  // This method is thread-safe
  const xpti::payload_t *payloadData(xpti::trace_event_data_t *Event) {
    if (!Event || !Event->uid.isValid())
      return nullptr;
    // Scoped lock until the information is retrieved from the map
    {
      std::lock_guard<std::mutex> Lock(MEventMutex);
      if (Event->reserved.payload)
        return Event->reserved.payload;
      else {
        // Cache it in case it is not already cached
        Event->reserved.payload = &MPayloads[Event->uid].first;
        return Event->reserved.payload;
      }
    }
  }

  const xpti::payload_t *payloadDataByUID(xpti::uid_t &uid) {
    if (!uid.isValid())
      return nullptr;
    // Scoped lock until the information is retrieved from the map
    {
      std::lock_guard<std::mutex> Lock(MEventMutex);
      // Return the payload information only
      return &MPayloads[uid].first;
    }
  }

  const xpti::trace_event_data_t *eventData(xpti::uid_t &UId) {
    if (!UId.isValid())
      return nullptr;

    std::lock_guard<std::mutex> Lock(MEventMutex);
    auto EvLoc = MEvents.find(UId);
    if (EvLoc != MEvents.end()) {
      auto EvInstanceLoc = EvLoc->second.find(UId.instance);
      if (EvInstanceLoc != EvLoc->second.end()) {
        return &(EvInstanceLoc->second);
      } else {
        return nullptr;
      }
    } else {
      return nullptr;
    }
  }

  /// @brief Releases a trace event.
  ///
  /// This function takes a pointer to a trace event data object and releases
  /// it. It removes the event from the event map
  ///
  /// @param Event Pointer to the trace event data object to be released.
  ///
  void releaseEvent(xpti::trace_event_data_t *Event) {
    if (!Event)
      return;
    std::lock_guard<std::mutex> Lock(MEventMutex);
    // Find the event list for a given UID
    auto EvLoc = MEvents.find(Event->uid);
    if (EvLoc != MEvents.end()) {
      // Now release the event associated with the UID instance
      EvLoc->second.erase(Event->uid.instance);
      // If there are no more events associated with the UID, we can release the
      // Payload as well, but we will not as the same payload may be revisited
      // and we need to keep the instance count going
    }
  }

  // Sometimes, the user may want to add key-value pairs as metadata associated
  // with an event; this would be in addition to the source_file, line_no and
  // column_no fields that may already be present. Since we are not sure of the
  // data types, we will allow them to add these pairs as strings. Internally,
  // we will store key-value pairs as a map of string ids.
  xpti::result_t addMetadata(xpti::trace_event_data_t *Event, const char *Key,
                             object_id_t ValueID) {
    if (!Event || !Key)
      return xpti::result_t::XPTI_RESULT_INVALIDARG;

    string_id_t KeyID = MStringTableRef.add(Key);
    if (KeyID == xpti::invalid_id) {
      return xpti::result_t::XPTI_RESULT_INVALIDARG;
    }

    // Protect simultaneous insert operations on the metadata tables
    {
      xpti::result_t res;
      std::lock_guard<std::mutex> HashLock(MMetadataMutex);
      if (Event->reserved.metadata.count(KeyID)) {
        // One already existed, but we overwrote it
        res = xpti::result_t::XPTI_RESULT_DUPLICATE;
      } else {
        res = xpti::result_t::XPTI_RESULT_SUCCESS;
      }
      Event->reserved.metadata[KeyID] = ValueID;
      return res;
    }
  }

  // Method to get the access statistics of the tracepoints.
  // It will print the number of insertions vs lookups that were
  // performed.
  //
  void printStatistics() {
#ifdef XPTI_STATISTICS
    std::cout << "Tracepoint inserts : " << MInsertions.load() << '\n';
    std::cout << "Tracepoint lookups : " << MRetrievals.load() << '\n';
#endif
  }

  /// @brief Registers a payload and generates a unique identifier (UID) for it.
  ///
  /// This function takes a pointer to a payload, registers it, and generates a
  /// UID for it. The UID is based on the payload's name, source file, line
  /// number, and column number. If the UID has already been generated and
  /// cached, the function returns the cached UID. Otherwise, it generates a new
  /// UID, caches it in the payload, and then returns it.
  ///
  /// @param Payload Pointer to the payload that is to be registered.
  /// @return A pair consisting of the UID and a pointer to the registered
  /// payload.
  ///
  uid_payload_t registerPayload(xpti::payload_t *Payload) {
    // Payload is updated with string references
    auto uid = makeUID(Payload);
    if (!uid.isValid())
      return std::make_pair(uid, nullptr);

    std::lock_guard<std::mutex> Lock(MEventMutex);
    // We also want to query the payload by universal ID that has been
    // generated. Retrieve the payload entry associated with the unique
    // identifier (uid); If it were previously registered, the values will be
    // set and the payload is valid
    auto &PayloadEntry = MPayloads[uid];
    // Check if the payload is already registered by checking the
    // 'PayloadRegistered' flag
    if (PayloadEntry.first.flags &
        (uint64_t)payload_flag_t::PayloadRegistered) {
      // If the payload is already registered, increment the instance count
      uid.instance = ++PayloadEntry.second;
    } else {
      // If the payload is not registered, set the 'PayloadRegistered' flag
      Payload->flags |= (uint64_t)payload_flag_t::PayloadRegistered;
      // Update the payload entry with the new payload
      // Note: This operation should be thread-safe
      PayloadEntry.first = *Payload;
      // Set the instance count to 1 as this is the first instance of the
      // payload
      uid.instance = PayloadEntry.second = 1;
    }

    return std::make_pair(uid, &PayloadEntry.first);
  }

  /// @brief Generates a unique identifier (UID) for a given payload.
  ///
  /// This function creates a UID based on the payload's name, source file, line
  /// number, and column number. If the UID has already been generated and
  /// cached, the function returns the cached UID. Otherwise, it generates a new
  /// UID, caches it in the payload, and then returns it. The operation also
  /// updates the payload string pointers with the string references from the
  /// string table where the strings have been registered.
  ///
  /// @param Payload Pointer to the payload for which the UID is to be
  /// generated.
  /// @return The UID for the given payload.
  ///
  xpti::uid_t makeUID(xpti::payload_t *Payload) {
    // Initialize identifiers for function name, source file, line number, and
    // column number
    uint32_t name_id = 0, source_id = 0, line_no = 0, col_no = 0;

    // If the payload is not valid, return an empty uid, which is invalid by
    // default
    if (!Payload->isValid())
      return xpti::uid_t();

    // If the uid has already been generated and cached, return the cached uid;
    // this is the case when the payload from treacepoint map is stored in the
    // trace event data
    if (Payload->flags & static_cast<uint64_t>(payload_flag_t::UIDAvailable))
      return Payload->uid;

    // If the payload's function name is available, add it to the string table
    // and get its id
    if ((Payload->flags &
         static_cast<uint64_t>(payload_flag_t::NameAvailable))) {
      // Add the kernel name to the string table
      name_id = MStringTableRef.add(Payload->name, &Payload->name);
    }
    // If the payload's source file is available, add it to the string table and
    // get its id Also, get the line number and column number from the payload
    if ((Payload->flags &
         static_cast<uint64_t>(payload_flag_t::SourceFileAvailable))) {
      // Add source file information ot string table
      source_id =
          MStringTableRef.add(Payload->source_file, &Payload->source_file);
      line_no = Payload->line_no;
      col_no = Payload->column_no;
    }
    // Generate a new uid based on the function name id, source file id, line
    // number, and column number and cache it in the payload
    Payload->uid = uid_t(source_id, name_id, line_no, col_no);
    // If the new uid is valid, update the flag to say it is cached
    if (Payload->uid.isValid())
      Payload->flags |= static_cast<uint64_t>(payload_flag_t::UIDAvailable);
    // Return the new uid
    return Payload->uid;
  }

private:
  /// @brief Registers/Creates an event associated with a given payload.
  ///
  /// This function takes a pointer to a payload and registers it with the
  /// tracing framework. The returned UID will provide the instance information
  /// for the payload and a new trace event object is created if it's a new
  /// instance, if not an earlier event object associated with the instance
  /// information is returned.
  ///
  /// @param Payload Pointer to the payload for which the event is to be
  /// registered.
  /// @return Pointer to the trace event data object for the registered event.
  ///
  xpti::trace_event_data_t *register_event(const xpti::payload_t *Payload) {

    xpti::payload_t TempPayload = *Payload;
    // We need an explicit lock for the rest of the operations as the same
    // payload could be registered from multiple-threads.
    //
    // 1. makeUID(p) is invariant, but different invocations on different
    // threads simultaneously return UIDs with different instace IDs.
    auto HashPair = registerPayload(&TempPayload);
    if (!HashPair.first.isValid())
      return nullptr;

    std::lock_guard<std::mutex> Lock(MEventMutex);
    // 2. Since we are trying to register/create an event, the payload
    // registration will result in incrementing the instance ID for the
    // payload. This will always create a new trace event object for the
    // instance.
#ifdef XPTI_STATISTICS
    MInsertions++;
#endif
    // Create the event to correspond to the instance ID of the payload
    // present in UID
    auto &EventSlot = MEvents[HashPair.first];
    xpti::trace_event_data_t *Event = &EventSlot[HashPair.first.instance];
    Event->uid = HashPair.first;
    Event->unused = 0;
    Event->reserved.payload = HashPair.second;
    // Assume the event represented by the trace is part of an algorithm
    Event->event_type = (uint16_t)xpti::trace_event_type_t::algorithm;
    // Assume the activity represented by the event is always active
    Event->activity_type = (uint16_t)xpti::trace_activity_type_t::active;
    return Event;
  }

  xpti::safe_int64_t MUId;
  xpti::StringTable &MStringTableRef;
  xpti::safe_uint64_t MInsertions, MRetrievals;
  uid_payload_lut MPayloads;
  uid_event_lut MEvents;
  std::mutex MMetadataMutex;
  std::mutex MEventMutex;
};

/// \brief Helper class to manage subscriber callbacks for a given tracepoint
/// \details This class provides a thread-safe way to register and unregister
/// callbacks for a given stream. This will be used by tool plugins.
///
/// The class also provided a way to notify registered callbacks for a given
/// stream and trace point type. This will be used by framework to trigger
/// notifications are instrumentation points.
class Notifications {
public:
  using cb_entry_t = std::pair<bool, xpti::tracepoint_callback_api_t>;
  using cb_entries_t = std::vector<cb_entry_t>;
  using cb_t = std::unordered_map<uint16_t, cb_entries_t>;
  using stream_cb_t = std::unordered_map<uint16_t, cb_t>;
  using statistics_t = std::unordered_map<uint16_t, uint64_t>;

  xpti::result_t registerCallback(uint8_t StreamID, uint16_t TraceType,
                                  xpti::tracepoint_callback_api_t cbFunc) {
    if (!cbFunc)
      return xpti::result_t::XPTI_RESULT_INVALIDARG;

#ifdef XPTI_STATISTICS
    //  Initialize first encountered trace
    //  type statistics counters
    {
      std::lock_guard<std::mutex> Lock(MStatsLock);
      auto InstanceNo = MStats.find(TraceType);
      if (InstanceNo == MStats.end()) {
        MStats[TraceType] = 0;
      }
    }
#endif
    // If reader-writer locks were emplyed, this is where the writer lock can
    // be used
    std::lock_guard<std::mutex> Lock(MCBsLock);
    auto &StreamCBs =
        MCallbacksByStream[StreamID]; // thread-safe
                                      // What we get is a concurrent_hash_map
                                      // of vectors holding the callbacks we
                                      // need access to;
    auto Acc = StreamCBs.find(TraceType);
    if (Acc == StreamCBs.end()) {
      // Create a new slot and return the accessor for the trace type
      auto Tmp = StreamCBs[TraceType];
      Acc = StreamCBs.find(TraceType);
    }
    // If the key does not exist, a new entry is created and an accessor to it
    // is returned. If it exists, we have access to the previous entry.
    //
    // Before we add this element, we scan all existing elements to see if it
    // has already been registered. If so, we return XPTI_RESULT_DUPLICATE.
    //
    // If not, we set the first element of new entry to 'true' indicating that
    // it is valid. Unregister will just set this flag to false, indicating
    // that it is no longer valid and is unregistered.
    for (auto &Ele : Acc->second) {
      if (Ele.second == cbFunc) {
        if (Ele.first) // Already here and active
          return xpti::result_t::XPTI_RESULT_DUPLICATE;
        else { // it has been unregistered before, re-enable
          Ele.first = true;
          return xpti::result_t::XPTI_RESULT_UNDELETE;
        }
      }
    }
    // If we come here, then we did not find the callback being registered
    // already in the framework. So, we insert it.
    Acc->second.push_back(std::make_pair(true, cbFunc));
    return xpti::result_t::XPTI_RESULT_SUCCESS;
  }

  xpti::result_t unregisterCallback(uint8_t StreamID, uint16_t TraceType,
                                    xpti::tracepoint_callback_api_t cbFunc) {
    if (!cbFunc)
      return xpti::result_t::XPTI_RESULT_INVALIDARG;

    // Since we do not remove the callback function when they are unregistered
    // and only reset the flag, the writer lock is not held for very long; use
    // writer lock here.
    std::lock_guard<std::mutex> Lock(MCBsLock);
    auto &StreamCBs =
        MCallbacksByStream[StreamID]; // thread-safe
                                      //  What we get is a concurrent_hash_map
                                      //  of vectors holding the callbacks we
                                      //  need access to;
    auto Acc = StreamCBs.find(TraceType);
    bool Success = (Acc != StreamCBs.end());
    if (Success) {
      for (auto &Ele : Acc->second) {
        if (Ele.second == cbFunc) {
          if (Ele.first) { // Already here and active
                           // unregister, since delete and simultaneous
                           // iterations by other threads are unsafe
            Ele.first = false;
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

  xpti::result_t unregisterStream(uint8_t StreamID) {
    // If there are no callbacks registered for the requested stream ID, we
    // return not found; use reader lock here if the implementation moves to
    // reaer-writer locks.
    std::lock_guard<std::mutex> Lock(MCBsLock);
    if (MCallbacksByStream.count(StreamID) == 0)
      return xpti::result_t::XPTI_RESULT_NOTFOUND;

    auto &StreamCBs = MCallbacksByStream[StreamID]; // thread-safe
    // Disable all callbacks registered for the stream represented by StreamID
    for (auto &Item : StreamCBs) {
      for (auto &Ele : Item.second) {
        Ele.first = false;
      }
    }
    //  Return success
    return xpti::result_t::XPTI_RESULT_SUCCESS;
  }

  bool checkSubscribed(uint16_t StreamID, uint16_t TraceType) {
    if (StreamID == 0)
      return false;

    // If the notification framework moves to reader-writer locks, use reader
    // lock here
    std::lock_guard<std::mutex> Lock(MCBsLock);
    auto &StreamCBs =
        MCallbacksByStream[StreamID]; // thread-safe
                                      // What we get is a concurrent_hash_map
                                      // of vectors holding the callbacks we
                                      // need access to;
    bool StreamSubscribed = (StreamCBs.size() > 0);
    if (TraceType) {
      bool TraceTypeSubscribed = (StreamCBs.count(TraceType) > 0);
      return (StreamSubscribed && TraceTypeSubscribed);
    } else
      return StreamSubscribed;
  }

  xpti::result_t notifySubscribers(uint16_t StreamID, uint16_t TraceType,
                                   xpti::trace_event_data_t *Parent,
                                   xpti::trace_event_data_t *Object,
                                   uint64_t InstanceNo, const void *UserData) {
    {
      bool Success = false;
      xpti::Notifications::cb_t::iterator Acc;
      std::vector<xpti::tracepoint_callback_api_t> LocalCBs;
      {
        // Addresses bug reported against XPTI where the lock was held for the
        // entire duration of the notification calls; now the logic will grab
        // the notification functions when the lock is held and then releases
        // the lock before calling the notification functions. When using
        // reader-writer locks, use reader lock here.
        std::lock_guard<std::mutex> Lock(MCBsLock);
        cb_t &Stream = MCallbacksByStream[StreamID]; // Thread-safe
        Acc = Stream.find(TraceType);
        Success = (Acc != Stream.end());
        if (Success) {
          // Go through all registered callbacks and copy them
          for (auto &Ele : Acc->second) {
            if (Ele.first)
              LocalCBs.push_back(Ele.second);
          }
        }
      }

      // Go through all local copies of the callbacks and invoke them
      for (auto &CB : LocalCBs) {
        (CB)(TraceType, Parent, Object, InstanceNo, UserData);
      }
    }
#ifdef XPTI_STATISTICS
    auto &Counter = MStats[TraceType];
    {
      std::lock_guard<std::mutex> Lock(MStatsLock);
      Counter++;
    }
#endif
    return xpti::result_t::XPTI_RESULT_SUCCESS;
  }

  void printStatistics() {
#ifdef XPTI_STATISTICS
    printf("Notification statistics:\n");
    for (auto &s : MStats) {
      printf("%19s: [%lu] \n",
             stringify_trace_type((xpti::trace_point_type_t)s.first).c_str(),
             s.second);
    }
#endif
  }

  void clear() { MCallbacksByStream.clear(); }

private:
#ifdef XPTI_STATISTICS
  std::string stringify_trace_type(xpti::trace_point_type_t TraceType) {
    switch (TraceType) {
    case xpti::trace_point_type_t::graph_create:
      return "graph_create";
    case xpti::trace_point_type_t::node_create:
      return "node_create";
    case xpti::trace_point_type_t::edge_create:
      return "edge_create";
    case xpti::trace_point_type_t::region_begin:
      return "region_begin";
    case xpti::trace_point_type_t::region_end:
      return "region_end";
    case xpti::trace_point_type_t::task_begin:
      return "task_begin";
    case xpti::trace_point_type_t::task_end:
      return "task_end";
    case xpti::trace_point_type_t::barrier_begin:
      return "barrier_begin";
    case xpti::trace_point_type_t::barrier_end:
      return "barrier_end";
    case xpti::trace_point_type_t::lock_begin:
      return "lock_begin";
    case xpti::trace_point_type_t::lock_end:
      return "lock_end";
    case xpti::trace_point_type_t::signal:
      return "signal";
    case xpti::trace_point_type_t::transfer_begin:
      return "transfer_begin";
    case xpti::trace_point_type_t::transfer_end:
      return "transfer_end";
    case xpti::trace_point_type_t::thread_begin:
      return "thread_begin";
    case xpti::trace_point_type_t::thread_end:
      return "thread_end";
    case xpti::trace_point_type_t::wait_begin:
      return "wait_begin";
    case xpti::trace_point_type_t::wait_end:
      return "wait_end";
      break;
    default: {
      std::string str =
          "unknown/user_defined[" +
          std::to_string(XPTI_EXTRACT_USER_DEFINED_ID(TraceType)) + "]";
      return str;
    }
    }
  }
#endif
  stream_cb_t MCallbacksByStream;
  std::mutex MCBsLock;
  std::mutex MStatsLock;
  statistics_t MStats;
};

class Framework {
public:
  Framework()
      : MUniversalIDs(0), MTracepoints(MStringTableRef), MTraceEnabled(false) {
    //  Load all subscribers on construction
    MSubscribers.loadFromEnvironmentVariable();
    // Since we are in the dynamic component, it is implied that the dispatcher
    // has been set
    MTraceEnabled =
        (g_helper.checkTraceEnv() && MSubscribers.hasValidSubscribers());
  }

  void clear() {
    MUniversalIDs = 1;
    MTracepoints.clear();
    MStringTableRef.clear();
    MNotifier.clear();
  }

  inline void setTraceEnabled(bool yesOrNo = true) { MTraceEnabled = yesOrNo; }

  inline bool traceEnabled() { return MTraceEnabled; }

  inline uint64_t makeUniqueID() { return MTracepoints.makeUniqueID(); }

  xpti::uid_t getUniversalID() const noexcept { return g_tls_uid; }

  void setUniversalID(xpti::uid_t &uid) noexcept { g_tls_uid = uid; }

  xpti::result_t stashTuple(const char *key, uint64_t value) {
    if (!key)
      return xpti::result_t::XPTI_RESULT_FAIL;

    std::get<0>(g_tls_stash_tuple) = key;
    std::get<1>(g_tls_stash_tuple) = value;
    return xpti::result_t::XPTI_RESULT_SUCCESS;
  }

  xpti::result_t getStashedTuple(char **key, uint64_t &value) {
    if (!key)
      return xpti::result_t::XPTI_RESULT_INVALIDARG;

    const char *tls_key = std::get<0>(g_tls_stash_tuple);
    if (!tls_key)
      return xpti::result_t::XPTI_RESULT_NOTFOUND;

    (*key) = const_cast<char *>(tls_key);
    value = std::get<1>(g_tls_stash_tuple);
    return xpti::result_t::XPTI_RESULT_SUCCESS;
  }

  void unstashTuple() {
    if (!std::get<0>(g_tls_stash_tuple))
      return;

    // std::get<0>(g_tls_stash_tuple) = nullptr;
    // std::get<1>(g_tls_stash_tuple) = 0;
    // We will use the actual unstash code when we implement a stack to allow
    // multiple stashes/thread
  }

  bool checkTraceEnabled(uint16_t stream, uint16_t type) {
    if (MTraceEnabled) {
      return MNotifier.checkSubscribed(stream, type);
    }
    return false;
  }

  xpti::result_t addMetadata(xpti::trace_event_data_t *Event, const char *Key,
                             object_id_t ValueID) {
    return MTracepoints.addMetadata(Event, Key, ValueID);
  }

  xpti::trace_event_data_t *
  createEvent(const xpti::payload_t *Payload, uint16_t EventType,
              xpti::trace_activity_type_t ActivityType, uint64_t *InstanceNo) {
    if (!Payload || !InstanceNo)
      return nullptr;

    if (Payload->flags == 0)
      return nullptr;

    xpti::trace_event_data_t *Event = MTracepoints.create(Payload, InstanceNo);

    // Event is not managed by anyone. The unique_id that is a part of the
    // event structure can be used to determine the payload that forms the
    // event. The attribute 'ev.UserData' and 'ev.reserved' can be used to
    // store user defined and system defined data respectively. Currently the
    // 'reserved' field is not used, but object lifetime management must be
    // employed once this is active.
    //
    Event->event_type = EventType;
    Event->activity_type = (uint16_t)ActivityType;
    return Event;
  }

  void releaseEvent(xpti::trace_event_data_t *event) {
    MTracepoints.releaseEvent(event);
  }

  inline const xpti::trace_event_data_t *findEvent(xpti::uid_t &UniversalID) {
    return MTracepoints.eventData(UniversalID);
  }

  xpti::result_t initializeStream(const char *Stream, uint32_t MajorRevision,
                                  uint32_t MinorRevision,
                                  const char *VersionString) {
    if (!Stream || !VersionString)
      return xpti::result_t::XPTI_RESULT_INVALIDARG;

    MSubscribers.initializeForStream(Stream, MajorRevision, MinorRevision,
                                     VersionString);
    return xpti::result_t::XPTI_RESULT_SUCCESS;
  }

  uint8_t registerStream(const char *StreamName) {
    return (uint8_t)MStreamStringTable.add(StreamName);
  }

  void closeAllStreams() {
    auto Table = MStreamStringTable.table();
    StringTable::st_reverse_t::iterator it;
    for (it = Table.begin(); it != Table.end(); ++it) {
      xptiFinalize(it->second);
    }
  }

  xpti::result_t unregisterStream(const char *StreamName) {
    return finalizeStream(StreamName);
  }

  uint8_t registerVendor(const char *StreamName) {
    return (uint8_t)MVendorStringTable.add(StreamName);
  }

  string_id_t registerString(const char *String, char **TableString) {
    if (!TableString || !String)
      return xpti::invalid_id;

    *TableString = 0;

    const char *RefStr;
    auto ID = MStringTableRef.add(String, &RefStr);
    *TableString = const_cast<char *>(RefStr);

    return ID;
  }

  const char *lookupString(string_id_t ID) {
    if (ID < 0)
      return nullptr;
    return MStringTableRef.query(ID);
  }

  object_id_t registerObject(const char *Object, size_t Size, uint8_t Type) {
    if (!Object)
      return xpti::invalid_id;

    return MObjectTable.insert(std::string_view(Object, Size), Type);
  }

  object_data_t lookupObject(object_id_t ID) {
    auto [Result, Type] = MObjectTable.lookup(ID);
    return {Result.size(), Result.data(), Type};
  }

  xpti::uid_t registerPayload(xpti::payload_t *payload) {
    if (!payload)
      return xpti::uid_t();

    return MTracepoints.registerPayload(payload).first;
  }

  xpti::trace_point_data_t
  registerPayloadAndMakeScopeData(xpti::payload_t *payload) {
    xpti::trace_point_data_t ScopeData;
    if (!payload || !payload->isValid())
      return ScopeData;

    uint64_t InstanceNo = 0;
    auto event = MTracepoints.create(payload, &InstanceNo);
    // Scope data is created by this function and needs to successfully populate
    // all attributes for it to be valid
    ScopeData.uid = event->uid;
    // Lifetime of a payload is the duration of the application
    ScopeData.payload = event->reserved.payload;
    // The caller will manage the lifetime of the event, which is the scope
    ScopeData.event = event;
    return ScopeData;
  }

  xpti::result_t registerCallback(uint8_t StreamID, uint16_t TraceType,
                                  xpti::tracepoint_callback_api_t cbFunc) {
    return MNotifier.registerCallback(StreamID, TraceType, cbFunc);
  }

  xpti::result_t unregisterCallback(uint8_t StreamID, uint16_t TraceType,
                                    xpti::tracepoint_callback_api_t cbFunc) {
    return MNotifier.unregisterCallback(StreamID, TraceType, cbFunc);
  }

  xpti::result_t notifySubscribers(uint8_t StreamID, uint16_t TraceType,
                                   xpti::trace_event_data_t *Parent,
                                   xpti::trace_event_data_t *Object,
                                   uint64_t InstanceNo, const void *UserData) {
    if (!MTraceEnabled)
      return xpti::result_t::XPTI_RESULT_FALSE;
    if (!Object) {
      // We have relaxed the rules for notifications: Notifications can now
      // have 'nullptr' for both the Parent and Object only if UserData is
      // provided and the trace_point_type is function_begin/function_end.
      // This allows us to trace function calls without too much effort.
      std::array<trace_point_type_t, 26> AllowedTypes = {
          trace_point_type_t::function_begin,
          trace_point_type_t::function_end,
          trace_point_type_t::function_with_args_begin,
          trace_point_type_t::function_with_args_end,
          trace_point_type_t::mem_alloc_begin,
          trace_point_type_t::mem_alloc_end,
          trace_point_type_t::mem_release_begin,
          trace_point_type_t::mem_release_end,
          trace_point_type_t::offload_alloc_memory_object_construct,
          trace_point_type_t::offload_alloc_memory_object_associate,
          trace_point_type_t::offload_alloc_memory_object_release,
          trace_point_type_t::offload_alloc_memory_object_destruct,
          trace_point_type_t::offload_alloc_accessor,
          trace_point_type_t::diagnostics};
      const auto Predicate = [TraceType](trace_point_type_t RHS) {
        return TraceType == static_cast<uint16_t>(RHS);
      };
      if (!(UserData &&
            std::any_of(AllowedTypes.begin(), AllowedTypes.end(), Predicate))) {
        return xpti::result_t::XPTI_RESULT_INVALIDARG;
      }
    }
    //
    //  Notify all subscribers for the stream 'StreamID'
    //
    return MNotifier.notifySubscribers(StreamID, TraceType, Parent, Object,
                                       InstanceNo, UserData);
  }

  bool hasSubscribers() { return MSubscribers.hasValidSubscribers(); }

  xpti::result_t finalizeStream(const char *Stream) {
    if (!Stream)
      return xpti::result_t::XPTI_RESULT_INVALIDARG;
    MSubscribers.finalizeForStream(Stream);
    return MNotifier.unregisterStream(MStreamStringTable.add(Stream));
  }

  const xpti::payload_t *queryPayload(xpti::trace_event_data_t *Event) {
    return MTracepoints.payloadData(Event);
  }

  const xpti::payload_t *queryPayloadByUID(xpti::uid_t &uid) {
    return MTracepoints.payloadDataByUID(uid);
  }

  void printStatistics() {
    MNotifier.printStatistics();
    MStringTableRef.printStatistics();
    MTracepoints.printStatistics();
  }

  static Framework &instance() {
    Framework *TmpFramework = MInstance.load(std::memory_order_relaxed);
    std::atomic_thread_fence(std::memory_order_acquire);
    if (TmpFramework == nullptr) {
      std::lock_guard<utils::SpinLock> Lock{MSingletonMutex};
      TmpFramework = MInstance.load(std::memory_order_relaxed);
      if (TmpFramework == nullptr) {
        TmpFramework = new Framework();
        std::atomic_thread_fence(std::memory_order_release);
        MInstance.store(TmpFramework, std::memory_order_relaxed);
      }
    }

    return *TmpFramework;
  }

private:
  friend void ::xptiFrameworkFinalize();

  static void release() {
    Framework *TmpFramework = MInstance.load(std::memory_order_relaxed);
    MInstance.store(nullptr, std::memory_order_relaxed);
    delete TmpFramework;
  }

  /// Stores singleton instance
  static std::atomic<Framework *> MInstance;
  /// Trivially destructible mutex for double-checked lock idiom
  static utils::SpinLock MSingletonMutex;
  /// Thread-safe counter used for generating universal IDs
  xpti::safe_uint64_t MUniversalIDs;
  /// Manages loading the subscribers and calling their init() functions
  xpti::Subscribers MSubscribers;
  /// Used to send event notification to subscribers
  xpti::Notifications MNotifier;
  /// Thread-safe string table
  xpti::StringTable MStringTableRef;
  /// Thread-safe object table
  xpti::ObjectTable<object_id_t> MObjectTable;
  /// Thread-safe string table, used for stream IDs
  xpti::StringTable MStreamStringTable;
  /// Thread-safe string table, used for vendor IDs
  xpti::StringTable MVendorStringTable;
  /// Manages the tracepoints - framework caching
  xpti::Tracepoints MTracepoints;
  /// Flag indicates whether tracing should be enabled
  bool MTraceEnabled;
};

static int GFrameworkReferenceCounter = 0;

std::atomic<Framework *> Framework::MInstance;
utils::SpinLock Framework::MSingletonMutex;
} // namespace xpti

extern "C" {

XPTI_EXPORT_API void xptiFrameworkInitialize() {
  std::lock_guard<xpti::utils::SpinLock> guard{xpti::g_framework_mutex};
  xpti::GFrameworkReferenceCounter++;
}

XPTI_EXPORT_API void xptiFrameworkFinalize() {
  std::lock_guard<xpti::utils::SpinLock> guard{xpti::g_framework_mutex};

  xpti::GFrameworkReferenceCounter--;
  if (xpti::GFrameworkReferenceCounter == 0) {
    xpti::Framework::release();
  }
}

XPTI_EXPORT_API xpti::uid_t xptiGetUniversalId() {
  return xpti::Framework::instance().getUniversalID();
}

XPTI_EXPORT_API void xptiSetUniversalId(xpti::uid_t &uid) {
  xpti::Framework::instance().setUniversalID(uid);
}

XPTI_EXPORT_API xpti::result_t xptiStashTuple(const char *key, uint64_t value) {
  return xpti::Framework::instance().stashTuple(key, value);
}

XPTI_EXPORT_API xpti::result_t xptiGetStashedTuple(char **key,
                                                   uint64_t &value) {
  return xpti::Framework::instance().getStashedTuple(key, value);
}

XPTI_EXPORT_API void xptiUnstashTuple() {
  xpti::Framework::instance().unstashTuple();
}

XPTI_EXPORT_API uint16_t
xptiRegisterUserDefinedTracePoint(const char *ToolName, uint8_t UserDefinedTP) {
  uint8_t ToolID = xpti::Framework::instance().registerVendor(ToolName);
  UserDefinedTP |= (uint8_t)xpti::trace_point_type_t::user_defined;
  uint16_t UserDefTracepoint = XPTI_PACK08_RET16(ToolID, UserDefinedTP);

  return UserDefTracepoint;
}

XPTI_EXPORT_API uint16_t xptiRegisterUserDefinedEventType(
    const char *ToolName, uint8_t UserDefinedEvent) {
  uint8_t ToolID = xpti::Framework::instance().registerVendor(ToolName);
  UserDefinedEvent |= (uint8_t)xpti::trace_event_type_t::user_defined;
  uint16_t UserDefEventType = XPTI_PACK08_RET16(ToolID, UserDefinedEvent);
  return UserDefEventType;
}

XPTI_EXPORT_API xpti::result_t xptiInitialize(const char *Stream, uint32_t maj,
                                              uint32_t min,
                                              const char *version) {
  return xpti::Framework::instance().initializeStream(Stream, maj, min,
                                                      version);
}

XPTI_EXPORT_API void xptiFinalize(const char *Stream) {
  xpti::Framework::instance().finalizeStream(Stream);
}

XPTI_EXPORT_API uint64_t xptiGetUniqueId() {
  return xpti::Framework::instance().makeUniqueID();
}

XPTI_EXPORT_API xpti::string_id_t xptiRegisterString(const char *String,
                                                     char **RefTableStr) {
  return xpti::Framework::instance().registerString(String, RefTableStr);
}

XPTI_EXPORT_API const char *xptiLookupString(xpti::string_id_t ID) {
  return xpti::Framework::instance().lookupString(ID);
}

XPTI_EXPORT_API xpti::object_id_t
xptiRegisterObject(const char *Data, size_t Size, uint8_t Type) {
  return xpti::Framework::instance().registerObject(Data, Size, Type);
}

XPTI_EXPORT_API xpti::object_data_t xptiLookupObject(xpti::object_id_t ID) {
  return xpti::Framework::instance().lookupObject(ID);
}

XPTI_EXPORT_API xpti::uid_t xptiRegisterPayload(xpti::payload_t *payload) {
  return xpti::Framework::instance().registerPayload(payload);
}

XPTI_EXPORT_API uint8_t xptiRegisterStream(const char *StreamName) {
  return xpti::Framework::instance().registerStream(StreamName);
}

XPTI_EXPORT_API xpti::result_t xptiUnregisterStream(const char *StreamName) {
  return xpti::Framework::instance().unregisterStream(StreamName);
}

XPTI_EXPORT_API xpti::trace_event_data_t *
xptiMakeEvent(const char * /*Name*/, xpti::payload_t *Payload, uint16_t Event,
              xpti::trace_activity_type_t Activity, uint64_t *InstanceNo) {
  return xpti::Framework::instance().createEvent(Payload, Event, Activity,
                                                 InstanceNo);
}

XPTI_EXPORT_API void xptiReset() { xpti::Framework::instance().clear(); }

XPTI_EXPORT_API const xpti::trace_event_data_t *
xptiFindEvent(xpti::uid_t &UId) {
  return xpti::Framework::instance().findEvent(UId);
}

XPTI_EXPORT_API const xpti::payload_t *
xptiQueryPayload(xpti::trace_event_data_t *LookupObject) {
  return xpti::Framework::instance().queryPayload(LookupObject);
}

XPTI_EXPORT_API const xpti::payload_t *xptiQueryPayloadByUID(xpti::uid_t &uid) {
  return xpti::Framework::instance().queryPayloadByUID(uid);
}

XPTI_EXPORT_API xpti::result_t
xptiRegisterCallback(uint8_t StreamID, uint16_t TraceType,
                     xpti::tracepoint_callback_api_t cbFunc) {
  return xpti::Framework::instance().registerCallback(StreamID, TraceType,
                                                      cbFunc);
}

XPTI_EXPORT_API xpti::result_t
xptiUnregisterCallback(uint8_t StreamID, uint16_t TraceType,
                       xpti::tracepoint_callback_api_t cbFunc) {
  return xpti::Framework::instance().unregisterCallback(StreamID, TraceType,
                                                        cbFunc);
}

XPTI_EXPORT_API xpti::result_t
xptiNotifySubscribers(uint8_t StreamID, uint16_t TraceType,
                      xpti::trace_event_data_t *Parent,
                      xpti::trace_event_data_t *Object, uint64_t InstanceNo,
                      const void *TemporalUserData) {
  return xpti::Framework::instance().notifySubscribers(
      StreamID, TraceType, Parent, Object, InstanceNo, TemporalUserData);
}

XPTI_EXPORT_API bool xptiTraceEnabled() {
  return xpti::Framework::instance().traceEnabled();
}

XPTI_EXPORT_API bool xptiCheckTraceEnabled(uint16_t stream, uint16_t ttype) {
  return xpti::Framework::instance().checkTraceEnabled(stream, ttype);
}

XPTI_EXPORT_API xpti::result_t xptiAddMetadata(xpti::trace_event_data_t *Event,
                                               const char *Key,
                                               xpti::object_id_t ID) {
  return xpti::Framework::instance().addMetadata(Event, Key, ID);
}

XPTI_EXPORT_API xpti::metadata_t *
xptiQueryMetadata(xpti::trace_event_data_t *Event) {
  return &Event->reserved.metadata;
}

XPTI_EXPORT_API void xptiForceSetTraceEnabled(bool YesOrNo) {
  xpti::Framework::instance().setTraceEnabled(YesOrNo);
}

XPTI_EXPORT_API void xptiReleaseEvent(xpti::trace_event_data_t *Event) {
  return xpti::Framework::instance().releaseEvent(Event);
}

XPTI_EXPORT_API uint8_t xptiGetDefaultStreamID() {
  return xpti::g_default_stream_id;
}

XPTI_EXPORT_API xpti::result_t xptiSetDefaultStreamID(uint8_t DefaultStreamId) {
  if ((int8_t)DefaultStreamId < 0)
    return xpti::result_t::XPTI_RESULT_INVALIDARG;

  xpti::g_default_stream_id = DefaultStreamId;
  return xpti::result_t::XPTI_RESULT_SUCCESS;
}
XPTI_EXPORT_API xpti::trace_event_type_t xptiGetDefaultEventType() {
  return xpti::g_default_event_type;
}

XPTI_EXPORT_API xpti::result_t
xptiSetDefaultEventType(xpti::trace_event_type_t DefaultEventType) {
  if (DefaultEventType == xpti::trace_event_type_t::unknown_event)
    return xpti::result_t::XPTI_RESULT_INVALIDARG;

  xpti::g_default_event_type = DefaultEventType;
  return xpti::result_t::XPTI_RESULT_SUCCESS;
}

XPTI_EXPORT_API xpti::trace_point_type_t xptiGetDefaultTraceType() {
  return xpti::g_default_trace_type;
}

XPTI_EXPORT_API xpti::result_t
xptiSetDefaultTraceType(xpti::trace_point_type_t DefaultTraceType) {
  if (DefaultTraceType == xpti::trace_point_type_t::unknown_type)
    return xpti::result_t::XPTI_RESULT_INVALIDARG;

  xpti::g_default_trace_type = DefaultTraceType;
  return xpti::result_t::XPTI_RESULT_SUCCESS;
}

XPTI_EXPORT_API xpti::trace_point_data_t xptiGetTracepointScopeData() {
  return g_tls_tracepoint_scope_data;
}

XPTI_EXPORT_API xpti::result_t
xptiSetTracepointScopeData(xpti::trace_point_data_t &data) {
  if (!data.isValid())
    return xpti::result_t::XPTI_RESULT_INVALIDARG;
  // Copy to TLS so it is available for the remainder of the scope
  g_tls_tracepoint_scope_data = data;
  // Also set Universal ID separately as it may be in use by older
  // implementations of tools
  xptiSetUniversalId(data.uid);
  return xpti::result_t::XPTI_RESULT_SUCCESS;
}

XPTI_EXPORT_API void xptiUnsetTracepointScopeData() {
  g_tls_tracepoint_scope_data = xpti::trace_point_data_t();
}

XPTI_EXPORT_API xpti::trace_point_data_t
xptiRegisterPayloadAndPrepareTracepointData(xpti::payload_t &payload) {
  return xpti::Framework::instance().registerPayloadAndMakeScopeData(&payload);
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
    xpti::Framework::instance().printStatistics();
#endif
    break;
  }

  return TRUE;
}

#else // Linux (possibly macOS?)

__attribute__((constructor)) static void framework_init() {}

__attribute__((destructor)) static void framework_fini() {
#ifdef XPTI_STATISTICS
  xpti::Framework::instance().printStatistics();
#endif
}

#endif
