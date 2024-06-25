//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
#include "xpti/xpti_trace_framework.hpp"
#include "emhash/hash_table8.hpp"
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
#include <shared_mutex>
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

/// @brief A thread-local 64-bit unsigned integer, default initialized.
/// This variable is used to store a unique identifier within a program scope
/// for each thread.
static thread_local uint64_t g_tls_uid = xpti::invalid_uid;

/// @brief A TLS of type xpti::trace_point_data_t, default initialized.
/// This variable is used to store trace point data generated from code location
/// information in a function scope for all top level entry points recorded for
/// each thread.
static thread_local xpti::tracepoint_data_t g_tls_tracepoint_scope_data;

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

/// @brief A global boolean flag to control self-notification of trace points.
///
/// If this flag is set to true, the trace point will notify itself when it is
/// hit. This can be useful for debugging or for generating more detailed trace
/// information. By default, this flag is set to false, meaning that trace
/// points do not notify themselves.
bool g_tracepoint_self_notify = false;

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

class uid_64bit_128bit_lut {
public:
  using uid64_128_t = emhash8::HashMap<uint64_t, uid128_t>;
  using instance2uid64_t = emhash8::HashMap<uint64_t, uint64_t>;
  using uid128_64_t = emhash8::HashMap<uid128_t, instance2uid64_t>;
  using uid64_to_payload_t = emhash8::HashMap<uint64_t, xpti::payload_t>;

  uid_64bit_128bit_lut() : M64bitUIDs(1) {}
  ~uid_64bit_128bit_lut() {
    M64to128.clear();
    M128to64.clear();
    M64bitUIDs = 1;
  }

  uint64_t findUID64(uid128_t *UId) {
    // If the UId already has a 64-bit ID, return it
    if (UId->uid64 != xpti::invalid_uid)
      return UId->uid64;

    // This is a read-only operation, so we can use a shared lock
    {
      std::unique_lock Lock(MMutex);
      auto &UIDInstanceMap = M128to64[*UId];
      if (UIDInstanceMap.count(UId->instance)) {
        // Update the UId->uid64 to contain the right 64-bit ID and return it
        UId->uid64 = UIDInstanceMap[UId->instance];
        return UId->uid64;
      }
      // If we are here, then UID instance not found. We have to create the
      // 64-bit ID and add it to the map and update reverse lookup
      uint64_t New64BitUID = M64bitUIDs++;
      UIDInstanceMap[UId->instance] = New64BitUID;
      // Update the UId to have the right 64-bit ID
      UId->uid64 = New64BitUID;
      M64to128[New64BitUID] = *UId;
      return New64BitUID;
    }
  }

  bool eraseUID128(uid128_t *UId) {
    std::unique_lock Lock(MMutex);
    auto Loc = M128to64.find(*UId);
    if (Loc != M128to64.end()) {
      M128to64.erase(Loc);
      // We will not delete the proxy payload as it may be in use and we do not
      // want to invalidate the pointer
      return true;
    }
    return false;
  }

  xpti::payload_t *findPayloadInstance(uint64_t UId, xpti::payload_t *Payload) {
    std::unique_lock Lock(MPayloadMutex);
    if (M64ToPayload.count(UId)) {
      // Someone has already queried this payload, so we have a proxy payload
      // that we can return
      return &M64ToPayload[UId];
    } else {
      // Since we do not have a proxy payload, we will create one and return it
      auto &PayloadEntry = M64ToPayload[UId];
      PayloadEntry = *Payload;
      PayloadEntry.internal = UId;
      PayloadEntry.flags |=
          static_cast<uint64_t>(payload_flag_t::HashAvailable);
      return &PayloadEntry;
    }
  }

  xpti::uid128_t findUID128(uint64_t UId) {
    {
      std::shared_lock Lock(MMutex);
      auto Loc = M64to128.find(UId);
      if (Loc != M64to128.end()) {
        return Loc->second;
      }
    }
    xpti::uid128_t invalid_uid128;
    return invalid_uid128;
  }

private:
  uint64_t M64bitUIDs;
  uid64_128_t M64to128;
  uid128_64_t M128to64;
  uid64_to_payload_t M64ToPayload;
  /// Lock to ensure the operation on these maps are safe
  mutable std::shared_mutex MMutex;
  mutable std::shared_mutex MPayloadMutex;
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
  using uid_entry_t = std::pair<xpti::payload_t, uint64_t>;

  /// @typedef uid_payload_lut
  /// @brief A type alias for an unordered map from uid_t to uid_entry_t.
  /// This type is used to create a lookup table (lut) that maps unique
  /// identifiers (uids) to their corresponding payload entries.
  using uid_payload_lut = emhash8::HashMap<xpti::uid128_t, uid_entry_t>;

  /// @typedef trace_instance_t
  /// @brief A type alias for an unordered map from uint64_t to
  /// xpti::trace_event_data_t. This type is used to map unique identifiers to
  /// their corresponding trace event data vectors that records instance data.
  using trace_instance_t = emhash8::HashMap<uint64_t, xpti::trace_event_data_t>;

  /// @typedef uid_event_lut
  /// @brief A type alias for an unordered map from uid_t to trace_instance_t.
  /// This type is used to create a lookup table (lut) that maps unique
  /// identifiers (uids) to their corresponding trace instances.
  using uid_event_lut = emhash8::HashMap<uid128_t, trace_instance_t>;

  /// @typedef uid_payload_t
  /// @brief A type alias for a pair containing a uid_t and a pointer to an
  /// xpti::payload_t. This type is used to store a unique identifier and a
  /// pointer to its corresponding payload together.
  using uid_payload_t = std::pair<uid128_t, xpti::payload_t *>;

  struct PayloadInstance {
    xpti::uid128_t UId;
    xpti::payload_t *Payload;
  };

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

  //  Create an event with the payload information. If one already exists, then
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
  //  2. Create a mapping from Universal ID <--> Payload
  //  3. Create a mapping from Universal ID <--> Event
  xpti::trace_event_data_t *create(const xpti::payload_t *Payload,
                                   uint64_t *InstanceNo) {
    return register_event(Payload, InstanceNo);
  }
  // Method to get the payload information from the event structure. This method
  // uses the Universal ID in the event structure to lookup the payload
  // information and returns the payload if available.
  //
  // This method is thread-safe
  const xpti::payload_t *payloadData(xpti::trace_event_data_t *Event) {
    if (!Event || xpti::is_valid_uid(Event->universal_id) == false)
      return nullptr;
    // Scoped lock until the information is retrieved from the map
    {
      if (Event->reserved.payload)
        return Event->reserved.payload;
      else {
        if (!xpti::is_valid_uid(Event->universal_id))
          return nullptr;
        else {
          std::shared_lock Lock(MPayloadMutex);
          auto &PayloadEntry = MPayloads[Event->universal_id];
          xpti::payload_t *Payload = &PayloadEntry.first;
          // Cache the payload information if it hasn't already been cached
          if (!Event->reserved.payload)
            Event->reserved.payload = Payload;
          return Payload;
        }
      }
    }
  }

  const xpti::payload_t *payloadDataByUniversalID(xpti::uid128_t *UId) {
    if (!UId || xpti::is_valid_uid(*UId) == false)
      return nullptr;
    // Scoped lock until the information is retrieved from the map
    {
      std::shared_lock Lock(MPayloadMutex);
      auto &PayloadEntry = MPayloads[*UId];
      return &PayloadEntry.first;
    }
  }

  const xpti::trace_event_data_t *lookupEventData(xpti::uid128_t *UId) {
    if (!UId || !xpti::is_valid_uid(*UId))
      return nullptr;

    std::shared_lock Lock(MEventMutex);
    if (MEvents.count(*UId)) {
      auto &EventEntry = MEvents[*UId];
      if (EventEntry.count(UId->instance))
        return &EventEntry[UId->instance];
      else
        return nullptr;
    } else
      return nullptr;
  }

  void releaseEvent(xpti::trace_event_data_t *Event) {
    if (!Event || !xpti::is_valid_uid(Event->universal_id))
      return;

    bool EventReleased = false;
    xpti::uid128_t UId = Event->universal_id;
    {
      std::unique_lock Lock(MEventMutex);
      // Find the event list for a given UID
      auto EvLoc = MEvents.find(UId);
      if (EvLoc != MEvents.end() && EvLoc->second.count(UId.instance)) {
        // Now release the event associated with the UID instance
        EvLoc->second.erase(UId.instance);
        // If there are no more events associated with the UID, we can release
        // the Payload as well, but we will not as the same payload may be
        // revisited and we need to keep the instance count going
        EventReleased = true;
      }
    }
    if (EventReleased)
      MUidLut64x128.eraseUID128(&UId);
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
    std::cout << "Tracepoint Hashmap :\n";
    MPayloadLUT.printStatistics();
#endif
  }

  PayloadInstance makeKeyFromPayload(xpti::payload_t *Payload) {
    xpti::uid128_t Key;
    // Validity of the payload structure is checked in the previous layer
    if (!Payload && !xpti::is_valid_payload(Payload))
      return PayloadInstance{Key, nullptr};

    // If not, we will use the name, source file, line number and column number
    // to create the key
    Key = makeUniversalId(Payload);
    // Check is Key is valid; If the payload is fully populated, then we will
    // have both Key.p1 and Key.p2 set. However, if only a function name is
    // provided, then we will have Key.p1 populated.
    std::unique_lock Lock(MPayloadMutex);
    auto &PayloadEntry = MPayloads[Key];
    if (PayloadEntry.first.flags == 0) {
      // We are seeing this UID for the first time, so we can update the Payload
      // information and set the instance to 1
      PayloadEntry.first = *Payload;
      PayloadEntry.first.uid.p3 = 1;
      PayloadEntry.first.internal = xpti::invalid_uid;
      Key.instance = PayloadEntry.second = 1;
    } else {
      // Since we have seen this Payload before, let's increment the instance
      Key.instance = ++PayloadEntry.second;
    }
    return PayloadInstance{Key, &PayloadEntry.first};
  }

  xpti::uid128_t makeUniversalId(xpti::payload_t *Payload) {
    xpti::uid128_t UId;
    if (!Payload || !xpti::is_valid_payload(Payload))
      return UId;

    // If the uid has not been generated and cached, update the flag to say so
    Payload->flags &= (~static_cast<uint64_t>(payload_flag_t::HashAvailable));
    uint64_t FileId, FuncId;
    int LineNo, ColNo;

    // If the payload's function name is available, add it to the string table
    // and get its id
    if ((Payload->flags &
         static_cast<uint64_t>(payload_flag_t::NameAvailable))) {
      // Add the kernel name/function name to the string table
      FuncId = MStringTableRef.add(Payload->name, &Payload->name);
    }

    // If the payload's source file is available, add it to the string table and
    // get its id Also, get the line number and column number from the payload
    if ((Payload->flags &
         static_cast<uint64_t>(payload_flag_t::SourceFileAvailable))) {
      // Add source file information ot string table
      FileId = MStringTableRef.add(Payload->source_file, &Payload->source_file);
      LineNo = Payload->line_no;
      ColNo = Payload->column_no;
    }

    UId = xpti::make_uid128(FileId, FuncId, LineNo, ColNo);
    // Update the fields of Payload that will remain invariant
    Payload->uid.p1 = XPTI_PACK32_RET64(FileId, LineNo);
    Payload->uid.p2 = XPTI_PACK32_RET64(0, FuncId);
    // UId.instance and UId.uid64 are still set to 0 and invalid

    return UId;
  }

  inline uint64_t findUID64(xpti::uid128_t *UId) {
    if (!UId || !xpti::is_valid_uid(*UId))
      return xpti::invalid_uid;

    // UId is deemed valid if we are here; check to see if it has a 64-bit ID
    // generated for it
    if (UId->uid64)
      return UId->uid64;

    // If not, register the new instance and return the new 64-bit ID; this will
    // also update the UId to have the right 64-bit value
    return MUidLut64x128.findUID64(UId);
  }

  inline xpti::uid128_t findUID128(uint64_t UId) {
    return MUidLut64x128.findUID128(UId);
  }

  inline xpti::payload_t *findPayloadInstance(uint64_t UId,
                                              xpti::payload_t *Payload) {
    return MUidLut64x128.findPayloadInstance(UId, Payload);
  }

private:
  // Register the payload and generate a universal ID for it.
  // Once registered, the payload is accessible through the
  // Universal ID that corresponds to the payload.
  //
  // This method is thread-safe
  xpti::trace_event_data_t *register_event(const xpti::payload_t *Payload,
                                           uint64_t *InstanceNo) {
    xpti::payload_t TempPayload = *Payload;
    // Initialize to invalid
    // We need an explicit lock for the rest of the operations as the same
    // payload could be registered from multiple-threads.
    //
    // 1. makeHash(p) is invariant, although the hash may be created twice and
    // written to the same field in the structure. If we have a lock guard, we
    // may be spinning and wasting time instead. We will just compute this in
    // parallel.
    // 2. MPayloads is queried and updated in a critical section. So, multiple
    // threads attempting to register the same payload to receive an event
    // should get the same event.
    //
    //  Make a hash value from the payload. If the hash value created is
    //  invalid, return immediately
    auto [UniversalId, InvarPayload] = makeKeyFromPayload(&TempPayload);
    if (!xpti::is_valid_uid(UniversalId))
      return nullptr;

    // Universal ID returned will have the right instance id for the payload
    {
#ifdef XPTI_STATISTICS
      MInsertions++;
#endif
      std::unique_lock Lock(MEventMutex);
      auto &EventSlot = MEvents[UniversalId];
      xpti::trace_event_data_t *Event = &EventSlot[UniversalId.instance];
      Event->universal_id = UniversalId;
      *InstanceNo = Event->instance_id = UniversalId.instance;
      Event->unique_id = xpti::invalid_uid;
      Event->unused = 0;
      // The Payload in here will point to the actual payload associated with a
      // UID. Since we are not going to have a UID instance associated with the
      // Payload, the instance ID for the UID associated with the Payload will
      // always be 1 - that is, when it was first registered.
      Event->reserved.payload = InvarPayload;
      return Event;
    }
  }

  xpti::safe_int64_t MUId;
  xpti::StringTable &MStringTableRef;
  xpti::safe_uint64_t MInsertions, MRetrievals;
  uid_payload_lut MPayloads;
  uid_event_lut MEvents;
  std::mutex MMetadataMutex;
  mutable std::shared_mutex MEventMutex;
  mutable std::shared_mutex MPayloadMutex;
  uid_64bit_128bit_lut MUidLut64x128;
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
    std::unique_lock Lock(MCBsLock);
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
    std::unique_lock Lock(MCBsLock);
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
    std::unique_lock Lock(MCBsLock);
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
    std::shared_lock Lock(MCBsLock);
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
        std::shared_lock Lock(MCBsLock);
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
             stringify_trace_type((xpti_trace_point_type_t)s.first).c_str(),
             s.second);
    }
#endif
  }

  void clear() { MCallbacksByStream.clear(); }

private:
#ifdef XPTI_STATISTICS
  std::string stringify_trace_type(xpti_trace_point_type_t TraceType) {
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
    default:
      std::string str =
          "unknown/user_defined[" +
          std::to_string(XPTI_EXTRACT_USER_DEFINED_ID(TraceType)) + "]";
      return str;
    }
  }
#endif
  stream_cb_t MCallbacksByStream;
  mutable std::shared_mutex MCBsLock;
  std::mutex MStatsLock;
  statistics_t MStats;
};

class Framework {
public:
  Framework()
      : MUniversalIDs(0), MTracepoints(MStringTableRef), MTraceEnabled(false) {
    //  Load all subscribers on construction
    MSubscribers.loadFromEnvironmentVariable();
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

  uint64_t getUniversalID() const noexcept { return g_tls_uid; }

  void setUniversalID(uint64_t uid) noexcept { g_tls_uid = uid; }

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

  xpti::trace_event_data_t *newEvent(const xpti::payload_t *Payload,
                                     uint64_t *InstanceNo, uint16_t EventType,
                                     xpti::trace_activity_type_t ActivityType) {
    if (!Payload || !xpti::is_valid_payload(Payload) || !InstanceNo)
      return nullptr;

    xpti::trace_event_data_t *Event = MTracepoints.create(Payload, InstanceNo);
    if (!Event)
      return nullptr;

    Event->event_type = EventType;
    Event->activity_type = (uint16_t)ActivityType;
    return Event;
  }

  void releaseEvent(xpti::trace_event_data_t *event) {
    MTracepoints.releaseEvent(event);
  }

  inline const xpti::trace_event_data_t *findEvent(uint64_t UniversalID) {
    if (UniversalID == xpti::invalid_uid)
      return nullptr;

    // UId should be populated with the right instance information
    xpti::uid128_t UId = findUID128(UniversalID);

    return lookupEvent(&UId);
  }

  inline const xpti::trace_event_data_t *lookupEvent(xpti::uid128_t *UId) {
    return MTracepoints.lookupEventData(UId);
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

  uint64_t registerPayload(xpti::payload_t *payload) {
    if (!payload || !xpti::is_valid_payload(payload))
      return xpti::invalid_uid;

    xpti::uid128_t UId;

    if (makeKeyFromPayload(payload, &UId) !=
        xpti::result_t::XPTI_RESULT_SUCCESS)
      return xpti::invalid_uid;

    // UId should be populated with the right instance information
    xpti::uid64_t Id = findUID64(&UId);
    payload->internal = Id;
    payload->flags |= static_cast<uint64_t>(payload_flag_t::HashAvailable);

    xpti::framework::uid_object_t UidHelper(UId);
    payload->uid.p1 = XPTI_PACK32_RET64(UidHelper.fileId(), UidHelper.lineNo());
    payload->uid.p2 = XPTI_PACK32_RET64(0, UidHelper.functionId());

    return Id;
  }

  xpti::result_t makeKeyFromPayload(xpti::payload_t *payload,
                                    xpti::uid128_t *uid) {
    if (!payload || !uid || !xpti::is_valid_payload(payload))
      return xpti::result_t::XPTI_RESULT_INVALIDARG;

    auto [UId, InvarPayload] = MTracepoints.makeKeyFromPayload(payload);
    if (!xpti::is_valid_uid(UId) || !InvarPayload)
      return xpti::result_t::XPTI_RESULT_FAIL;

    *uid = UId;
    *payload = *InvarPayload;

    return xpti::result_t::XPTI_RESULT_SUCCESS;
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

  inline const xpti::payload_t *queryPayload(xpti::trace_event_data_t *Event) {
    return MTracepoints.payloadData(Event);
  }

  const xpti::payload_t *queryPayloadByUID(uint64_t uid) {
    if (uid == xpti::invalid_uid)
      return nullptr;

    auto UId = findUID128(uid);
    auto Payload = lookupPayload(&UId);
    // Since payload does not have the attribute `internal` set, we may have to
    // create a copy of the payload for this instance and set it top be
    // comaptible with legacy API. This adds extra cost when legacy APIs are
    // used
    auto LegacyPayload =
        findPayloadInstance(uid, const_cast<xpti::payload_t *>(Payload));
    return LegacyPayload;
  }

  inline const xpti::payload_t *lookupPayload(xpti::uid128_t *uid) {
    return MTracepoints.payloadDataByUniversalID(uid);
  }

  void printStatistics() {
    MNotifier.printStatistics();
    MStringTableRef.printStatistics();
    MTracepoints.printStatistics();
  }

  inline uint64_t findUID64(xpti::uid128_t *UId128) {
    return MTracepoints.findUID64(UId128);
  }

  inline xpti::uid128_t findUID128(uint64_t UId64) {
    return MTracepoints.findUID128(UId64);
  }

  inline xpti::payload_t *findPayloadInstance(uint64_t UId64,
                                              xpti::payload_t *Payload) {
    return MTracepoints.findPayloadInstance(UId64, Payload);
  }

  static Framework &instance() {
    Framework *TmpFramework = MInstance.load(std::memory_order_relaxed);
    std::atomic_thread_fence(std::memory_order_acquire);
    if (TmpFramework == nullptr) {
      std::lock_guard<utils::SpinLock> Lock{MSingletoneMutex};
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
  static utils::SpinLock MSingletoneMutex;
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
utils::SpinLock Framework::MSingletoneMutex;
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

XPTI_EXPORT_API uint64_t xptiGetUniversalId() {
  return xpti::Framework::instance().getUniversalID();
}

XPTI_EXPORT_API void xptiSetUniversalId(uint64_t uid) {
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

XPTI_EXPORT_API uint64_t xptiRegisterPayload(xpti::payload_t *payload) {
  return xpti::Framework::instance().registerPayload(payload);
}

XPTI_EXPORT_API xpti::result_t xptiMakeKeyFromPayload(xpti::payload_t *payload,
                                                      xpti::uid128_t *key) {
  if (!payload || !key)
    return xpti::result_t::XPTI_RESULT_INVALIDARG;

  auto &FW = xpti::Framework::instance();
  return FW.makeKeyFromPayload(payload, key);
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
  auto &FW = xpti::Framework::instance();
  auto RetEv = FW.newEvent(Payload, InstanceNo, Event, Activity);
  if (RetEv) {
    RetEv->unique_id = FW.findUID64(&RetEv->universal_id);
    RetEv->universal_id.uid64 = RetEv->unique_id;
  }
  return RetEv;
}

XPTI_EXPORT_API xpti::trace_event_data_t *
xptiCreateEvent(xpti::payload_t *Payload, uint64_t *InstanceNo, uint16_t Event,
                xpti::trace_activity_type_t Activity) {
  return xpti::Framework::instance().newEvent(Payload, InstanceNo, Event,
                                              Activity);
}

XPTI_EXPORT_API void xptiReset() { xpti::Framework::instance().clear(); }

XPTI_EXPORT_API const xpti::trace_event_data_t *xptiFindEvent(uint64_t UId) {
  return xpti::Framework::instance().findEvent(UId);
}

XPTI_EXPORT_API const xpti::trace_event_data_t *
xptiLookupEvent(xpti::uid128_t *UId) {
  return xpti::Framework::instance().lookupEvent(UId);
}

XPTI_EXPORT_API const xpti::payload_t *
xptiQueryPayload(xpti::trace_event_data_t *LookupObject) {
  return xpti::Framework::instance().queryPayload(LookupObject);
}

XPTI_EXPORT_API const xpti::payload_t *xptiQueryPayloadByUID(uint64_t uid) {
  return xpti::Framework::instance().queryPayloadByUID(uid);
}

XPTI_EXPORT_API const xpti::payload_t *xptiLookupPayload(xpti::uid128_t *uid) {
  return xpti::Framework::instance().lookupPayload(uid);
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

XPTI_EXPORT_API void
xptiEnableTracepointScopeNotification(bool enableOrDisable) {
  xpti::g_tracepoint_self_notify = enableOrDisable;
}

XPTI_EXPORT_API bool xptiCheckTracepointScopeNotification() {
  return xpti::g_tracepoint_self_notify;
}

XPTI_EXPORT_API void xptiReleaseEvent(xpti::trace_event_data_t *Event) {
  return xpti::Framework::instance().releaseEvent(Event);
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
