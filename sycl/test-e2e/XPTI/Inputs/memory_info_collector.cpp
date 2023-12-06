#include "xpti/xpti_trace_framework.hpp"

#include <algorithm>
#include <iostream>
#include <mutex>
#include <string_view>

std::mutex GMutex;

XPTI_CALLBACK_API void syclCallback(uint16_t, xpti::trace_event_data_t *,
                                    xpti::trace_event_data_t *, uint64_t,
                                    const void *);

XPTI_CALLBACK_API void memCallback(uint16_t, xpti::trace_event_data_t *,
                                   xpti::trace_event_data_t *, uint64_t,
                                   const void *);

XPTI_CALLBACK_API void syclBufferCallback(uint16_t, xpti::trace_event_data_t *,
                                          xpti::trace_event_data_t *, uint64_t,
                                          const void *);

XPTI_CALLBACK_API void syclImageCallback(uint16_t, xpti::trace_event_data_t *,
                                         xpti::trace_event_data_t *, uint64_t,
                                         const void *);

XPTI_CALLBACK_API void xptiTraceInit(unsigned int MajorVersion,
                                     unsigned int MinorVersion,
                                     const char *VersionStr,
                                     const char *StreamName) {
  std::cout << "xptiTraceInit: Stream Name = " << StreamName << "\n";
  std::string_view NameView{StreamName};

  using type = xpti::trace_point_type_t;
  if (NameView == "sycl.experimental.mem_alloc") {
    uint8_t StreamID = xptiRegisterStream(StreamName);
    for (type t : std::initializer_list<type>{
             type::mem_alloc_begin, type::mem_alloc_end,
             type::mem_release_begin, type::mem_release_end})
      xptiRegisterCallback(StreamID, static_cast<uint16_t>(t), memCallback);
  }

  auto buffer_image_traces = std::initializer_list<type>{
      type::offload_alloc_memory_object_construct,
      type::offload_alloc_memory_object_associate,
      type::offload_alloc_memory_object_release,
      type::offload_alloc_memory_object_destruct, type::offload_alloc_accessor};
  if (NameView == "sycl.experimental.buffer") {
    uint8_t StreamID = xptiRegisterStream(StreamName);
    for (type t : buffer_image_traces)
      xptiRegisterCallback(StreamID, static_cast<uint16_t>(t),
                           syclBufferCallback);
  }
  if (NameView == "sycl.experimental.image") {
    uint8_t StreamID = xptiRegisterStream(StreamName);
    for (type t : buffer_image_traces)
      xptiRegisterCallback(StreamID, static_cast<uint16_t>(t),
                           syclImageCallback);
  }
  if (NameView == "sycl") {
    uint8_t StreamID = xptiRegisterStream(StreamName);
    for (type t : std::initializer_list<type>{
             type::graph_create, type::node_create, type::edge_create,
             type::task_begin, type::task_end, type::signal, type::wait_begin,
             type::wait_end, type::barrier_begin, type::barrier_end,
             type::diagnostics})
      xptiRegisterCallback(StreamID, static_cast<uint16_t>(t), syclCallback);
  }
}

XPTI_CALLBACK_API void xptiTraceFinish(const char *streamName) {
  std::cout << "xptiTraceFinish: Stream Name = " << streamName << "\n";
}

XPTI_CALLBACK_API void memCallback(uint16_t TraceType,
                                   xpti::trace_event_data_t *,
                                   xpti::trace_event_data_t *, uint64_t,
                                   const void *UserData) {
  std::lock_guard Lock{GMutex};
  auto Type = static_cast<xpti::trace_point_type_t>(TraceType);
  auto *Data = static_cast<const xpti::mem_alloc_data_t *>(UserData);
  if (Type == xpti::trace_point_type_t::mem_alloc_begin) {
    std::cout << "Mem Alloc Begin : ";
  } else if (Type == xpti::trace_point_type_t::mem_alloc_end) {
    std::cout << "Mem Alloc End : ";
  } else if (Type == xpti::trace_point_type_t::mem_release_begin) {
    std::cout << "Mem Release Begin : ";
  } else if (Type == xpti::trace_point_type_t::mem_release_end) {
    std::cout << "Mem Release End : ";
  }
  std::cout << "mem_obj_handle:0x" << std::hex << Data->mem_object_handle;
  std::cout << "|alloc_pointer:0x" << Data->alloc_pointer;
  std::cout << "|alloc_size:" << std::dec << Data->alloc_size << std::endl;
}

XPTI_CALLBACK_API void syclBufferCallback(uint16_t TraceType,
                                          xpti::trace_event_data_t *Parent,
                                          xpti::trace_event_data_t *Event,
                                          uint64_t IId, const void *UserData) {
  std::lock_guard Lock{GMutex};
  auto Type = static_cast<xpti::trace_point_type_t>(TraceType);
  switch (Type) {
  case xpti::trace_point_type_t::offload_alloc_memory_object_construct: {
    auto BufConstr = (xpti::offload_buffer_data_t *)UserData;
    std::cout << IId << "|Create buffer|0x" << std::hex
              << BufConstr->user_object_handle << "|0x"
              << BufConstr->host_object_handle << "|" << std::dec
              << BufConstr->element_type << "|" << BufConstr->element_size
              << "|" << BufConstr->dim << "|"
              << "{" << BufConstr->range[0] << "," << BufConstr->range[1] << ","
              << BufConstr->range[2] << "}|"
              << Event->reserved.payload->source_file << ":"
              << Event->reserved.payload->line_no << ":"
              << Event->reserved.payload->column_no << "\n";

    break;
  }
  case xpti::trace_point_type_t::offload_alloc_memory_object_associate: {
    auto BufAssoc = (xpti::offload_association_data_t *)UserData;
    std::cout << IId << "|Associate buffer|0x" << std::hex
              << BufAssoc->user_object_handle << "|0x"
              << BufAssoc->mem_object_handle << std::dec << std::endl;
    break;
  }
  case xpti::trace_point_type_t::offload_alloc_memory_object_release: {
    auto BufRelease = (xpti::offload_association_data_t *)UserData;
    std::cout << IId << "|Release buffer|0x" << std::hex
              << BufRelease->user_object_handle << "|0x"
              << BufRelease->mem_object_handle << std::dec << std::endl;
    break;
  }
  case xpti::trace_point_type_t::offload_alloc_memory_object_destruct: {
    auto BufDestr = (xpti::offload_buffer_data_t *)UserData;
    std::cout << IId << "|Destruct buffer|0x" << std::hex
              << BufDestr->user_object_handle << std::dec << std::endl;
    break;
  }
  case xpti::trace_point_type_t::offload_alloc_accessor: {
    auto BufAccessor = (xpti::offload_accessor_data_t *)UserData;
    std::cout << IId << "|Construct accessor|0x" << std::hex
              << BufAccessor->buffer_handle << "|0x"
              << BufAccessor->accessor_handle << std::dec << "|"
              << BufAccessor->target << "|" << BufAccessor->mode << "|"
              << Event->reserved.payload->source_file << ":"
              << Event->reserved.payload->line_no << ":"
              << Event->reserved.payload->column_no << "\n";
    break;
  }
  default:
    std::cout << "Unknown tracepoint\n";
  }
}

XPTI_CALLBACK_API void syclImageCallback(uint16_t TraceType,
                                         xpti::trace_event_data_t *Parent,
                                         xpti::trace_event_data_t *Event,
                                         uint64_t IId, const void *UserData) {
  std::lock_guard Lock{GMutex};
  auto Type = static_cast<xpti::trace_point_type_t>(TraceType);
  switch (Type) {
  case xpti::trace_point_type_t::offload_alloc_memory_object_construct: {
    auto ImgConstr = (xpti::offload_image_data_t *)UserData;
    bool IsSampledImage = ImgConstr->addressing &&
                          ImgConstr->coordinate_normalization &&
                          ImgConstr->filtering;
    std::cout << IId << "|Create ";
    if (!IsSampledImage)
      std::cout << "un";
    std::cout << "sampled image|0x" << std::hex << ImgConstr->user_object_handle
              << "|0x" << ImgConstr->host_object_handle << "|" << std::dec
              << ImgConstr->dim << "|"
              << "{" << ImgConstr->range[0] << "," << ImgConstr->range[1] << ","
              << ImgConstr->range[2] << "}|" << ImgConstr->format << "|";
    if (IsSampledImage)
      std::cout << *ImgConstr->addressing << "|"
                << *ImgConstr->coordinate_normalization << "|"
                << *ImgConstr->filtering << "|";
    std::cout << Event->reserved.payload->source_file << ":"
              << Event->reserved.payload->line_no << ":"
              << Event->reserved.payload->column_no << "\n";

    break;
  }
  case xpti::trace_point_type_t::offload_alloc_memory_object_associate: {
    auto ImgAssoc = (xpti::offload_association_data_t *)UserData;
    std::cout << IId << "|Associate image|0x" << std::hex
              << ImgAssoc->user_object_handle << "|0x"
              << ImgAssoc->mem_object_handle << std::dec << std::endl;
    break;
  }
  case xpti::trace_point_type_t::offload_alloc_memory_object_release: {
    auto ImgRelease = (xpti::offload_association_data_t *)UserData;
    std::cout << IId << "|Release image|0x" << std::hex
              << ImgRelease->user_object_handle << "|0x"
              << ImgRelease->mem_object_handle << std::dec << std::endl;
    break;
  }
  case xpti::trace_point_type_t::offload_alloc_memory_object_destruct: {
    auto ImgDestr = (xpti::offload_image_data_t *)UserData;
    std::cout << IId << "|Destruct image|0x" << std::hex
              << ImgDestr->user_object_handle << std::dec << std::endl;
    break;
  }
  case xpti::trace_point_type_t::offload_alloc_accessor: {
    auto ImgAccessor = (xpti::offload_image_accessor_data_t *)UserData;
    // Host accessors do not have a target.
    bool IsHostAccessor = !ImgAccessor->target;
    // Only unsampled image accessors have a mode.
    bool IsUnsampledAccessor = bool(ImgAccessor->mode);
    std::cout << IId << "|Construct ";
    if (IsHostAccessor)
      std::cout << "host ";
    if (IsUnsampledAccessor)
      std::cout << "un";
    std::cout << "sampled image accessor|0x" << std::hex
              << ImgAccessor->image_handle << "|0x"
              << ImgAccessor->accessor_handle << std::dec << "|";
    if (!IsHostAccessor)
      std::cout << *ImgAccessor->target << "|";
    if (IsUnsampledAccessor)
      std::cout << *ImgAccessor->mode << "|";
    std::cout << ImgAccessor->element_type << "|" << ImgAccessor->element_size
              << "|" << Event->reserved.payload->source_file << ":"
              << Event->reserved.payload->line_no << ":"
              << Event->reserved.payload->column_no << "\n";
    break;
  }
  default:
    std::cout << "Unknown tracepoint\n";
  }
}

template <typename T>
T getMetadataByKey(xpti::metadata_t *Metadata, const char *key) {
  for (auto &Item : *Metadata) {
    if (std::string(xptiLookupString(Item.first)) == key) {
      return xpti::getMetadata<T>(Item).second;
    }
  }
  return {};
}

bool isMetadataPresent(xpti::metadata_t *Metadata, const char *key) {
  for (auto &Item : *Metadata) {
    if (std::string(xptiLookupString(Item.first)) == key) {
      return true;
    }
  }
  return false;
}
void parseMetadata(xpti::trace_event_data_t *Event) {
  xpti::metadata_t *Metadata = xptiQueryMetadata(Event);
  if (isMetadataPresent(Metadata, "kernel_name")) {
    std::cout << getMetadataByKey<std::string>(Metadata, "kernel_name") << "|";
  }
  if (isMetadataPresent(Metadata, "sym_source_file_name") &&
      isMetadataPresent(Metadata, "sym_line_no") &&
      isMetadataPresent(Metadata, "sym_column_no")) {
    std::cout << getMetadataByKey<std::string>(Metadata, "sym_source_file_name")
              << ":" << getMetadataByKey<int>(Metadata, "sym_line_no") << ":"
              << getMetadataByKey<int>(Metadata, "sym_column_no") << "|";
  }
  if (isMetadataPresent(Metadata, "enqueue_kernel_data")) {
    auto KernelEnqueueData =
        getMetadataByKey<xpti::offload_kernel_enqueue_data_t>(
            Metadata, "enqueue_kernel_data");

    std::cout << "{" << KernelEnqueueData.global_size[0] << ", "
              << KernelEnqueueData.global_size[1] << ", "
              << KernelEnqueueData.global_size[2] << "}, {"
              << KernelEnqueueData.local_size[0] << ", "
              << KernelEnqueueData.local_size[1] << ", "
              << KernelEnqueueData.local_size[2] << "}, {"
              << KernelEnqueueData.offset[0] << ", "
              << KernelEnqueueData.offset[1] << ", "
              << KernelEnqueueData.offset[2] << "}, "
              << KernelEnqueueData.args_num << "|\n";

    for (int i = 0; i < KernelEnqueueData.args_num; i++) {
      std::string Name("arg" + std::to_string(i));

      auto arg = getMetadataByKey<xpti::offload_kernel_arg_data_t>(
          Metadata, Name.c_str());
      std::cout << "  " << Name << " : {" << arg.type << ", " << std::hex
                << "0x" << (uintptr_t)arg.pointer << std::dec << ", "
                << arg.size << ", " << arg.index << "} "
                << "\n";
    }
  } else {
    std::cout << "\n";
  }
}
XPTI_CALLBACK_API void syclCallback(uint16_t TraceType,
                                    xpti::trace_event_data_t *,
                                    xpti::trace_event_data_t *Event, uint64_t,
                                    const void *UserData) {
  std::lock_guard Lock{GMutex};
  auto Type = static_cast<xpti::trace_point_type_t>(TraceType);
  switch (Type) {
  case xpti::trace_point_type_t::graph_create:
    std::cout << "Graph create|";
    break;
  case xpti::trace_point_type_t::node_create:
    std::cout << "Node create|";
    break;
  case xpti::trace_point_type_t::edge_create:
    std::cout << "Edge create|";
    break;
  case xpti::trace_point_type_t::task_begin:
    std::cout << "Task begin|";
    break;
  case xpti::trace_point_type_t::task_end:
    std::cout << "Task end|";
    break;
  case xpti::trace_point_type_t::signal:
    std::cout << "Signal|";
    break;
  case xpti::trace_point_type_t::wait_begin:
    std::cout << "Wait begin|";
    break;
  case xpti::trace_point_type_t::wait_end:
    std::cout << "Wait end|";
    break;
  case xpti::trace_point_type_t::barrier_begin:
    std::cout << "Barrier begin|";
    break;
  case xpti::trace_point_type_t::barrier_end:
    std::cout << "Barrier end|";
    break;
  default:
    std::cout << "Unknown tracepoint|";
  }
  parseMetadata(Event);
}
