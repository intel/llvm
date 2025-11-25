#pragma once
#include "spirv_vars.h"
// super block structure
union superblk {
  unsigned long SuperblkMetadata;
  struct {
    unsigned char block_enable_partition_info : 8;
    unsigned char partition_block_occupancy_tracker
        : 8; // bits used for tracking blks
    unsigned short alloc_length_tracker_bits : 16; // bits used for tracking
                                                   // blks
    unsigned int alloc_size : 32;

  } fields;
};

struct random_walk_params_t {
  int num_walks;
  int walk_length;
  int index_range_start;
  int index_range_end;
  int step_size;
  int initial_pos;

  unsigned long long base_seed;
  unsigned long long base_subseed;
};

struct device_heap_t {
  unsigned long base;
  unsigned long size;
  unsigned int blocksize;
  unsigned int max_num_blocks;

  unsigned int max_num_heap_partitions;
  unsigned long *ptr_superblk;
  struct random_walk_params_t *prwalkparams;
  int *prwalk_path;
};

DeviceGlobal<void *> __DeviceHeapPtr;

#if defined(__SPIR__) || defined(__SPIRV__)

#define __SYCL_CONSTANT__ __attribute__((opencl_constant))

static const __SYCL_CONSTANT__ char __malloc_prwalk_debug[] =
    "[kernel] random walk params fileds: num_walks=%d, walk_length=%d, "
    "index_range_start=%d, index_range_end=%d, step_size=%d\n";

extern SYCL_EXTERNAL int
__spirv_ocl_printf(const __SYCL_CONSTANT__ char *Format, ...);
DEVICE_EXTERN_C
void *malloc(size_t size) {
  void *temp = __DeviceHeapPtr.get();
  device_heap_t *device_heap_ptr = reinterpret_cast<device_heap_t *>(temp);
  random_walk_params_t *prwalk = device_heap_ptr->prwalkparams;

  // Debug print in device code
  __spirv_ocl_printf(__malloc_prwalk_debug, prwalk->num_walks,
                     prwalk->walk_length, prwalk->index_range_start,
                     prwalk->index_range_end, prwalk->step_size);
  return reinterpret_cast<void *>(device_heap_ptr->base);
}

DEVICE_EXTERN_C
void free(void *ptr) { return; }
#endif
