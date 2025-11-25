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
