#commit 3dbf8b247a6f42bfed1db6e6bdfdfd0b0f1067fc 
#Author: Zhang, Winston <winston.zhang@intel.com>
#Date:   Wed Jan 29 16:52:47 2025 -0800
#
#[L0] MAX_COMPUTE_UNITS using ze_eu_count_ext_t
#
#For some recovery SKUs, MAX_COMPUTE_COUNT calculation does not provide the correct number of EUs. Now we will  use ze_eu_count_t when
#available.
set(UNIFIED_RUNTIME_TAG 3dbf8b247a6f42bfed1db6e6bdfdfd0b0f1067fc)
