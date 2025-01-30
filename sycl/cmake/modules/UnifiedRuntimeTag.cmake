set(UNIFIED_RUNTIME_REPO "https://github.com/RossBrunton/unified-runtime.git")
# commit 80fa413d390d07de396daa851d3fc2ff3ea8cb35
# Author: Ross Brunton <ross@codeplay.com>
# Date:   Mon Jan 27 12:34:34 2025 +0000
#     Remove virtual methods from ur_mem_handle_t_
#
#     We want to transition to handle pointers containing the ddi table as the
#     first element. For this to work, handle object must not have a vtable.
#
#     Since ur_mem_handle_t_ is relatively simple, it's easy enough to roll
#     out our own version of dynamic dispatch.
set(UNIFIED_RUNTIME_TAG 80fa413d390d07de396daa851d3fc2ff3ea8cb35)
