# commit 56dd5137d3e0e39576291d72e216e94dd18b2f71
# Author: Harald van Dijk <harald.vandijk@codeplay.com>
# Date:   Thu Feb 13 16:28:42 2025 +0000
# 
#     [NativeCPU] Fix kernel argument passing.
#     
#     We were reading the kernel arguments at kernel execution time, but kernel
#     arguments are allowed to change between enqueuing and executing. Make
#     sure to create a copy of kernel arguments ahead of time.
set(UNIFIED_RUNTIME_TAG 56dd5137d3e0e39576291d72e216e94dd18b2f71)
