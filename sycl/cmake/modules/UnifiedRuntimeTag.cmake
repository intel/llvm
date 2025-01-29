set(UNIFIED_RUNTIME_REPO "https://github.com/RossBrunton/unified-runtime.git")
# commit a7cd75633517d96f87ea2172f95549a3d1e5c1bd
# Author: Ross Brunton <ross@codeplay.com>
# Date:   Tue Jan 28 13:17:06 2025 +0000
#     Don't use inheritence for L0 V2 event handles
#
#     In the future, we are going to require that handle objects have no
#     vtable, so this change combines "native" and "pooled" event into one
#     class. A variant is used to determine whether the event is native or
#     pooled.
#
#     For consistency, parameter order in constructors have been changed to
#     always start with the context.
set(UNIFIED_RUNTIME_TAG a7cd75633517d96f87ea2172f95549a3d1e5c1bd)
