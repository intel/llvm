# commit f6d00b8a95ddc41e17ac2faeba13afacd48252d2
# Author: Harald van Dijk <harald.vandijk@codeplay.com>
# Date:   Fri Nov 1 11:08:39 2024 +0000
#
#     Avoid potential ambiguity in UrReturnHelper
#
#     If UrReturnHelper's operator() is explicitly called as
#     operator()<T>(...), there is a potential for ambiguity when the
#     specified RetType and the inferred T are the same: this is ambiguous
#     with the version of operator() where only T is a template parameter,
#     and T is specified explicitly. We already have code that explicitly
#     calls operator()<T>(...), so prevent this from becoming a problem.
set(UNIFIED_RUNTIME_TAG f6d00b8a95ddc41e17ac2faeba13afacd48252d2)
