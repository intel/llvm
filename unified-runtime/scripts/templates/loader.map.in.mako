<%!
import re
from templates import helper as th
%><%
    n=namespace
%>\
@TARGET_LIBNAME@ {
	global:
%for func in th.get_loader_functions(specs, meta, n, tags):
%if 'guard' in func:
#if ${func['guard']}
%endif
		${func['name']};
%if 'guard' in func:
#endif // ${func['guard']}
%endif
%endfor
	local:
		*;
};
