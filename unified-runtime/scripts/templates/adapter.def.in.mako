<%!
import re
from templates import helper as th
%><%
    n=namespace
%>\
LIBRARY @TARGET_LIBNAME@
EXPORTS
%for tbl in th.get_pfntables(specs, meta, n, tags):
%if 'guard' in tbl:
#if ${tbl['guard']}
%endif
	${tbl['export']['name']}
%if 'guard' in tbl:
#endif // ${tbl['guard']}
%endif
%endfor
