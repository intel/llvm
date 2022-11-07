<%
import os

apidocs = []
for section in sections:
    apidocs.append(section + "/api.rst")
%>
===================
 API Documentation
===================

.. toctree::

%for apidoc in apidocs:
%if os.path.exists(os.path.join(sourcepath, apidoc)):
    ${apidoc}
%endif
%endfor