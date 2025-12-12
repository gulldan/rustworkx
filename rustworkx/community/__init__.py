# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# This module re-exports community detection functions from the Rust extension.
# The actual implementation is in src/community/

from rustworkx.rustworkx import community as _community

# Re-export all functions from the native module
louvain_communities = _community.louvain_communities
modularity = _community.modularity
leiden_communities = _community.leiden_communities
cpm_communities = _community.cpm_communities
find_maximal_cliques = _community.find_maximal_cliques
asyn_lpa_communities = _community.asyn_lpa_communities
asyn_lpa_communities_strongest = _community.asyn_lpa_communities_strongest

__all__ = [
    "louvain_communities",
    "modularity",
    "leiden_communities",
    "cpm_communities",
    "find_maximal_cliques",
    "asyn_lpa_communities",
    "asyn_lpa_communities_strongest",
]
