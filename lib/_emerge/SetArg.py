# Copyright 1999-2012 Gentoo Foundation
# Distributed under the terms of the GNU General Public License v2

from _emerge.DependencyArg import DependencyArg
from portage._sets import SETPREFIX


class SetArg(DependencyArg):
    """ A 'set' dependency - e.g. from /etc/portage/sets - a set of package dependencies."""
    __slots__ = ("name", "pset")

    def __init__(self, pset=None, **kwargs):
        DependencyArg.__init__(self, **kwargs)
        self.pset = pset
        self.name = self.arg[len(SETPREFIX) :]
