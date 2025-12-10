"""
Z3 SMT Solver-based Dependency Resolution

This module provides an experimental alternative dependency resolver using the
Z3 SMT (Satisfiability Modulo Theories) solver. It demonstrates how constraint
solving can be applied to Portage's dependency resolution problem.

ARCHITECTURE:
    The Z3 solver operates at the select_files() level, encoding the entire
    dependency resolution problem as a Boolean satisfiability formula:

    1. Package variables: Each package version gets a Boolean variable
    2. Root constraints: User-requested packages must be satisfied
    3. Dependency implications: If pkg A is installed, its deps must be satisfied
    4. Slot constraints: Only one version per (cat/pkg, slot) tuple
    5. Blocker constraints: Incompatible packages cannot coexist

LIMITATIONS:
    - Basic DEPEND/RDEPEND only (no BDEPEND/PDEPEND/IDEPEND)
    - No USE flag dependency evaluation (uses default/forced flags only)
    - No OR dependencies || ( ) support
    - Basic slot constraints only (no subslot operators)
    - Simple blocker support (no uninstall reasoning)
    - No optimization for --update (just satisfiability)
    - No autounmask suggestions
    - No backtracking information beyond SAT/UNSAT

Enable with: PORTAGE_USE_Z3=1 ./bin/emerge <package>
Or: PORTAGE_USE_Z3=1 python -m pytest -v lib/portage/tests/resolver/test_or_choices.py
Requires: pip install z3-solver
"""

from typing import Any, Dict, List, Set, Tuple

import portage
from _emerge.Dependency import Dependency
from _emerge.Package import Package
from _emerge.PackageArg import PackageArg
from _emerge.SetArg import SetArg
from portage.dep import Atom, use_reduce
from portage.exception import InvalidAtom, InvalidDependString
from portage.output import colorize
from portage.util import writemsg_level, writemsg
from portage.versions import cpv_getkey

from z3 import And, Bool, Implies, Not, Or, Solver, sat, unsat

class Z3DepGraphResolver:
    """
    Z3-based dependency resolver for Portage.

    This class encodes Portage's dependency resolution problem as a Boolean
    satisfiability formula and uses Z3 to find a solution.
    """

    def __init__(self, depgraph):
        """
        Initialize Z3 resolver with reference to depgraph.

        Args:
            depgraph: The parent depgraph instance (for access to data structures)
        """
        self.depgraph = depgraph

        # Z3 solver instance
        self._solver: Solver = Solver()

        # Package -> Z3 Bool variable mapping
        self._pkg_vars: Dict[Any, Bool] = {}

        # Track packages by (cat/pkg, slot) for slot constraints
        self._slot_packages: Dict[Tuple[str, str], List[Any]] = {}

        # Track all candidate packages we've seen
        self._all_packages: Set[Any] = set()

        # Track installed packages for reference
        self._installed_packages: Set[Any] = set()

        # Statistics for debugging
        self._stats = {
            "packages_considered": 0,
            "constraints_added": 0,
            "atoms_processed": 0,
        }

    def resolve(self, myfavorites) -> Tuple[int, List[str]]:
        """
        Main entry point for Z3-based resolution.

        This mirrors the structure of _select_files() but skips a lot of error checking
        and uses Z3 for the core dependency resolution instead of the greedy
        stack-based approach.

        Args:
            myfiles: List of packages/atoms/sets requested by user

        Returns:
            Tuple of (success_code, favorites_list)
            - success_code: 1 for success, 0 for failure
            - favorites_list: List of atom strings for world file
        """
        # Get root configuration
        root_config = self.depgraph._frozen_config.roots[self.depgraph._frozen_config.target_root]
        eroot = root_config.root
        vardb = self.depgraph._frozen_config.trees[eroot]["vartree"].dbapi

        # Load installed packages
        #self._load_installed_packages(vardb)

        # Add root packages (Phase 1 of depgraph.py::_resolve())
        self.add_root_packages(myfavorites)

        writemsg(
            f">>> Z3: Resolving {len(root_atoms)} root atom(s)...\n",
            noiselevel=-1,
        )

        # Encode the problem as Z3 constraints
        self._encode_problem(root_atoms, root_config)

        # Solve with Z3
        result = self._solver.check()

        if result == sat:
            # Extract solution and populate depgraph
            new_packages = self._extract_solution()

            if not self._populate_depgraph(new_packages):
                return 0, myfavorites

            writemsg(
                colorize("GOOD", ">>> ")
                + f"Z3 solver found solution with {len(new_packages)} package(s)\n",
                noiselevel=-1,
            )

            # Statistics for debugging
            writemsg(
                f">>> Z3 stats: {self._stats['packages_considered']} packages considered, "
                f"{self._stats['constraints_added']} constraints added\n",
                noiselevel=-1,
            )

            return 1, myfavorites

        elif result == unsat:
            writemsg(
                colorize("BAD", "!!! ") + "Z3 solver: No solution exists (UNSAT)\n",
                noiselevel=-1,
            )
            # TODO: Extract UNSAT core for better error messages
            return 0, myfavorites

        else:
            writemsg(
                colorize("WARN", "!!! ") + f"Z3 solver: Unknown result ({result})\n",
                noiselevel=-1,
            )
            return 0, myfavorites

    def _load_installed_packages(self, vardb):
        """Load currently installed packages into internal set."""
        for cpv in vardb.cpv_all():
            try:
                pkg = self.depgraph._pkg(
                    cpv, "installed", root_config=None, installed=True
                )
                self._installed_packages.add(pkg)
            except Exception:
                # Skip packages we can't load
                pass

    def add_root_packages(self, myfavorites) -> Tuple[Any, List[str]]:
        """Given self.depgraph._dynamic_config._initial_arg_list, pull in the root nodes

        # Phase 1 ONLY of the depgraph::_resolve()

        Returns:
            root_atoms
        """
        debug = "--debug" in self.depgraph._frozen_config.myopts
        onlydeps = "--onlydeps" in self.depgraph._frozen_config.myopts
        args = self.depgraph._dynamic_config._initial_arg_list[:]

        # Essentially things on the command line expanded to PackageArg (SetArg/AtomArg are expanded properly)
        for arg in self.depgraph._expand_set_args(args, add_to_digraph=True):
            myroot = arg.root_config.root
            pkgsettings = self.depgraph._frozen_config.pkgsettings[myroot]
            pprovideddict = pkgsettings.pprovideddict
            virtuals = pkgsettings.getvirtuals()

            for atom in sorted(arg.pset.getAtoms()):
                self.depgraph._spinner_update()
                dep = Dependency(atom=atom, onlydeps=onlydeps, root=myroot, parent=arg)
                try:
                    pprovided = pprovideddict.get(atom.cp)
                    # "provided" package is a package declared in `package.provided` - pretend this is installed.
                    if pprovided and portage.match_from_list(atom, pprovided):
                        # A provided package has been specified on the command line.
                        self.depgraph._dynamic_config._pprovided_args.append((arg, atom))
                        continue
                    # if this is a .tbz2 or .ebuild file
                    if isinstance(arg, PackageArg):

                        # native depgraph: Immediately go into solver mode for each .ebuild or .tbz2 file
                        # Instead we add pacakge as a Z3 equation
                        self.add_pkg(arg.package, dep)
                        continue
                    if debug:
                        writemsg_level(
                            f"\n      Arg: {arg}\n     Atom: {atom}\n",
                            noiselevel=-1,
                            level=logging.DEBUG,
                        )
                    pkg, existing_node = self.depgraph._select_package(
                        myroot, atom, onlydeps=onlydeps
                    )

                    # Is the package installed (at any version)?
                    if pkg and "update_if_installed" in self.depgraph._dynamic_config.myparams:
                        package_is_installed = any(
                            self.depgraph._iter_match_pkgs(
                                self.depgraph._frozen_config.roots[myroot], "installed", atom
                            )
                        )

                        # This package isn't eligible for selection in the
                        # merge list as the user passed --update-if-installed
                        # and it isn't installed.
                        if not package_is_installed:
                            continue

                    if not pkg:
                        pprovided_match = False
                        for virt_choice in virtuals.get(atom.cp, []):
                            expanded_atom = portage.dep.Atom(
                                atom.replace(atom.cp, virt_choice.cp, 1)
                            )
                            pprovided = pprovideddict.get(expanded_atom.cp)
                            if pprovided and portage.match_from_list(
                                expanded_atom, pprovided
                            ):
                                # A provided package has been
                                # specified on the command line.
                                self.depgraph._dynamic_config._pprovided_args.append((arg, atom))
                                pprovided_match = True
                                break
                        if pprovided_match:
                            continue

                        excluded = False
                        for any_match in self.depgraph._iter_match_pkgs_any(
                            self.depgraph._frozen_config.roots[myroot], atom
                        ):
                            if self.depgraph._frozen_config.excluded_pkgs.findAtomForPackage(
                                any_match, modified_use=self.depgraph._pkg_use_enabled(any_match)
                            ):
                                excluded = True
                                break
                        if excluded:
                            continue

                        if not (
                            isinstance(arg, SetArg)
                            and arg.name in ("selected", "world")
                        ):
                            self.depgraph._dynamic_config._unsatisfied_deps_for_display.append(
                                ((myroot, atom), {"myparent": arg})
                            )
                            return 0, myfavorites

                        self.depgraph._dynamic_config._missing_args.append((arg, atom))
                        continue
                    if atom.cp != pkg.cp:
                        # For old-style virtuals, we need to repeat the
                        # package.provided check against the selected package.
                        expanded_atom = atom.replace(atom.cp, pkg.cp)
                        pprovided = pprovideddict.get(pkg.cp)
                        if pprovided and portage.match_from_list(
                            expanded_atom, pprovided
                        ):
                            # A provided package has been
                            # specified on the command line.
                            self.depgraph._dynamic_config._pprovided_args.append((arg, atom))
                            continue
                    if (
                        pkg.installed
                        and "selective" not in self.depgraph._dynamic_config.myparams
                        and not self.depgraph._frozen_config.excluded_pkgs.findAtomForPackage(
                            pkg, modified_use=self.depgraph._pkg_use_enabled(pkg)
                        )
                    ):
                        self.depgraph._dynamic_config._unsatisfied_deps_for_display.append(
                            ((myroot, atom), {"myparent": arg})
                        )
                        # Previous behavior was to bail out in this case, but
                        # since the dep is satisfied by the installed package,
                        # it's more friendly to continue building the graph
                        # and just show a warning message. Therefore, only bail
                        # out here if the atom is not from either the system or
                        # world set.
                        if not (
                            isinstance(arg, SetArg)
                            and arg.name in ("selected", "system", "world")
                        ):
                            return 0, myfavorites

                    # Add the selected package to the graph as soon as possible
                    # so that later dep_check() calls can use it as feedback
                    # for making more consistent atom selections.
                    self.add_pkg(pkg, dep)


                except SystemExit as e:
                    raise  # Needed else can't exit
                except Exception as e:
                    writemsg(
                        f"\n\n!!! Problem in '{atom}' dependencies.\n", noiselevel=-1
                    )
                    writemsg(f"!!! {str(e)} {str(getattr(e, '__module__', None))}\n")
                    raise

    def add_pkg(self, pkg:Package, dep:Dependency):
        if pkg in self._all_packages:
            return

        self._all_packages.add(pkg)
        self._stats["packages_considered"] += 1

        # Track by slot for slot constraints
        cp = cpv_getkey(pkg.cpv)
        slot = pkg.slot
        slot_key = (cp, slot)
        self._slot_packages.setdefault(slot_key, []).append(pkg)

        # # Queue dependencies for discovery
        # # (POC: only DEPEND and RDEPEND, no USE flag evaluation)
        # for dep_type in ("DEPEND", "RDEPEND"):
        #     dep_str = pkg._metadata.get(dep_type, "")
        #     if dep_str:
        #         dep_atoms = self._parse_dep_string(dep_str, pkg)
        #         queue.extend(dep_atoms)

    def _encode_problem(self, root_atoms: List[Atom], root_config):
        """
        Encode the dependency resolution problem as Z3 constraints.

        This builds up the constraint system:
        1. Find all candidate packages for root atoms and their dependencies
        2. Add root constraints (these atoms must be satisfied)
        3. Add dependency implications
        4. Add slot constraints
        5. Add blocker constraints
        """
        writemsg(">>> Z3: Encoding constraints...\n", noiselevel=-1)

        # Phase 2: Encode root constraints
        self._encode_root_constraints(root_atoms)

        # Phase 3: Encode dependency implications
        self._encode_dependency_constraints(root_config)

        # Phase 4: Encode slot constraints
        self._encode_slot_constraints()

        # Phase 5: Encode blocker constraints
        self._encode_blocker_constraints(root_config)

        writemsg(
            f">>> Z3: Encoding complete ({self._stats['constraints_added']} constraints)\n",
            noiselevel=-1,
        )


    def _find_matching_packages(self, atom: Atom, root_config) -> List[Any]:
        """Find all packages in the tree that match the given atom."""
        matches = []
        eroot = root_config.root

        # Search in portage tree
        try:
            for pkg in self.depgraph._iter_match_pkgs(root_config, "ebuild", atom):
                matches.append(pkg)
        except Exception:
            pass

        # Also check binary packages
        try:
            for pkg in self.depgraph._iter_match_pkgs(root_config, "binary", atom):
                matches.append(pkg)
        except Exception:
            pass

        return matches

    def _parse_dep_string(self, dep_str: str, pkg) -> List[Atom]:
        """
        Parse a dependency string into a list of atoms.

        POC limitations:
        - No USE flag evaluation (uses default/forced flags)
        - No OR dependencies
        - Ignores blockers in discovery phase
        """
        atoms = []

        try:
            # use_reduce flattens the dependency tree
            # For POC, we ignore USE flags and OR deps
            use_reduce_result = use_reduce(
                dep_str,
                uselist=[],  # Empty USE flags for POC
                flat=True,
                token_class=Atom,
            )

            for item in use_reduce_result:
                if isinstance(item, Atom):
                    # Skip blockers in discovery (we handle them separately)
                    if not item.blocker:
                        atoms.append(item)

        except (InvalidDependString, InvalidAtom) as e:
            # Skip packages with invalid dependencies
            pass

        return atoms

    def _get_pkg_var(self, pkg) -> Bool:
        """Get or create Z3 Boolean variable for a package."""
        if pkg not in self._pkg_vars:
            # Create unique variable name: p_category/package-version_slot
            var_name = f"p_{pkg.cpv.replace('/', '_')}_{pkg.slot.replace('/', '_')}"
            self._pkg_vars[pkg] = Bool(var_name)
        return self._pkg_vars[pkg]

    def _encode_root_constraints(self, root_atoms: List[Atom]):
        """
        Encode constraints for user-requested atoms.

        Each root atom must be satisfied by at least one package.
        """
        for atom in root_atoms:
            # Find all packages that satisfy this atom
            matching_vars = []

            for pkg in self._all_packages:
                if atom.match(pkg):
                    matching_vars.append(self._get_pkg_var(pkg))

            if matching_vars:
                # At least one matching package must be installed
                constraint = (
                    Or(*matching_vars) if len(matching_vars) > 1 else matching_vars[0]
                )
                self._solver.add(constraint)
                self._stats["constraints_added"] += 1
            else:
                # No packages match - this will make the problem UNSAT
                writemsg(
                    colorize("WARN", "!!! ") + f"No packages match root atom: {atom}\n",
                    noiselevel=-1,
                )

    def _encode_dependency_constraints(self, root_config):
        """
        Encode dependency implications.

        For each package, if it's installed, then its dependencies must be satisfied.
        POC: Only handles DEPEND and RDEPEND, ignores USE flags and OR deps.
        """
        for pkg in self._all_packages:
            pkg_var = self._get_pkg_var(pkg)

            for dep_type in ("DEPEND", "RDEPEND"):
                dep_str = pkg._metadata.get(dep_type, "")
                if not dep_str:
                    continue

                # Parse dependencies into atoms
                dep_atoms = self._parse_dep_string(dep_str, pkg)

                for atom in dep_atoms:
                    self._stats["atoms_processed"] += 1

                    # Find packages satisfying this dependency
                    matching_vars = []
                    for dep_pkg in self._all_packages:
                        if atom.match(dep_pkg):
                            matching_vars.append(self._get_pkg_var(dep_pkg))

                    if matching_vars:
                        # If pkg is installed, at least one dep must be satisfied
                        dep_constraint = (
                            Or(*matching_vars)
                            if len(matching_vars) > 1
                            else matching_vars[0]
                        )
                        self._solver.add(Implies(pkg_var, dep_constraint))
                        self._stats["constraints_added"] += 1

    def _encode_slot_constraints(self):
        """
        Encode slot constraints.

        Only one package version per (category/package, slot) can be installed.
        This is encoded as pairwise mutual exclusion.
        """
        for (cp, slot), packages in self._slot_packages.items():
            if len(packages) <= 1:
                continue

            pkg_vars = [self._get_pkg_var(pkg) for pkg in packages]

            # Add pairwise exclusion: not (pkg_i AND pkg_j) for all i != j
            for i in range(len(pkg_vars)):
                for j in range(i + 1, len(pkg_vars)):
                    self._solver.add(Not(And(pkg_vars[i], pkg_vars[j])))
                    self._stats["constraints_added"] += 1

    def _encode_blocker_constraints(self, root_config):
        """
        Encode blocker constraints.

        POC: Simple blocker support - if pkg A has blocker !B, then
        A and B cannot both be installed.

        Limitations:
        - No uninstall reasoning
        - No strong (!!) vs weak (!) blocker distinction
        """
        for pkg in self._all_packages:
            pkg_var = self._get_pkg_var(pkg)

            for dep_type in ("DEPEND", "RDEPEND"):
                dep_str = pkg._metadata.get(dep_type, "")
                if not dep_str:
                    continue

                try:
                    # Parse with blockers included
                    use_reduce_result = use_reduce(
                        dep_str,
                        uselist=[],
                        flat=True,
                        token_class=Atom,
                    )

                    for item in use_reduce_result:
                        if isinstance(item, Atom) and item.blocker:
                            # Find packages blocked by this atom
                            for blocked_pkg in self._all_packages:
                                # Remove blocker operator to match
                                blocker_atom = Atom(str(item).lstrip("!"))
                                if blocker_atom.match(blocked_pkg):
                                    blocked_var = self._get_pkg_var(blocked_pkg)
                                    # pkg and blocked_pkg cannot both be installed
                                    self._solver.add(Not(And(pkg_var, blocked_var)))
                                    self._stats["constraints_added"] += 1

                except (InvalidDependString, InvalidAtom):
                    pass

    def _extract_solution(self) -> List[Any]:
        """
        Extract the solution from Z3 model.

        Returns list of packages that should be installed (excluding already
        installed packages).
        """
        model = self._solver.model()
        new_packages = []

        for pkg, var in self._pkg_vars.items():
            # Check if this package is True in the solution
            if model.evaluate(var):
                # Only include if not already installed
                if pkg not in self._installed_packages:
                    new_packages.append(pkg)

        return new_packages

    def _populate_depgraph(self, packages: List[Any]) -> bool:
        """
        Populate the depgraph with the solution packages.

        This adds the packages to the depgraph's internal structures so that
        the native Phase 3 (conflict resolution and serialization) can work.

        Returns True on success, False on failure.
        """
        try:
            # Add each package to the depgraph
            for pkg in packages:
                # Create a dependency object for this package
                dep = self.depgraph._add_pkg_deps

                # Add to the graph
                if not self.depgraph._add_pkg(pkg, None):
                    writemsg(
                        colorize("WARN", "!!! ")
                        + f"Failed to add package to graph: {pkg.cpv}\n",
                        noiselevel=-1,
                    )
                    return False

            return True

        except Exception as e:
            writemsg(
                colorize("WARN", "!!! ") + f"Error populating depgraph: {e}\n",
                noiselevel=-1,
            )
            return False
