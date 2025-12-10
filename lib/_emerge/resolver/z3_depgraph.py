"""
Z3 SMT Solver-based Dependency Resolution

This module provides an experimental alternative dependency resolver using the
Z3 SMT (Satisfiability Modulo Theories) solver. It demonstrates how constraint
solving can be applied to Portage's dependency resolution problem.

ARCHITECTURE:
    The Z3 solver uses incremental constraint encoding that mirrors the native
    depgraph flow:

    Phase 1 (add_root_packages):
        - Expand set args to atoms
        - For each atom, select best package
        - Call add_pkg() which recursively:
          * Creates Z3 variable for package
          * Adds slot constraints (mutual exclusion)
          * Processes dependencies and adds implications
          * Handles blockers
        - Mark root packages as required

    Phase 2 (solve):
        - Call Z3 solver on accumulated constraints
        - Extract solution if SAT

    Phase 3 (populate and validate):
        - Add solution packages to native depgraph
        - Call native altlist() for conflict resolution and merge order

    Constraints:
        1. Package variables: Each package version gets a Boolean variable
        2. Root constraints: User-requested packages must be True
        3. Slot constraints: Only one version per (cat/pkg, slot) - added incrementally
        4. Dependency implications: pkg → (dep1 OR dep2 OR ...) - added recursively
        5. Blocker constraints: pkg → NOT(blocked) - added after dependencies

Enable with: PORTAGE_USE_Z3=1 ./bin/emerge <package>
Or: PORTAGE_USE_Z3=1 python -m pytest -v lib/portage/tests/resolver/test_or_choices.py
Requires: pip install z3-solver
"""

import logging
from typing import Any, Dict, List, Set, Tuple, Optional

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

from z3 import And, Bool, Implies, Not, Or, Solver, Optimize, sat, unsat

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

        # Z3 optimizer instance (supports both hard and soft constraints)
        self._solver: Optimize = Optimize()

        # Package -> Z3 Bool variable mapping
        self._pkg_vars: Dict[Any, Bool] = {}

        # Track packages by (cat/pkg, slot) for incremental slot constraints
        self._slot_packages: Dict[Tuple[str, str], List[Any]] = {}

        # Track all packages we've processed
        self._all_packages: Set[Any] = set()

        # Track installed packages for reference
        self._installed_packages: Set[Any] = set()
        
        # Package cache to deduplicate Package objects: (cpv, type_name) -> Package
        # This ensures we only have one Package instance per (cpv, "installed"/"ebuild")
        self._pkg_cache: Dict[Tuple[str, str], Any] = {}

        # Statistics for debugging
        self._stats = {
            "packages_considered": 0,
            "constraints_added": 0,
            "atoms_processed": 0,
        }

    def resolve(self, myfavorites) -> Tuple[int, List[str]]:
        """
        Main entry point for Z3-based resolution.

        Mirrors depgraph._resolve() flow:
            Phase 1: Add root packages (adds all constraints incrementally)
            Phase 2: Solve with Z3
            Phase 3: Populate depgraph (skip native conflict resolution - Z3 already solved it!)

        Args:
            myfavorites: List of favorite atoms for world file

        Returns:
            Tuple of (success_code, favorites_list)
            - success_code: 1 for success, 0 for failure
            - favorites_list: List of atom strings for world file
        """
        writemsg(">>> Z3: Starting dependency resolution...\n", noiselevel=-1)

        # Phase 0: Load installed packages and mark them as already satisfied
        self._load_installed_packages()

        # Phase 1: Add root packages and encode constraints incrementally
        if not self.add_root_packages(myfavorites):
            return 0, myfavorites

        writemsg(
            f">>> Z3: Encoded {self._stats['packages_considered']} packages, "
            f"{self._stats['constraints_added']} constraints\n",
            noiselevel=-1,
        )

        # Add optimization preferences for --update mode
        if "--update" in self.depgraph._frozen_config.myopts:
            self._add_update_preferences()
        
        # Phase 2: Solve with Z3
        writemsg(">>> Z3: Solving...\n", noiselevel=-1)
        result = self._solver.check()

        if result == sat:
            # Extract solution
            new_packages = self._extract_solution()

            writemsg(
                colorize("GOOD", ">>> ")
                + f"Z3 found solution with {len(new_packages)} new package(s)\n",
                noiselevel=-1,
            )

            # Show what Z3 selected
            writemsg(">>> Z3 selected packages:\n", noiselevel=-1)
            for pkg in sorted(new_packages, key=lambda p: p.cpv):
                writemsg(f"    {pkg.cpv}\n", noiselevel=-1)

            # Phase 3: Populate depgraph with solution
            # Add packages to digraph and set serialized cache directly
            for pkg in new_packages:
                self.depgraph._dynamic_config.digraph.add(pkg, None)
            
            # Set the serialized tasks cache so it can be displayed
            self.depgraph._dynamic_config._serialized_tasks_cache = tuple(new_packages)

            return 1, myfavorites

        elif result == unsat:
            writemsg(
                colorize("BAD", "!!! ") + "Z3 solver: No solution exists (UNSAT)\n",
                noiselevel=-1,
            )
            writemsg(
                f"!!! Packages considered: {self._stats['packages_considered']}, "
                f"constraints: {self._stats['constraints_added']}\n",
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

    def _load_installed_packages(self):
        """
        Load currently installed packages and add them to Z3 problem.

        Installed packages are:
        1. Added to the problem (with Z3 variables)
        2. NOT marked as True - Z3 decides whether to keep or upgrade them
        3. Subject to same slot constraints

        This allows Z3 to see them as candidates for satisfying dependencies.
        If an installed package satisfies the constraints, Z3 will select it.
        If it needs upgrading, Z3 will select a newer version instead.
        """
        root_config = self.depgraph._frozen_config.roots[
            self.depgraph._frozen_config.target_root
        ]
        eroot = root_config.root
        vardb = self.depgraph._frozen_config.trees[eroot]["vartree"].dbapi

        installed_count = 0
        for cpv in vardb.cpv_all():
            try:
                # Get the installed package
                pkg = self.depgraph._pkg(cpv, "installed", root_config)
                
                # Deduplicate
                pkg = self._get_or_cache_pkg(pkg)

                # Add to installed tracking only (NOT _all_packages!)
                # We'll add to _all_packages when we actually process via add_pkg()
                # so that dependencies get discovered
                self._installed_packages.add(pkg)

                # Pre-create Z3 variable so it exists
                pkg_var = self._get_pkg_var(pkg)

                installed_count += 1

            except Exception as e:
                # Skip packages we can't load
                pass

        writemsg(
            f">>> Z3: Loaded {installed_count} installed packages\n",
            noiselevel=-1
        )

    def add_root_packages(self, myfavorites) -> bool:
        """
        Phase 1 of depgraph::_resolve() - add root packages and encode constraints.

        This is copied from depgraph._resolve() but calls our add_pkg() instead
        of the native _add_pkg(). Our add_pkg() recursively discovers dependencies
        and encodes Z3 constraints incrementally.

        Returns:
            True on success, False on failure
        """
        debug = "--debug" in self.depgraph._frozen_config.myopts
        onlydeps = "--onlydeps" in self.depgraph._frozen_config.myopts
        args = self.depgraph._dynamic_config._initial_arg_list[:]

        # Expand set args and iterate over atoms
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
                    # "provided" package is declared in package.provided - pretend installed
                    if pprovided and portage.match_from_list(atom, pprovided):
                        self.depgraph._dynamic_config._pprovided_args.append((arg, atom))
                        continue

                    # Handle .tbz2 or .ebuild files
                    if isinstance(arg, PackageArg):
                        # native depgraph: Immediately go into solver mode for each .ebuild or .tbz2 file
                        # Instead we add pacakge as a Z3 equation
                        if not self.add_pkg(arg.package, dep):
                            writemsg(
                                f"\n\n!!! Problem resolving dependencies for {arg.arg}\n",
                                noiselevel=-1,
                            )
                            return False

                        # Mark as root package (must be installed)
                        pkg_var = self._get_pkg_var(arg.package)
                        self._solver.add(pkg_var)
                        self._stats["constraints_added"] += 1
                        continue

                    if debug:
                        writemsg_level(
                            f"\n      Arg: {arg}\n     Atom: {atom}\n",
                            noiselevel=-1,
                            level=logging.DEBUG,
                        )

                    # Select best package for this atom
                    pkg, existing_node = self.depgraph._select_package(
                        myroot, atom, onlydeps=onlydeps
                    )

                    # Check --update-if-installed
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

                    # Handle case where no package found
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
                            return False

                        self.depgraph._dynamic_config._missing_args.append((arg, atom))
                        continue
                    if atom.cp != pkg.cp:
                        expanded_atom = atom.replace(atom.cp, pkg.cp)
                        pprovided = pprovideddict.get(pkg.cp)
                        if pprovided and portage.match_from_list(
                            expanded_atom, pprovided
                        ):
                            # A provided package has been
                            # specified on the command line.
                            self.depgraph._dynamic_config._pprovided_args.append((arg, atom))
                            continue

                    # Handle already installed packages in non-selective mode
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
                            return False

                    # Add ALL packages matching this atom (not just the selected one)
                    # This ensures Z3 has dependency constraints for all candidates
                    root_config_for_atom = self.depgraph._frozen_config.roots[pkg.root]
                    matching_for_atom = self._find_matching_packages(atom, root_config_for_atom)
                    
                    if not matching_for_atom:
                        if isinstance(arg, SetArg):
                            writemsg(
                                f"\n\n!!! Problem resolving dependencies for {atom} from "
                                f"{arg.arg}\n",
                                noiselevel=-1,
                            )
                        else:
                            writemsg(
                                f"\n\n!!! Problem resolving dependencies for {atom}\n",
                                noiselevel=-1,
                            )
                        return False
                    
                    # Add each matching package (and its dependencies)
                    # If --deep is not set and an installed package satisfies the atom, 
                    # only add that installed package (don't explore ebuild alternatives)
                    deep = "--deep" in self.depgraph._frozen_config.myopts
                    has_installed = any(
                        p.type_name == "installed" if hasattr(p, 'type_name') else p.installed 
                        for p in matching_for_atom
                    )
                    
                    added_packages = []  # Track which packages we actually added
                    for match_pkg in matching_for_atom:
                        match_is_installed = match_pkg.type_name == "installed" if hasattr(match_pkg, 'type_name') else match_pkg.installed
                        
                        # Skip ebuild versions if not --deep and installed version exists
                        if not deep and has_installed and not match_is_installed:
                            continue
                        
                        if not self.add_pkg(match_pkg, dep):
                            if isinstance(arg, SetArg):
                                writemsg(
                                    f"\n\n!!! Problem resolving dependencies for {atom} from "
                                    f"{arg.arg}\n",
                                    noiselevel=-1,
                                )
                            else:
                                writemsg(
                                    f"\n\n!!! Problem resolving dependencies for {atom}\n",
                                    noiselevel=-1,
                                )
                            return False
                        
                        added_packages.append(match_pkg)

                    # Mark atom as required (at least one package satisfying this atom must be present)
                    # Create constraint: at least one of the ADDED packages must be selected
                    matching_vars = [self._get_pkg_var(p) for p in added_packages]
                    atom_constraint = Or(*matching_vars) if len(matching_vars) > 1 else matching_vars[0]
                    self._solver.add(atom_constraint)
                    self._stats["constraints_added"] += 1

                except SystemExit:
                    raise
                except Exception as e:
                    writemsg(
                        f"\n\n!!! Problem in '{atom}' dependencies.\n", noiselevel=-1
                    )
                    writemsg(f"!!! {str(e)} {str(getattr(e, '__module__', None))}\n")
                    raise

        return True

    def add_pkg(self, pkg: Package, dep: Dependency) -> bool:
        """
        Add a package to the Z3 problem, encoding its constraints incrementally.

        This mirrors depgraph._add_pkg() but adds Z3 constraints instead of
        manipulating the digraph.

        Incrementally adds:
            1. Package Z3 variable
            2. Slot constraints (mutual exclusion with same-slot packages)
            3. Dependency implications (recursive)
            4. Blocker constraints

        Returns:
            True on success, False on failure
        """
        # Deduplicate package objects - ensure we only have one instance per (cpv, type)
        pkg = self._get_or_cache_pkg(pkg)
        
        # Skip if already processed
        if pkg in self._all_packages:
            return True
        self._all_packages.add(pkg)
        self._stats["packages_considered"] += 1

        # Get or create Z3 variable for this package
        pkg_var = self._get_pkg_var(pkg)

        # 1. ADD SLOT CONSTRAINTS (incremental)
        cp = cpv_getkey(pkg.cpv)
        slot_key = (cp, pkg.slot)
        existing_pkgs = self._slot_packages.get(slot_key, [])

        # Add mutual exclusion with all existing packages in same slot
        for existing_pkg in existing_pkgs:
            existing_var = self._get_pkg_var(existing_pkg)
            self._solver.add(Not(And(pkg_var, existing_var)))
            self._stats["constraints_added"] += 1

        # Track this package in the slot
        self._slot_packages.setdefault(slot_key, []).append(pkg)
        
        # Check if we should process dependencies for this package
        # Without --deep, don't process dependencies of installed packages
        deep = "--deep" in self.depgraph._frozen_config.myopts
        pkg_is_installed = pkg.type_name == "installed" if hasattr(pkg, 'type_name') else pkg.installed
        should_process_deps = deep or not pkg_is_installed

        # 1b. ADD CPV MUTUAL EXCLUSION (installed vs ebuild versions)
        # If an installed and ebuild version of the same CPV exist, only one can be selected
        # Check both _all_packages (already processed) and _installed_packages (pre-loaded)
        all_pkgs = self._all_packages | self._installed_packages
        pkg_is_installed = pkg.type_name == "installed" if hasattr(pkg, 'type_name') else pkg.installed
        for existing_pkg in all_pkgs:
            existing_is_installed = existing_pkg.type_name == "installed" if hasattr(existing_pkg, 'type_name') else existing_pkg.installed
            if (existing_pkg.cpv == pkg.cpv and 
                existing_is_installed != pkg_is_installed):
                # One is installed, one is ebuild - they're mutually exclusive
                existing_var = self._get_pkg_var(existing_pkg)
                self._solver.add(Not(And(pkg_var, existing_var)))
                self._stats["constraints_added"] += 1

        # 2. ADD DEPENDENCY CONSTRAINTS (recursive expansion)
        # Only process dependencies if we should (skip installed packages without --deep)
        if should_process_deps:
            if not self._add_pkg_deps(pkg, dep):
                return False

            # 3. ADD BLOCKER CONSTRAINTS (after dependencies discovered)
            if not self._add_pkg_blockers(pkg):
                return False

        return True

    def _add_pkg_deps(self, pkg: Package, dep: Dependency) -> bool:
        """
        Process dependencies for a package and add constraint implications.

        Mirrors depgraph._add_pkg_deps() but:
            - Discovers dependency packages recursively
            - Adds Z3 implications: pkg → (dep1 OR dep2 OR ...)

        Returns:
            True on success, False on failure
        """
        pkg_var = self._get_pkg_var(pkg)
        root_config = self.depgraph._frozen_config.roots[pkg.root]

        # Process each dependency type (POC: DEPEND and RDEPEND only)
        for dep_type in ("DEPEND", "RDEPEND"):
            dep_str = pkg._metadata.get(dep_type, "")
            if not dep_str:
                continue

            # Parse dependency string with structure preserved
            try:
                dep_struct = use_reduce(
                    dep_str,
                    uselist=[],  # POC: no USE flag evaluation
                    flat=False,  # Keep structure for OR deps
                    token_class=Atom,
                )
            except (InvalidDependString, InvalidAtom) as e:
                writemsg(
                    f"!!! Invalid {dep_type} in {pkg.cpv}: {e}\n",
                    noiselevel=-1
                )
                continue

            # Process the dependency structure recursively
            if not self._process_dep_struct(pkg, dep_struct, dep_type):
                return False

        return True

    def _process_dep_struct(self, parent_pkg: Package, dep_struct, dep_type: str) -> bool:
        """
        Recursively process dependency structure and add Z3 constraints.

        Handles:
            - Simple atoms: find matches and add implication
            - OR dependencies: create disjunction
            - Nested structures: recurse

        Returns:
            True on success, False on failure
        """
        parent_var = self._get_pkg_var(parent_pkg)

        if isinstance(dep_struct, list):
            # Check for OR dependency
            if dep_struct and dep_struct[0] == "||":
                # OR dependency: parent → (dep1 OR dep2 OR ...)
                or_constraints = []
                for alt in dep_struct[1:]:
                    if isinstance(alt, list):
                        # Nested list - this is either another OR, an AND group, or just a list of atoms
                        # In Portage, || ( a b c ) becomes ['||', ['a', 'b', 'c']]
                        # So the inner list is the alternatives, not an AND group
                        for item in alt:
                            if isinstance(item, Atom):
                                constraint = self._process_dep_item(parent_pkg, item, dep_type)
                                if constraint is not None:
                                    or_constraints.append(constraint)
                            elif isinstance(item, list):
                                # Nested structure - recursively process
                                nested_constraint = self._process_dep_struct(parent_pkg, item, dep_type)
                                if nested_constraint:
                                    or_constraints.append(nested_constraint)
                    else:
                        # Single atom
                        constraint = self._process_dep_item(parent_pkg, alt, dep_type)
                        if constraint is not None:
                            or_constraints.append(constraint)

                if or_constraints:
                    dep_constraint = (
                        Or(*or_constraints) if len(or_constraints) > 1
                        else or_constraints[0]
                    )
                    self._solver.add(Implies(parent_var, dep_constraint))
                    self._stats["constraints_added"] += 1
            else:
                # AND group: process each item
                for item in dep_struct:
                    if not self._process_dep_struct(parent_pkg, item, dep_type):
                        return False

        elif isinstance(dep_struct, Atom):
            # Simple atom
            constraint = self._process_dep_item(parent_pkg, dep_struct, dep_type)
            if constraint is not None:
                self._solver.add(Implies(parent_var, constraint))
                self._stats["constraints_added"] += 1

        return True

    def _process_dep_item(self, parent_pkg: Package, atom: Atom, dep_type: str) -> Optional[Any]:
        """
        Process a single dependency atom and return Z3 constraint.

        Recursively discovers matching packages and adds them to the problem.

        Returns:
            Z3 constraint (Or of matching package vars) or None
        """
        if hasattr(atom, "blocker") and atom.blocker:
            # Blockers handled separately in _add_pkg_blockers
            return None

        root_config = self.depgraph._frozen_config.roots[parent_pkg.root]
        self._stats["atoms_processed"] += 1

        # Find all packages matching this atom
        matching_pkgs = self._find_matching_packages(atom, root_config)

        if not matching_pkgs:
            # No matches - dependency cannot be satisfied
            writemsg(
                colorize("WARN", "!!! ") +
                f"No packages match {atom} (required by {parent_pkg.cpv})\n",
                noiselevel=-1,
            )
            # Return None - this will make parent implication unsatisfiable
            # unless parent is not selected
            return None

        # Recursively add each matching package (if not already added)
        for dep_pkg in matching_pkgs:
            if dep_pkg not in self._all_packages:
                dep_obj = Dependency(
                    atom=atom,
                    root=dep_pkg.root,
                    parent=parent_pkg
                )
                if not self.add_pkg(dep_pkg, dep_obj):
                    return None

        # Create disjunction of all matching packages
        matching_vars = [self._get_pkg_var(pkg) for pkg in matching_pkgs]
        return Or(*matching_vars) if len(matching_vars) > 1 else matching_vars[0]

    def _add_update_preferences(self):
        """
        Add optimization preferences for --update mode.
        
        In update mode with --deep, we want to prefer newer versions over installed ones.
        We do this by adding constraints that prefer selecting ebuild packages over 
        installed packages when both exist for the same CPV.
        
        For each (cpv, slot) that has both installed and ebuild versions:
        - If ebuild version is newer, strongly prefer it
        - This encourages upgrades while still allowing installed if dependencies require it
        """
        from portage.versions import vercmp
        
        deep = "--deep" in self.depgraph._frozen_config.myopts
        
        # Strategy 1: Prefer ebuild over installed for same CPV
        cpv_packages = {}
        for pkg in self._all_packages:
            if pkg.cpv not in cpv_packages:
                cpv_packages[pkg.cpv] = []
            cpv_packages[pkg.cpv].append(pkg)
        
        for cpv, pkgs in cpv_packages.items():
            installed_pkg = None
            ebuild_pkg = None
            
            for pkg in pkgs:
                pkg_is_installed = pkg.type_name == "installed" if hasattr(pkg, 'type_name') else pkg.installed
                if pkg_is_installed:
                    installed_pkg = pkg
                else:
                    ebuild_pkg = pkg
            
            if installed_pkg and ebuild_pkg and deep:
                ebuild_var = self._get_pkg_var(ebuild_pkg)
                installed_var = self._get_pkg_var(installed_pkg)
                # Prefer ebuild over installed with high weight to ensure determinism
                self._solver.add_soft(ebuild_var, weight=10)
                self._stats["constraints_added"] += 1
        
        # Strategy 2: Prefer newer versions over older ones (for same package in different slots)
        if deep:
            # Group by category/package to find version alternatives
            cat_pkg_versions = {}
            for pkg in self._all_packages:
                cp = cpv_getkey(pkg.cpv)
                if cp not in cat_pkg_versions:
                    cat_pkg_versions[cp] = []
                cat_pkg_versions[cp].append(pkg)
            
            # For each package with multiple versions
            for cp, pkgs in cat_pkg_versions.items():
                if len(pkgs) <= 1:
                    continue
                
                # Sort by version (newest first)
                from portage.versions import pkgsplit
                try:
                    sorted_pkgs = sorted(pkgs, key=lambda p: pkgsplit(p.cpv)[1], reverse=True)
                    
                    # Prefer newer versions with higher weight
                    for i, pkg in enumerate(sorted_pkgs):
                        # Weight decreases with age: newest=len(pkgs), oldest=1
                        weight = len(sorted_pkgs) - i
                        pkg_var = self._get_pkg_var(pkg)
                        self._solver.add_soft(pkg_var, weight=weight)
                        self._stats["constraints_added"] += 1
                except Exception:
                    # Skip if version parsing fails
                    pass

    def _add_pkg_blockers(self, pkg: Package) -> bool:
        """
        Add blocker constraints for this package.

        For each blocker in pkg's dependencies, add constraint:
            pkg → NOT(blocked_pkg)

        This is called after dependencies are processed, so we check against
        all discovered packages.

        Returns:
            True on success, False on failure
        """
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
                        # Remove blocker prefix to get actual atom
                        blocker_atom = Atom(str(item).lstrip("!"))

                        # Find all packages matching this blocker
                        for blocked_pkg in self._all_packages:
                            if blocker_atom.match(blocked_pkg):
                                blocked_var = self._get_pkg_var(blocked_pkg)
                                # pkg → NOT(blocked_pkg)
                                self._solver.add(Implies(pkg_var, Not(blocked_var)))
                                self._stats["constraints_added"] += 1

            except (InvalidDependString, InvalidAtom):
                pass

        return True

    def _find_matching_packages(self, atom: Atom, root_config) -> List[Package]:
        """
        Find all packages that match the given atom.

        Searches installed packages, ebuild tree, and binary packages.
        This is crucial for --update to find both current and available versions.

        Returns:
            List of matching Package instances (deduplicated)
        """
        matches = []
        seen_keys = set()  # Track (cpv, type) to avoid duplicates

        # Search installed packages first (pre-loaded in _load_installed_packages)
        try:
            for pkg in self.depgraph._iter_match_pkgs(root_config, "installed", atom):
                pkg = self._get_or_cache_pkg(pkg)
                type_name = "installed" if pkg.installed else "ebuild"
                key = (pkg.cpv, type_name)
                if key not in seen_keys:
                    matches.append(pkg)
                    seen_keys.add(key)
        except Exception:
            pass

        # Search in portage tree (available ebuilds)
        try:
            for pkg in self.depgraph._iter_match_pkgs(root_config, "ebuild", atom):
                pkg = self._get_or_cache_pkg(pkg)
                type_name = "installed" if pkg.installed else "ebuild"
                key = (pkg.cpv, type_name)
                if key not in seen_keys:
                    matches.append(pkg)
                    seen_keys.add(key)
        except Exception:
            pass

        # Also check binary packages
        try:
            for pkg in self.depgraph._iter_match_pkgs(root_config, "binary", atom):
                pkg = self._get_or_cache_pkg(pkg)
                type_name = "installed" if pkg.installed else "ebuild"
                key = (pkg.cpv, type_name)
                if key not in seen_keys:
                    matches.append(pkg)
                    seen_keys.add(key)
        except Exception:
            pass

        return matches

    def _get_or_cache_pkg(self, pkg: Package) -> Package:
        """
        Get or cache a package to ensure uniqueness.
        
        Returns the canonical Package instance for this (cpv, type).
        This prevents duplicate Package objects from being added to the problem.
        
        Args:
            pkg: Package instance to deduplicate
            
        Returns:
            Canonical Package instance
        """
        # Use type_name attribute (which is reliable) instead of installed property
        type_name = pkg.type_name if hasattr(pkg, 'type_name') else ("installed" if pkg.installed else "ebuild")
        cache_key = (pkg.cpv, type_name)
        
        if cache_key not in self._pkg_cache:
            self._pkg_cache[cache_key] = pkg
        
        return self._pkg_cache[cache_key]

    def _get_pkg_var(self, pkg: Package) -> Bool:
        """
        Get or create Z3 Boolean variable for a package.

        Variable naming: p_{cpv_sanitized}_{slot_sanitized}_{type}
        The type suffix (inst/ebuild) distinguishes installed vs available packages.

        Returns:
            Z3 Bool variable
        """
        if pkg not in self._pkg_vars:
            # Create unique variable name including package type
            type_suffix = "inst" if pkg.installed else "ebuild"
            var_name = f"p_{pkg.cpv.replace('/', '_')}_{pkg.slot.replace('/', '_')}_{type_suffix}"
            self._pkg_vars[pkg] = Bool(var_name)
        return self._pkg_vars[pkg]

    def _extract_solution(self) -> List[Package]:
        """
        Extract the solution from Z3 model.

        Returns list of packages that should be installed (excluding already
        installed packages).

        Returns:
            List of Package instances to install
        """
        from z3 import is_true
        
        model = self._solver.model()
        new_packages = []
        selected_cpvs = set()  # Track which CPVs we've selected

        for pkg, var in self._pkg_vars.items():
            # Check if this package is True in the solution
            # Use is_true() to properly check Z3 boolean values
            if is_true(model.evaluate(var)):
                pkg_is_installed = pkg.type_name == "installed" if hasattr(pkg, 'type_name') else pkg.installed
                # Only include non-installed packages
                if not pkg_is_installed:
                    # Skip if we already have this CPV (prefer first encountered, which should be ebuild)
                    if pkg.cpv not in selected_cpvs:
                        # Check if an installed version of this CPV exists
                        already_installed = any(
                            inst_pkg.cpv == pkg.cpv and (inst_pkg.type_name == "installed" if hasattr(inst_pkg, 'type_name') else inst_pkg.installed)
                            for inst_pkg in self._installed_packages
                        )
                        if not already_installed:
                            new_packages.append(pkg)
                            selected_cpvs.add(pkg.cpv)

        return new_packages

    def _populate_depgraph(self, packages: List[Package]) -> bool:
        """
        Populate the depgraph with the solution packages.

        This adds the packages to the depgraph's internal structures so that
        the native Phase 3 (conflict resolution and serialization) can work.

        Returns:
            True on success, False on failure
        """
        try:
            # Add each package to the depgraph
            for pkg in packages:
                # Add to the graph (native _add_pkg)
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
