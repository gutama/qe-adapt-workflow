"""Tests for the QE pw.x input parser and writer."""

from __future__ import annotations

import unittest
from pathlib import Path

from qeanalyzer.io.pw_input import (
    AtomicPosition,
    AtomicSpecies,
    CellParameters,
    KPoints,
    PWInput,
    parse_pw_input,
    read_pw_input,
    write_pw_input,
    _parse_value,
    _strip_comment,
)

FIXTURES = Path(__file__).parent / "fixtures"


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

class TestValueParsing(unittest.TestCase):
    """Fortran namelist value conversion."""

    def test_boolean_true_variants(self):
        for token in (".true.", ".TRUE.", ".True.", ".t.", ".T."):
            self.assertIs(_parse_value(token), True, msg=token)

    def test_boolean_false_variants(self):
        for token in (".false.", ".FALSE.", ".False.", ".f.", ".F."):
            self.assertIs(_parse_value(token), False, msg=token)

    def test_integer(self):
        self.assertEqual(_parse_value("42"), 42)
        self.assertIsInstance(_parse_value("42"), int)

    def test_negative_integer(self):
        self.assertEqual(_parse_value("-5"), -5)
        self.assertIsInstance(_parse_value("-5"), int)

    def test_float(self):
        self.assertAlmostEqual(_parse_value("3.14"), 3.14)
        self.assertIsInstance(_parse_value("3.14"), float)

    def test_negative_float(self):
        self.assertAlmostEqual(_parse_value("-3.14"), -3.14)

    def test_fortran_double_lowercase(self):
        self.assertAlmostEqual(_parse_value("1.0d-8"), 1.0e-8)

    def test_fortran_double_uppercase(self):
        self.assertAlmostEqual(_parse_value("2.5D+3"), 2.5e3)

    def test_fortran_double_zero_exponent(self):
        self.assertAlmostEqual(_parse_value("1.0d0"), 1.0)

    def test_single_quoted_string(self):
        self.assertEqual(_parse_value("'scf'"), "scf")

    def test_single_quoted_path(self):
        self.assertEqual(_parse_value("'./tmp/'"), "./tmp/")

    def test_double_quoted_string(self):
        self.assertEqual(_parse_value('"nscf"'), "nscf")


class TestCommentStripping(unittest.TestCase):
    """Inline comment removal."""

    def test_no_comment(self):
        self.assertEqual(
            _strip_comment("ecutwfc = 30.0,"),
            "ecutwfc = 30.0,",
        )

    def test_trailing_comment(self):
        result = _strip_comment("ecutwfc = 30.0,  ! kinetic energy cutoff")
        self.assertEqual(result, "ecutwfc = 30.0,  ")

    def test_full_line_comment(self):
        self.assertEqual(_strip_comment("! this is a comment"), "")

    def test_bang_inside_quotes(self):
        line = "title = 'warning! test'"
        self.assertEqual(_strip_comment(line), line)


# ---------------------------------------------------------------------------
# Fixture-based parser tests
# ---------------------------------------------------------------------------

class TestParseSiSCF(unittest.TestCase):
    """Parse the silicon SCF fixture."""

    @classmethod
    def setUpClass(cls):
        cls.pw = read_pw_input(FIXTURES / "si_scf.in")

    def test_namelists_present(self):
        for name in ("CONTROL", "SYSTEM", "ELECTRONS"):
            self.assertIn(name, self.pw.namelists)

    def test_calculation_type(self):
        self.assertEqual(self.pw.calculation, "scf")

    def test_prefix(self):
        self.assertEqual(self.pw.namelists["CONTROL"]["prefix"], "silicon")

    def test_ecutwfc(self):
        self.assertAlmostEqual(self.pw.ecutwfc, 30.0)

    def test_ecutrho(self):
        self.assertAlmostEqual(self.pw.ecutrho, 240.0)

    def test_nat(self):
        self.assertEqual(self.pw.nat, 2)

    def test_ntyp(self):
        self.assertEqual(self.pw.ntyp, 1)

    def test_ibrav(self):
        self.assertEqual(self.pw.namelists["SYSTEM"]["ibrav"], 2)

    def test_celldm(self):
        self.assertAlmostEqual(
            self.pw.namelists["SYSTEM"]["celldm(1)"], 10.2,
        )

    def test_conv_thr_fortran_double(self):
        self.assertAlmostEqual(
            self.pw.namelists["ELECTRONS"]["conv_thr"], 1.0e-8,
        )

    def test_mixing_beta(self):
        self.assertAlmostEqual(
            self.pw.namelists["ELECTRONS"]["mixing_beta"], 0.7,
        )

    def test_atomic_species_count(self):
        self.assertEqual(len(self.pw.atomic_species), 1)

    def test_atomic_species_values(self):
        sp = self.pw.atomic_species[0]
        self.assertEqual(sp.symbol, "Si")
        self.assertAlmostEqual(sp.mass, 28.086)
        self.assertIn("Si", sp.pseudo_file)

    def test_atomic_positions_option(self):
        self.assertEqual(self.pw.atomic_positions_option, "crystal")

    def test_atomic_positions_count(self):
        self.assertEqual(len(self.pw.atomic_positions), 2)

    def test_atomic_positions_values(self):
        p0 = self.pw.atomic_positions[0]
        self.assertEqual(p0.symbol, "Si")
        self.assertAlmostEqual(p0.position[0], 0.0)
        p1 = self.pw.atomic_positions[1]
        self.assertAlmostEqual(p1.position[0], 0.25)
        self.assertAlmostEqual(p1.position[1], 0.25)

    def test_kpoints_automatic(self):
        self.assertIsNotNone(self.pw.kpoints)
        self.assertEqual(self.pw.kpoints.option, "automatic")
        self.assertEqual(self.pw.kpoints.grid, (4, 4, 4))
        self.assertEqual(self.pw.kpoints.offsets, (0, 0, 0))

    def test_no_cell_parameters(self):
        self.assertIsNone(self.pw.cell_parameters)


class TestParseVCRelax(unittest.TestCase):
    """Parse the vc-relax fixture with CELL_PARAMETERS."""

    @classmethod
    def setUpClass(cls):
        cls.pw = read_pw_input(FIXTURES / "si_vc_relax.in")

    def test_calculation(self):
        self.assertEqual(self.pw.calculation, "vc-relax")

    def test_ions_namelist_present(self):
        self.assertIn("IONS", self.pw.namelists)

    def test_ions_namelist_empty(self):
        self.assertEqual(self.pw.namelists["IONS"], {})

    def test_cell_namelist_present(self):
        self.assertIn("CELL", self.pw.namelists)

    def test_cell_dofree(self):
        self.assertEqual(
            self.pw.namelists["CELL"]["cell_dofree"], "all",
        )

    def test_ibrav_zero(self):
        self.assertEqual(self.pw.namelists["SYSTEM"]["ibrav"], 0)

    def test_cell_parameters_present(self):
        self.assertIsNotNone(self.pw.cell_parameters)

    def test_cell_parameters_option(self):
        self.assertEqual(self.pw.cell_parameters.option, "angstrom")

    def test_cell_vectors(self):
        v1, v2, v3 = self.pw.cell_parameters.vectors
        self.assertAlmostEqual(v1[0], 0.0)
        self.assertAlmostEqual(v1[1], 2.715)
        self.assertAlmostEqual(v1[2], 2.715)
        self.assertAlmostEqual(v2[0], 2.715)
        self.assertAlmostEqual(v2[1], 0.0)


class TestParseAlMetal(unittest.TestCase):
    """Parse the aluminum metal fixture with smearing."""

    @classmethod
    def setUpClass(cls):
        cls.pw = read_pw_input(FIXTURES / "al_scf_metal.in")

    def test_occupations(self):
        self.assertEqual(
            self.pw.namelists["SYSTEM"]["occupations"], "smearing",
        )

    def test_smearing(self):
        self.assertEqual(self.pw.namelists["SYSTEM"]["smearing"], "mp")

    def test_degauss(self):
        self.assertAlmostEqual(
            self.pw.namelists["SYSTEM"]["degauss"], 0.02,
        )

    def test_single_atom(self):
        self.assertEqual(self.pw.nat, 1)
        self.assertEqual(len(self.pw.atomic_positions), 1)

    def test_denser_kpoints(self):
        self.assertEqual(self.pw.kpoints.grid, (8, 8, 8))

    def test_al_species(self):
        sp = self.pw.atomic_species[0]
        self.assertEqual(sp.symbol, "Al")
        self.assertAlmostEqual(sp.mass, 26.982)


class TestParseH2O(unittest.TestCase):
    """Parse the water molecule fixture with gamma k-point."""

    @classmethod
    def setUpClass(cls):
        cls.pw = read_pw_input(FIXTURES / "h2o_scf.in")

    def test_gamma_kpoints(self):
        self.assertIsNotNone(self.pw.kpoints)
        self.assertEqual(self.pw.kpoints.option, "gamma")

    def test_two_species(self):
        self.assertEqual(len(self.pw.atomic_species), 2)
        symbols = {sp.symbol for sp in self.pw.atomic_species}
        self.assertEqual(symbols, {"O", "H"})

    def test_three_atoms(self):
        self.assertEqual(len(self.pw.atomic_positions), 3)
        self.assertEqual(self.pw.nat, 3)
        self.assertEqual(self.pw.ntyp, 2)

    def test_positions_option_angstrom(self):
        self.assertEqual(self.pw.atomic_positions_option, "angstrom")

    def test_negative_coordinate(self):
        # H atom at y = -0.763239
        h_atoms = [
            ap for ap in self.pw.atomic_positions if ap.symbol == "H"
        ]
        y_values = sorted(ap.position[1] for ap in h_atoms)
        self.assertAlmostEqual(y_values[0], -0.763239)

    def test_cell_parameters_cubic_box(self):
        self.assertIsNotNone(self.pw.cell_parameters)
        v1, v2, v3 = self.pw.cell_parameters.vectors
        self.assertAlmostEqual(v1[0], 10.0)
        self.assertAlmostEqual(v2[1], 10.0)
        self.assertAlmostEqual(v3[2], 10.0)
        # Off-diagonals are zero.
        self.assertAlmostEqual(v1[1], 0.0)
        self.assertAlmostEqual(v1[2], 0.0)


# ---------------------------------------------------------------------------
# Round-trip tests
# ---------------------------------------------------------------------------

class TestRoundTrip(unittest.TestCase):
    """parse → write → parse must preserve all semantic content."""

    def _assert_pw_equal(self, a: PWInput, b: PWInput) -> None:
        # Namelists
        self.assertEqual(set(a.namelists), set(b.namelists))
        for name in a.namelists:
            nl_a, nl_b = a.namelists[name], b.namelists[name]
            self.assertEqual(
                set(nl_a), set(nl_b),
                msg=f"keys differ in &{name}",
            )
            for key in nl_a:
                va, vb = nl_a[key], nl_b[key]
                if isinstance(va, float):
                    self.assertAlmostEqual(
                        va, vb, places=8,
                        msg=f"&{name}/{key}",
                    )
                else:
                    self.assertEqual(va, vb, msg=f"&{name}/{key}")

        # Species
        self.assertEqual(len(a.atomic_species), len(b.atomic_species))
        for sa, sb in zip(a.atomic_species, b.atomic_species):
            self.assertEqual(sa.symbol, sb.symbol)
            self.assertAlmostEqual(sa.mass, sb.mass)
            self.assertEqual(sa.pseudo_file, sb.pseudo_file)

        # Positions
        self.assertEqual(a.atomic_positions_option, b.atomic_positions_option)
        self.assertEqual(len(a.atomic_positions), len(b.atomic_positions))
        for pa, pb in zip(a.atomic_positions, b.atomic_positions):
            self.assertEqual(pa.symbol, pb.symbol)
            for j in range(3):
                self.assertAlmostEqual(
                    pa.position[j], pb.position[j], places=8,
                )
            self.assertEqual(pa.if_pos, pb.if_pos)

        # K-points
        if a.kpoints is None:
            self.assertIsNone(b.kpoints)
        else:
            self.assertIsNotNone(b.kpoints)
            self.assertEqual(a.kpoints.option, b.kpoints.option)
            self.assertEqual(a.kpoints.grid, b.kpoints.grid)
            self.assertEqual(a.kpoints.offsets, b.kpoints.offsets)
            self.assertEqual(a.kpoints.nk, b.kpoints.nk)
            if a.kpoints.points is not None:
                self.assertIsNotNone(b.kpoints.points)
                self.assertEqual(
                    len(a.kpoints.points), len(b.kpoints.points),
                )
                for ka, kb in zip(a.kpoints.points, b.kpoints.points):
                    for j in range(4):
                        self.assertAlmostEqual(ka[j], kb[j], places=8)

        # Cell parameters
        if a.cell_parameters is None:
            self.assertIsNone(b.cell_parameters)
        else:
            self.assertIsNotNone(b.cell_parameters)
            self.assertEqual(
                a.cell_parameters.option, b.cell_parameters.option,
            )
            for i in range(3):
                for j in range(3):
                    self.assertAlmostEqual(
                        a.cell_parameters.vectors[i][j],
                        b.cell_parameters.vectors[i][j],
                        places=8,
                    )

    def _roundtrip(self, path: Path) -> None:
        original = read_pw_input(path)
        text = write_pw_input(original)
        reparsed = parse_pw_input(text)
        self._assert_pw_equal(original, reparsed)

    def test_roundtrip_si_scf(self):
        self._roundtrip(FIXTURES / "si_scf.in")

    def test_roundtrip_si_vc_relax(self):
        self._roundtrip(FIXTURES / "si_vc_relax.in")

    def test_roundtrip_al_metal(self):
        self._roundtrip(FIXTURES / "al_scf_metal.in")

    def test_roundtrip_h2o(self):
        self._roundtrip(FIXTURES / "h2o_scf.in")


# ---------------------------------------------------------------------------
# Writer tests
# ---------------------------------------------------------------------------

class TestWriter(unittest.TestCase):
    """Verify writer output contains expected tokens."""

    def test_namelists_present(self):
        text = write_pw_input(read_pw_input(FIXTURES / "si_scf.in"))
        self.assertIn("&CONTROL", text)
        self.assertIn("&SYSTEM", text)
        self.assertIn("&ELECTRONS", text)

    def test_species_card(self):
        text = write_pw_input(read_pw_input(FIXTURES / "si_scf.in"))
        self.assertIn("ATOMIC_SPECIES", text)
        self.assertIn("Si", text)

    def test_positions_card(self):
        text = write_pw_input(read_pw_input(FIXTURES / "si_scf.in"))
        self.assertIn("ATOMIC_POSITIONS", text)
        self.assertIn("crystal", text)

    def test_kpoints_automatic(self):
        text = write_pw_input(read_pw_input(FIXTURES / "si_scf.in"))
        self.assertIn("K_POINTS", text)
        self.assertIn("automatic", text)

    def test_kpoints_gamma(self):
        text = write_pw_input(read_pw_input(FIXTURES / "h2o_scf.in"))
        self.assertIn("K_POINTS {gamma}", text)

    def test_cell_parameters_card(self):
        text = write_pw_input(
            read_pw_input(FIXTURES / "si_vc_relax.in"),
        )
        self.assertIn("CELL_PARAMETERS", text)
        self.assertIn("angstrom", text)

    def test_namelist_terminators(self):
        text = write_pw_input(read_pw_input(FIXTURES / "si_scf.in"))
        # Each namelist must end with a bare '/'.
        self.assertGreaterEqual(text.count("\n/\n"), 2)


# ---------------------------------------------------------------------------
# Edge-case and inline tests
# ---------------------------------------------------------------------------

class TestEdgeCases(unittest.TestCase):
    """Parser edge cases that don't need fixture files."""

    def test_inline_comments(self):
        text = (
            "&CONTROL\n"
            "  calculation = 'scf', ! run type\n"
            "  prefix = 'test',\n"
            "/\n"
            "&SYSTEM\n"
            "  ibrav = 2,  ! FCC\n"
            "  celldm(1) = 10.2,\n"
            "  nat = 1,\n"
            "  ntyp = 1,\n"
            "  ecutwfc = 30.0,\n"
            "/\n"
            "&ELECTRONS\n"
            "/\n"
            "ATOMIC_SPECIES\n"
            "  Si  28.086  Si.UPF\n"
            "ATOMIC_POSITIONS crystal\n"
            "  Si  0.0  0.0  0.0\n"
            "K_POINTS gamma\n"
        )
        pw = parse_pw_input(text)
        self.assertEqual(pw.calculation, "scf")
        self.assertEqual(pw.namelists["SYSTEM"]["ibrav"], 2)

    def test_minimal_input(self):
        text = (
            "&CONTROL\n"
            "/\n"
            "&SYSTEM\n"
            "  nat = 1, ntyp = 1, ecutwfc = 20.0,\n"
            "/\n"
            "&ELECTRONS\n"
            "/\n"
            "ATOMIC_SPECIES\n"
            "  H  1.008  H.UPF\n"
            "ATOMIC_POSITIONS angstrom\n"
            "  H  0.0  0.0  0.0\n"
            "K_POINTS gamma\n"
        )
        pw = parse_pw_input(text)
        self.assertEqual(pw.nat, 1)
        self.assertEqual(len(pw.atomic_species), 1)
        self.assertEqual(pw.kpoints.option, "gamma")

    def test_empty_ions_cell_namelists(self):
        text = (
            "&CONTROL\n"
            "  calculation = 'vc-relax',\n"
            "/\n"
            "&SYSTEM\n"
            "  ibrav = 0, nat = 1, ntyp = 1, ecutwfc = 30.0,\n"
            "/\n"
            "&ELECTRONS\n"
            "/\n"
            "&IONS\n"
            "/\n"
            "&CELL\n"
            "/\n"
            "CELL_PARAMETERS angstrom\n"
            "  5.0  0.0  0.0\n"
            "  0.0  5.0  0.0\n"
            "  0.0  0.0  5.0\n"
            "ATOMIC_SPECIES\n"
            "  Si  28.086  Si.UPF\n"
            "ATOMIC_POSITIONS crystal\n"
            "  Si  0.0  0.0  0.0\n"
            "K_POINTS gamma\n"
        )
        pw = parse_pw_input(text)
        self.assertIn("IONS", pw.namelists)
        self.assertIn("CELL", pw.namelists)
        self.assertEqual(pw.namelists["IONS"], {})
        self.assertEqual(pw.namelists["CELL"], {})

    def test_crystal_kpoints(self):
        text = (
            "&CONTROL\n/\n"
            "&SYSTEM\n"
            "  nat = 1, ntyp = 1, ecutwfc = 30.0,\n"
            "/\n"
            "&ELECTRONS\n/\n"
            "ATOMIC_SPECIES\n"
            "  Si  28.086  Si.UPF\n"
            "ATOMIC_POSITIONS crystal\n"
            "  Si  0.0  0.0  0.0\n"
            "K_POINTS crystal\n"
            "3\n"
            "  0.0  0.0  0.0  0.25\n"
            "  0.5  0.0  0.0  0.50\n"
            "  0.5  0.5  0.0  0.25\n"
        )
        pw = parse_pw_input(text)
        self.assertEqual(pw.kpoints.option, "crystal")
        self.assertEqual(pw.kpoints.nk, 3)
        self.assertEqual(len(pw.kpoints.points), 3)
        self.assertAlmostEqual(pw.kpoints.points[1][0], 0.5)
        self.assertAlmostEqual(pw.kpoints.points[1][3], 0.50)

    def test_card_option_braces(self):
        """Options specified with {braces} are parsed the same as bare."""
        text_bare = (
            "&CONTROL\n/\n&SYSTEM\n"
            "  nat = 1, ntyp = 1, ecutwfc = 20.0,\n/\n"
            "&ELECTRONS\n/\n"
            "ATOMIC_SPECIES\n  H 1.008 H.UPF\n"
            "ATOMIC_POSITIONS crystal\n  H 0.0 0.0 0.0\n"
            "K_POINTS automatic\n  2 2 2 0 0 0\n"
        )
        text_brace = text_bare.replace(
            "ATOMIC_POSITIONS crystal",
            "ATOMIC_POSITIONS {crystal}",
        ).replace(
            "K_POINTS automatic",
            "K_POINTS {automatic}",
        )
        pw_bare = parse_pw_input(text_bare)
        pw_brace = parse_pw_input(text_brace)
        self.assertEqual(
            pw_bare.atomic_positions_option,
            pw_brace.atomic_positions_option,
        )
        self.assertEqual(pw_bare.kpoints.option, pw_brace.kpoints.option)

    def test_if_pos_constraints(self):
        text = (
            "&CONTROL\n/\n&SYSTEM\n"
            "  nat = 2, ntyp = 1, ecutwfc = 20.0,\n/\n"
            "&ELECTRONS\n/\n"
            "ATOMIC_SPECIES\n  Si 28.086 Si.UPF\n"
            "ATOMIC_POSITIONS crystal\n"
            "  Si  0.0  0.0  0.0  0  0  0\n"
            "  Si  0.25 0.25 0.25 1  1  1\n"
            "K_POINTS gamma\n"
        )
        pw = parse_pw_input(text)
        self.assertEqual(pw.atomic_positions[0].if_pos, (0, 0, 0))
        self.assertEqual(pw.atomic_positions[1].if_pos, (1, 1, 1))

    def test_multiple_entries_one_line(self):
        """Multiple key=value pairs on a single comma-separated line."""
        text = (
            "&CONTROL\n/\n"
            "&SYSTEM\n"
            "  nat = 2, ntyp = 1, ecutwfc = 30.0, ecutrho = 240.0\n"
            "/\n"
            "&ELECTRONS\n/\n"
            "ATOMIC_SPECIES\n  Si 28.086 Si.UPF\n"
            "ATOMIC_POSITIONS crystal\n"
            "  Si 0.0 0.0 0.0\n  Si 0.25 0.25 0.25\n"
            "K_POINTS gamma\n"
        )
        pw = parse_pw_input(text)
        self.assertEqual(pw.nat, 2)
        self.assertEqual(pw.ntyp, 1)
        self.assertAlmostEqual(pw.ecutwfc, 30.0)
        self.assertAlmostEqual(pw.ecutrho, 240.0)


# ---------------------------------------------------------------------------
# Modification tests
# ---------------------------------------------------------------------------

class TestModification(unittest.TestCase):
    """Modify a parsed input → write → re-parse."""

    def test_change_calculation(self):
        pw = read_pw_input(FIXTURES / "si_scf.in")
        pw.namelists["CONTROL"]["calculation"] = "nscf"
        reparsed = parse_pw_input(write_pw_input(pw))
        self.assertEqual(reparsed.calculation, "nscf")

    def test_add_nbnd(self):
        pw = read_pw_input(FIXTURES / "si_scf.in")
        pw.namelists["SYSTEM"]["nbnd"] = 20
        reparsed = parse_pw_input(write_pw_input(pw))
        self.assertEqual(reparsed.namelists["SYSTEM"]["nbnd"], 20)

    def test_change_kpoints_grid(self):
        pw = read_pw_input(FIXTURES / "si_scf.in")
        pw.kpoints = KPoints(
            option="automatic",
            grid=(8, 8, 8),
            offsets=(0, 0, 0),
        )
        reparsed = parse_pw_input(write_pw_input(pw))
        self.assertEqual(reparsed.kpoints.grid, (8, 8, 8))

    def test_add_smearing(self):
        pw = read_pw_input(FIXTURES / "si_scf.in")
        pw.namelists["SYSTEM"]["occupations"] = "smearing"
        pw.namelists["SYSTEM"]["smearing"] = "mp"
        pw.namelists["SYSTEM"]["degauss"] = 0.02
        reparsed = parse_pw_input(write_pw_input(pw))
        self.assertEqual(
            reparsed.namelists["SYSTEM"]["occupations"], "smearing",
        )
        self.assertAlmostEqual(
            reparsed.namelists["SYSTEM"]["degauss"], 0.02,
        )

    def test_replace_species(self):
        pw = read_pw_input(FIXTURES / "si_scf.in")
        pw.atomic_species = [
            AtomicSpecies("Ge", 72.63, "Ge.UPF"),
        ]
        pw.atomic_positions = [
            AtomicPosition("Ge", (0.0, 0.0, 0.0)),
            AtomicPosition("Ge", (0.25, 0.25, 0.25)),
        ]
        reparsed = parse_pw_input(write_pw_input(pw))
        self.assertEqual(reparsed.atomic_species[0].symbol, "Ge")
        self.assertEqual(reparsed.atomic_positions[0].symbol, "Ge")

class TestDoubleQuotedValues(unittest.TestCase):
    """Fortran accepts " as well as ' as a string delimiter, and QE emits both.

    A '/' inside a double-quoted value previously terminated the namelist early,
    silently dropping every key after it.
    """

    TEXT = (
        "&CONTROL\n"
        '  calculation = "scf"\n'
        '  outdir = "./tmp"\n'
        '  pseudo_dir = "/opt/qe/pseudo"\n'
        '  prefix = "si"\n'
        "/\n"
        "&SYSTEM\n"
        "  ibrav = 2\n"
        "  nat = 2\n"
        "  ntyp = 1\n"
        "  ecutwfc = 30.0\n"
        "/\n"
        "&ELECTRONS\n"
        "  conv_thr = 1.0d-8\n"
        "/\n"
    )

    def setUp(self):
        self.inp = parse_pw_input(self.TEXT)

    def test_value_containing_slash_is_not_truncated(self):
        self.assertEqual(self.inp.get_param("CONTROL", "outdir"), "./tmp")
        self.assertEqual(self.inp.get_param("CONTROL", "pseudo_dir"), "/opt/qe/pseudo")

    def test_keys_after_a_slash_bearing_value_survive(self):
        self.assertEqual(self.inp.get_param("CONTROL", "prefix"), "si")
        self.assertEqual(self.inp.get_param("CONTROL", "calculation"), "scf")

    def test_later_namelists_still_parse(self):
        self.assertEqual(self.inp.get_param("SYSTEM", "ecutwfc"), 30.0)
        self.assertEqual(self.inp.get_param("ELECTRONS", "conv_thr"), 1.0e-8)

    def test_roundtrip_preserves_paths(self):
        reparsed = parse_pw_input(write_pw_input(self.inp))
        self.assertEqual(reparsed.get_param("CONTROL", "outdir"), "./tmp")
        self.assertEqual(reparsed.get_param("CONTROL", "prefix"), "si")

    def test_comment_stripping_respects_double_quotes(self):
        inp = parse_pw_input('&CONTROL\n  outdir = "./a!b"  ! trailing comment\n/\n')
        self.assertEqual(inp.get_param("CONTROL", "outdir"), "./a!b")

    def test_single_quoted_values_still_work(self):
        inp = parse_pw_input("&CONTROL\n  outdir = './tmp'\n  prefix = 'si'\n/\n")
        self.assertEqual(inp.get_param("CONTROL", "outdir"), "./tmp")
        self.assertEqual(inp.get_param("CONTROL", "prefix"), "si")

    def test_apostrophe_inside_double_quoted_value(self):
        inp = parse_pw_input("&CONTROL\n  title = \"it's fine\"\n  prefix = 'si'\n/\n")
        self.assertEqual(inp.get_param("CONTROL", "title"), "it's fine")
        self.assertEqual(inp.get_param("CONTROL", "prefix"), "si")

class TestUnmodelledNamelistsAndCardsSurviveRoundTrip(unittest.TestCase):
    """Every policy deepcopies a PWInput and re-emits it via write_pw_input.

    Anything the writer drops is silently removed from the next calculation, so
    cards and namelists without a structured representation must round-trip.
    """

    TEXT = (
        "&CONTROL\n  calculation = 'relax'\n  prefix = 'si'\n/\n"
        "&SYSTEM\n  ibrav = 0\n  nat = 2\n  ntyp = 1\n  ecutwfc = 30.0\n/\n"
        "&ELECTRONS\n  conv_thr = 1.0d-8\n/\n"
        "&IONS\n  ion_dynamics = 'bfgs'\n/\n"
        "&FCP\n  fcp_mu = 0.5\n/\n"
        "ATOMIC_SPECIES\n Si 28.086 Si.upf\n"
        "ATOMIC_POSITIONS crystal\n Si 0.0 0.0 0.0\n Si 0.25 0.25 0.25\n"
        "K_POINTS automatic\n 4 4 4 0 0 0\n"
        "CONSTRAINTS\n 1\n 'distance' 1 2 2.35\n"
        "ATOMIC_FORCES\n Si 0.0 0.0 0.01\n Si 0.0 0.0 -0.01\n"
        "HUBBARD (ortho-atomic)\n U Si-3d 4.5\n"
    )

    def setUp(self):
        self.inp = parse_pw_input(self.TEXT)
        self.written = write_pw_input(self.inp)
        self.reparsed = parse_pw_input(self.written)

    def test_unmodelled_namelist_is_parsed(self):
        self.assertEqual(self.inp.get_param("FCP", "fcp_mu"), 0.5)

    def test_unmodelled_namelist_is_written(self):
        self.assertIn("&FCP", self.written)
        self.assertEqual(self.reparsed.get_param("FCP", "fcp_mu"), 0.5)

    def test_constraints_card_survives(self):
        self.assertIn("CONSTRAINTS", self.written)
        names = [c.name for c in self.reparsed.extra_cards]
        self.assertIn("CONSTRAINTS", names)
        card = next(c for c in self.reparsed.extra_cards if c.name == "CONSTRAINTS")
        self.assertIn("'distance' 1 2 2.35", card.lines)

    def test_atomic_forces_card_survives(self):
        self.assertIn("ATOMIC_FORCES", self.written)
        card = next(c for c in self.reparsed.extra_cards if c.name == "ATOMIC_FORCES")
        self.assertEqual(len(card.lines), 2)

    def test_hubbard_card_survives_with_option(self):
        """The QE >= 7.1 HUBBARD card must not be dropped or lose its projector."""
        card = next(c for c in self.reparsed.extra_cards if c.name == "HUBBARD")
        self.assertEqual(card.option, "ortho-atomic")
        self.assertIn("U Si-3d 4.5", card.lines)

    def test_structured_content_still_round_trips(self):
        self.assertEqual(self.reparsed.get_param("CONTROL", "calculation"), "relax")
        self.assertEqual(len(self.reparsed.atomic_positions), 2)
        self.assertEqual(self.reparsed.kpoints.grid, (4, 4, 4))

    def test_second_round_trip_is_stable(self):
        self.assertEqual(write_pw_input(self.reparsed), self.written)

    def test_policy_style_deepcopy_preserves_cards(self):
        """This is the path every workflow policy takes."""
        from copy import deepcopy

        clone = deepcopy(self.inp)
        clone.set_param("ELECTRONS", "conv_thr", 1.0e-9)
        out = parse_pw_input(write_pw_input(clone))
        self.assertEqual(out.get_param("ELECTRONS", "conv_thr"), 1.0e-9)
        self.assertIn("CONSTRAINTS", [c.name for c in out.extra_cards])
        self.assertIn("HUBBARD", [c.name for c in out.extra_cards])


if __name__ == "__main__":
    unittest.main()
