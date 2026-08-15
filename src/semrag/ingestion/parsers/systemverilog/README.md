# SystemVerilog Parser — Known Gaps (tracked for next iteration)

- **Tree-sitter fallback not implemented.** The registry (`interfaces.py`) supports
  an ordered fallback chain per `FileCategory`; only the pyslang syntax-tree parser
  is currently registered. Files that pyslang cannot parse at all (vs. files it
  parses with recoverable diagnostics, which already work today) currently fail
  with no result rather than falling back to a more permissive parser.
- **Non-ANSI port lists** (ports declared in the module body rather than the port
  list itself) are not extracted — `_extract_ports` returns an empty list for these,
  it does not error.
- **VHDL is not covered by this package.** RTL_VHDL has its own parser —
  see `../vhdl/parser.py` (regex-structural, same trade-off class as TCL).
- **No elaboration-level facts** (resolved types, whether an instantiated module
  actually exists) — see the design-decision docstring in `parser.py` for why this
  is intentional, not an oversight.
