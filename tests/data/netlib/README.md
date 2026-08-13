# Netlib LP test data

`adlittle.mps` is the Netlib ADLITTLE linear-programming benchmark expanded
from Netlib's compact MPS representation with Netlib's `emps.c` utility.

- Source: https://www.netlib.org/lp/data/adlittle
- Expander: https://www.netlib.org/lp/data/emps.c
- Published dimensions: 57 MPS rows, 97 columns, 465 nonzeros
- Published optimal objective: `2.2549496316E+05`
- Expanded-file SHA-256:
  `fec81e24fa91bc545d97239b108b43e6034f37b4bf2455a3f8c179726b44d44c`

The objective row is included in Netlib's published row/nonzero counts. The
mlxPDLP parser exposes 56 constraints and excludes objective coefficients from
the constraint-matrix nonzero count.
