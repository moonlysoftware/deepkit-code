# Notebook fixture

`walkthrough.ipynb` is an original, hand-authored nbformat4.5 notebook under the
repository's MIT OR Apache-2.0 license. It contains two Markdown cells, a small
Python function, an empty code cell and a raw cell. The output and attachment
sentinels are invented strings that must never appear in extracted units.

The reviewed `.golden.json` records the three admitted units, their cell headers,
AST function scope and local text positions. Public tests also serialize the same
notebook on one JSON line to verify that navigation does not confuse container
lines with extracted cell lines. Large2,000-cell notebooks are generated in tests;
no external Jupyter application, Python dependency or executed code is required.
