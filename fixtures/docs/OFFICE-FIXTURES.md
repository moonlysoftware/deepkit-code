# Office fixtures

These original fixtures contain only invented repository documentation. They are
covered by the repository's MIT OR Apache-2.0 license. None exceeds 40 KB.

- `handbook.docx`: two headings, paragraphs and a task/state table.
- `inventory.xlsx`: Documents and Operations sheets with small tables.
- `workflow.pptx`: Indexing and Searching slides with text.
- `encrypted.docx`: an encrypted copy of handbook, using the public test password
  `fixture-only`. It is a parser-rejection fixture, not a stored credential.

Regenerate the three plaintext files with Python 3 and the pinned tools in
`notes/runs/t3-2-fixture-tools-requirements.txt` installed in an isolated venv:

```sh
python scripts/corpus/make-office-fixtures.py --output /tmp/office-fixtures
```

The generator normalizes ZIP entry metadata, order and core timestamps. Two runs
with those tools reproduced the committed plaintext files byte for byte. Add
`--encrypted` with `msoffcrypto-tool==5.4.2` installed to regenerate the encrypted
case; encryption is randomized, so ciphertext hashes will differ. The generator
verifies password, integrity and an exact plaintext round trip before writing it.

The `.golden.json` files are reviewed extraction expectations. They retain
Markdown formatting, tables, local text positions and sheet/slide locators.
The generator does not update goldens automatically. Tests require no Python,
network, credentials or external Office application.
