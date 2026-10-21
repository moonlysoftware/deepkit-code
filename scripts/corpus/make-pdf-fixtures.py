#!/usr/bin/env python3
"""Generate original, deterministic PDF acceptance fixtures without dependencies."""
import argparse
from pathlib import Path


def pdf(pages, title=None):
    """Build a basic PDF with two fonts and valid indirect objects/xref offsets."""
    objects = [b"<< /Type /Catalog /Pages 2 0 R >>", b"",
               b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
               b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>"]
    kids = []
    for blocks in pages:
        page_id = len(objects) + 1
        kids.append(f"{page_id} 0 R")
        image = f" /XObject << /Scan {page_id + 2} 0 R >>" if not blocks else ""
        objects.append((f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
                        f"/Resources << /Font << /F1 3 0 R /F2 4 0 R >>{image} >> "
                        f"/Contents {page_id + 1} 0 R >>").encode())
        lines = []
        for x, y, size, bold, text in blocks:
            escaped = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
            lines.append(f"BT /F{2 if bold else 1} {size} Tf {x} {y} Td ({escaped}) Tj ET")
        stream = "\n".join(lines).encode("ascii") if blocks else b"q 512 0 0 692 50 50 cm /Scan Do Q"
        objects.append(f"<< /Length {len(stream)} >>\nstream\n".encode() + stream + b"\nendstream")
        if not blocks:
            objects.append(b"<< /Type /XObject /Subtype /Image /Width 1 /Height 1 "
                           b"/ColorSpace /DeviceRGB /BitsPerComponent 8 /Length 3 >>\n"
                           b"stream\n\x88\x88\x88\nendstream")
    objects[1] = f"<< /Type /Pages /Count {len(pages)} /Kids [{' '.join(kids)}] >>".encode()
    info = ""
    if title:
        title_hex = (b"\xfe\xff" + title.encode("utf-16-be")).hex()
        objects.append(f"<< /Title <{title_hex}> >>".encode())
        info = f" /Info {len(objects)} 0 R"
    out = bytearray(b"%PDF-1.7\n%\xe2\xe3\xcf\xd3\n")
    offsets = [0]
    for number, obj in enumerate(objects, 1):
        offsets.append(len(out))
        out.extend(f"{number} 0 obj\n".encode() + obj + b"\nendobj\n")
    xref = len(out)
    out.extend(f"xref\n0 {len(offsets)}\n0000000000 65535 f \n".encode())
    for offset in offsets[1:]:
        out.extend(f"{offset:010d} 00000 n \n".encode())
    out.extend((f"trailer\n<< /Size {len(offsets)} /Root 1 0 R{info} >>\n"
                f"startxref\n{xref}\n%%EOF\n").encode())
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("fixtures/docs"))
    parser.add_argument("--pages", type=int, help="Generate only a numbered manual of this size")
    parser.add_argument("--probe-chars", type=int, help="Use this many text characters per probe page")
    parser.add_argument("--text-pages", type=int, default=0, help="Number of text-bearing probe pages")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.pages is not None:
        pages = [[(50, 740, 18, True, f"Chapter {i + 1}"),
                  (50, 710, 11, False, f"Page {i + 1} explains reliable document extraction and bounded indexing.")]
                 for i in range(args.pages)]
        if args.probe_chars is not None:
            pages = [[(50, 740, 11, False, "x" * args.probe_chars)] if i < args.text_pages else []
                     for i in range(args.pages)]
        (args.output / "large-manual.pdf").write_bytes(pdf(pages, "Generated large manual"))
        return
    fixtures = {
        "paper": ("Small search paper", [[
            (50, 740, 18, True, "Search Evaluation"),
            (50, 700, 11, False, "Abstract: Search connects questions to source code."),
            (50, 650, 14, True, "Method"),
            (50, 625, 10, False, "We compare lexical retrieval"),
            (50, 610, 10, False, "with dense retrieval on code."),
            (320, 650, 14, True, "Results"),
            (320, 625, 10, False, "Relevant files appear first."),
            (320, 610, 10, False, "Older answers remain useful.")]]),
        "invoice": (None, [[
            (50, 740, 18, True, "Invoice 1042"),
            (50, 700, 11, False, "Customer: Example Workshop"),
            (50, 680, 11, False, "Service: Repository document indexing"),
            (50, 660, 11, False, "Quantity: 2   Unit price: 25.00   Total: 50.00"),
            (50, 620, 11, False, "Payment is due within thirty days.")]]),
        "manual": ("Indexer Manual", [[
            (50, 740, 18, True, "Installation"),
            (50, 700, 11, False, "Install the binary and open a repository directory."),
            (50, 680, 11, False, "Run the index command to catalogue your source files.")], [
            (50, 740, 18, True, "Searching"),
            (50, 700, 11, False, "Search accepts words and returns matching source sections."),
            (50, 680, 11, False, "Open a result to inspect its original context.")]]),
        "scanned": ("No text layer", [[]]),
    }
    for name, (title, pages) in fixtures.items():
        (args.output / f"{name}.pdf").write_bytes(pdf(pages, title))


if __name__ == "__main__":
    main()
