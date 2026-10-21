#!/usr/bin/env python3
"""Generate original single-image/unsupported/hostile PDF page fixtures, stdlib only."""
import argparse, base64, zlib
from pathlib import Path
# Original 32x16 red/blue RGB image; embedded bytes keep regeneration independent
# of Pillow or a platform JPEG encoder. No third-party photograph or document.
JPEG = base64.b64decode('/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAMCAgMCAgMDAwMEAwMEBQgFBQQEBQoHBwYIDAoMDAsKCwsNDhIQDQ4RDgsLEBYQERMUFRUVDA8XGBYUGBIUFRT/2wBDAQMEBAUEBQkFBQkUDQsNFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBT/wAARCAAQACADAREAAhEBAxEB/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/8QAHwEAAwEBAQEBAQEBAQAAAAAAAAECAwQFBgcICQoL/8QAtREAAgECBAQDBAcFBAQAAQJ3AAECAxEEBSExBhJBUQdhcRMiMoEIFEKRobHBCSMzUvAVYnLRChYkNOEl8RcYGRomJygpKjU2Nzg5OkNERUZHSElKU1RVVldYWVpjZGVmZ2hpanN0dXZ3eHl6goOEhYaHiImKkpOUlZaXmJmaoqOkpaanqKmqsrO0tba3uLm6wsPExcbHyMnK0tPU1dbX2Nna4uPk5ebn6Onq8vP09fb3+Pn6/9oADAMBAAIRAxEAPwDxKvyo/v0KAOPr/Tg/y3CgDsK/zHP9SAoA4+v9OD/LcKAP/9k=')
CASES = ('jpeg','raw','flate-raw','flate-content','overlay','multiple','clipped',
         'rotated','masked','inverted','form','annotations','garbage','wrong-dimensions',
         'content-bomb','decode-bomb','first-overlay','stencil','square-annotation')

def stream(data, attributes=''):
    return f'<< /Length {len(data)} {attributes} >>\nstream\n'.encode()+data+b'\nendstream'

def pdf(case, pages=1):
    objects=[b'<< /Type /Catalog /Pages 2 0 R >>',b'']
    kids=[]
    for index in range(pages):
        page=len(objects)+1; kids.append(f'{page} 0 R')
        content=b'q 512 0 0 692 50 50 cm /Scan Do Q'
        page_extra=''
        image_extra=''
        subtype='Image'; width=32; height=16
        data=JPEG; image_filter='/Filter /DCTDecode'
        color='/ColorSpace /DeviceRGB /BitsPerComponent 8'
        content_filter=''
        if case in ('raw','flate-raw','decode-bomb'):
            data=bytes([200,20,20])*(width*height);image_filter=''
            if case=='decode-bomb': data=bytes(1024*1024)
            if case!='raw': data=zlib.compress(data,9);image_filter='/Filter /FlateDecode'
        if case=='overlay' or (case=='first-overlay' and index==0): content+=b' 0 0 10 10 re f'
        if case=='multiple': content+=b' q 10 0 0 10 0 0 cm /Scan Do Q'
        if case=='clipped': page_extra='/CropBox [100 100 500 700]'
        if case=='rotated': page_extra='/Rotate 90'
        if case=='masked': image_extra='/ImageMask true'
        if case=='stencil':
            data=bytes(width*height//8);image_filter='';image_extra='/ImageMask true'
            color='/BitsPerComponent 1'
        if case=='inverted': image_extra='/Decode [1 0 1 0 1 0]'
        if case=='form': subtype='Form'
        if case=='annotations': page_extra='/Annots [<< /Subtype /Text /Rect [0 0 10 10] >>]'
        if case=='square-annotation':
            page_extra='/Annots [<< /Type /Annot /Subtype /Square /Rect [0 0 10 10] /C [1 0 0] /IC [1 0 0] /F 4 >>]'
        if case=='garbage': content+=b' unexpected-operator'
        if case=='wrong-dimensions': width=99
        if case=='content-bomb': content=b'q '*1000000
        if case in ('content-bomb','flate-content'):
            content=zlib.compress(content,9);content_filter='/Filter /FlateDecode'
        objects.append((f'<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] '
            f'/Resources << /XObject << /Scan {page+2} 0 R >> >> /Contents {page+1} 0 R {page_extra} >>').encode())
        objects.append(stream(content,content_filter))
        objects.append(stream(data,f'/Type /XObject /Subtype /{subtype} /Width {width} /Height {height} '
            f'{color} {image_filter} {image_extra}'))
    objects[1]=f"<< /Type /Pages /Count {pages} /Kids [{' '.join(kids)}] >>".encode()
    objects.append(b'<< /Title (Generated scan) >>');info=len(objects)
    output=bytearray(b'%PDF-1.7\n%\xe2\xe3\xcf\xd3\n');offsets=[0]
    for index,obj in enumerate(objects,1):
        offsets.append(len(output));output.extend(f'{index} 0 obj\n'.encode()+obj+b'\nendobj\n')
    xref=len(output);output.extend(f'xref\n0 {len(offsets)}\n0000000000 65535 f \n'.encode())
    for offset in offsets[1:]: output.extend(f'{offset:010d} 00000 n \n'.encode())
    output.extend(f'trailer\n<< /Size {len(offsets)} /Root 1 0 R /Info {info} 0 R >>\nstartxref\n{xref}\n%%EOF\n'.encode())
    return output

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--case',choices=CASES,action='append')
    parser.add_argument('--pages',type=int,default=1)
    args=parser.parse_args()
    if args.pages<1: parser.error('--pages must be positive')
    args.output.mkdir(parents=True,exist_ok=True)
    for case in args.case or CASES:
        (args.output/f'scanned-{case}.pdf').write_bytes(pdf(case,args.pages))
if __name__=='__main__': main()
