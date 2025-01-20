#!/usr/bin/env python3

import sys
import os

from collections import defaultdict
from argparse import ArgumentParser
from logging import error

from common import DocReader, SpanReader, open_file

#python3 tagger2standoff.py 20_documents_GWAS.tsv 20_database_matches.tsv GWAS_test
# Placeholder value for missing norm IDs
DUMMY_SERIAL = 'SERIAL'


def argparser():
    ap = ArgumentParser()
    ap.add_argument('--char-offsets', default=False, action='store_true',
                    help='offsets are character- instead of byte-based')
    ap.add_argument('docs', help='documents in database_documents.tsv format')
    ap.add_argument('tags', help='tagged strings in all_matches.tsv format')
    ap.add_argument('dir', help='output directory')
    return ap


def make_offset_map(text):
    """Return mapping from offsets to surrogate-escaped ascii to characters."""
    # TODO add fast path for all-ascii
    offsets, byte_offset, char_offset = [], 0, 0
    while byte_offset < len(text):
        for length in range(1, len(text)):
            span = text[byte_offset:byte_offset+length]
            try:
                # https://lucumr.pocoo.org/2013/7/2/the-updated-guide-to-unicode/#different-types-of-unicode-strings
                encoded = span.encode('utf-8', errors='surrogateescape')
                decoded = encoded.decode('utf-8')
                assert len(decoded) == 1    # single character
                break
            except UnicodeDecodeError:
                pass    # assume incomplete, try longer
        for i in range(length):
            offsets.append(char_offset)
        byte_offset += length
        char_offset += 1
    offsets.append(char_offset)
    return dict(enumerate(offsets))


def normalize_type(type_):
    type_ = type_.lower()
    type_ = type_[0].upper() + type_[1:]
    return type_


def deduplicate_spans(spans, options):
    """Combine serials and sources for spans with identical boundaries and
    types."""
    make_key = lambda s: (s.start, s.end, s.type)
    span_map = defaultdict(list)
    deduped = []
    for span in spans:
        key = make_key(span)
        span.sources = set([span.source])
        if key not in span_map:
            deduped.append(span)
        else:
            span_map[key][0].serials.extend(span.serials)
            span_map[key][0].sources.update(span.sources)
        span_map[key].append(span)
    for span in spans:
        span.source = ','.join(sorted(span.sources))
    return deduped


def convert_single(doc, spans, out_dir, options):
    for span in spans:
        span.type = normalize_type(span.type)
    spans = deduplicate_spans(spans, options)
    with open_file(os.path.join(out_dir, f'{doc.id}.txt'), 'w', options) as f:
        print(doc.text.replace('\t', '\n'), file=f)
    offset_map = make_offset_map(doc.text)
    with open_file(os.path.join(out_dir, f'{doc.id}.ann'), 'w', options) as f:
        n = 1
        for i, span in enumerate(spans, start=1):
            s, e = span.start, span.end+1    # end-exclusive
            pmid, par, sent, end = span.doc_id, span.par_num, span.sent_num, span.end
            s, e = offset_map[s], offset_map[e]    # char offsets
            t = f'{span.type}'
            # if len(span.sources) == 2:    # assume two sources
            #     t = f'{span.type}'
            # else:
            #     t = f'{span.type}-{span.source}'
            print(f'T{i}\t{t} {s} {e}\t{span.text}', file=f)
            
            for serial in span.serials:
                if serial != DUMMY_SERIAL:
                    #1	AnnotatorNotes T1	this annotation is suspect
                    print(f'#{n}\tAnnotatorNotes T{i} T{i}|{pmid}|{par}|{sent}|{s}|{end}|{serial}',
                          file=f)
                    n += 1


def convert_to_standoff(doc_fn, tag_fn, out_dir, options):
    NOTE_TYPE = 'AnnotatorNotes'
    with open_file(doc_fn, 'r', options) as doc_f:
        doc_reader = DocReader(doc_f)
        with open_file(tag_fn, 'r', options) as tag_f:
            # Read spans that include source information
            span_reader = SpanReader(tag_f, source=True)
            for doc in doc_reader:
                spans = list(span_reader.document_spans(doc.id))
                try:
                    convert_single(doc, spans, out_dir, options)
                except Exception as e:
                    error(f'failed to convert {doc.id}: {e}')
                   # raise


def main(argv):
    args = argparser().parse_args(argv[1:])
    convert_to_standoff(args.docs, args.tags, args.dir, args)
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))

