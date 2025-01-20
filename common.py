#!/usr/bin/env python3

# Common functionality for working with STRING DB / JensenLab tagger data

import sys
import re

from collections.abc import Iterator
from itertools import tee
from collections import namedtuple
from itertools import zip_longest


# From https://bitbucket.org/larsjuhljensen/tagger/
TYPE_MAP = {
    -1: 'Chemical',
    -2: 'Organism',    # NCBI species taxonomy id (tagging species)
    -3: 'Organism',    # NCBI species taxonomy id (tagging proteins)
    -11: 'Wikipedia',
    -21: 'Biological_process',    # GO biological process
    -22: 'Cellular_component',    # GO cellular component
    -23: 'Molecular_function',    # GO molecular function
    -24: 'GO_other',    # GO other (unused)
    -25: 'Tissue',    # BTO tissues
    -26: 'Disease',    # DOID diseases
    -27: 'Environment',    # ENVO environments
    -28: 'Phenotype',    # APO phenotypes
    -29: 'Phenotype',    # FYPO phenotypes
    -30: 'Phenotype',    # MPheno phenotypes
    -31: 'Behaviour',    # NBO behaviors
    -36: 'Phenotype',    # mammalian phenotypes
}


# Regex for STRING identifiers that should be filtered out. Katerina:
# COG/KOG are EggNOG identifiers and irrelevant for normalization.
FILTER_NORM_RE = re.compile(r'^[KC]OG.*')


# Regex for STRING identifiers prefixed by a taxnomy ID that should be
# filtered out. Katerina: everything that has a txid 1 (root of
# taxonomic tree) and 2759 (Eukaryota) should be filtered out.
FILTER_PREFIXED_NORM_RE = re.compile(r'^(1|2759)\..*')


def type_name(type_):
    """Map JensenLab tagger numeric types to names. No-op for non-numeric."""
    if isinstance(type_, str):
        try:
            type_ = int(type_)
        except ValueError:
            return type_    # Not numeric
    if type_ > 0:
        return 'Protein'
    else:
        return TYPE_MAP.get(type_, 'UNKNOWN-TYPE')


class StringDocument:
    def __init__(self, id_, other_ids, authors, forum, year, text):
        self.id = id_
        self.other_ids = other_ids
        self.authors = authors
        self.forum = forum
        self.year = year
        #self.text = stringdb_unescape_text(text)
        self.text = text

    def __str__(self):
        return '\t'.join([
            self.id,
            self.other_ids,
            self.authors,
            self.forum,
            self.year,
            self.text
            #stringdb_escape_text(self.text)
        ])


class StringSpan:
    def __init__(self, doc_id, par_num, sent_num, start, end, text, type_,
                 serial, source=None, line_no=None, no_type_mapping=False):
        self.doc_id = doc_id
        self.par_num = par_num
        self.sent_num = sent_num
        self.start = start
        self.end = end
        self.text = text
        self.type = type_
        self.serials = serial.split(',')
        self.source = source
        self.line_no = line_no

        self.serials = [
            s for s in self.serials
            if not FILTER_NORM_RE.match(s)
        ]

        if not no_type_mapping:
            orig_type = self.type
            self.type = type_name(self.type)
            if self.type == 'Protein' and orig_type != 'Protein':
                # STRING norm IDs should be prefixed by the organism ID
                self.serials = [f'{orig_type}.{s}' for s in self.serials]
                # And normalization to normalization to orthologous groups
                # dropped
                self.serials = [
                    s for s in self.serials
                    if not FILTER_PREFIXED_NORM_RE.match(s)
                ]

    def matches(self, other):
        return self.span_matches(other) and self.type_matches(other)

    def overlap_matches(self, other):
        return self.overlaps(other) and self.type_matches(other)

    def type_matches(self, other):
        # type matching is case-insensitive
        return (self.type == other.type or
                self.type.lower() == other.type.lower())

    def span_matches(self, other):
        return self.start == other.start and self.end == other.end

    def overlaps(self, other):
        return not (self.end+1 <= other.start or other.end+1 <= self.start)

    def contains(self, other):
        return ((self.start <= other.start and self.end > other.end) or
                (self.start < other.start and self.end >= other.end))

    def __lt__(self, other):
        if self.start != other.start:
            return self.start < other.start
        elif self.end != other.end:
            return self.end > other.end
        else:
            return self.type < other.type    # arbitrary but fixed

    def __str__(self):
        fields = [
            self.doc_id, self.par_num, self.sent_num,
            str(self.start), str(self.end),
            self.text, self.type, ','.join(self.serials)
        ]
        if self.source is not None:
            fields.append(self.source)
        return '\t'.join(fields)


class LookaheadIterator(Iterator):
    """Lookahead iterator from http://stackoverflow.com/a/1518097."""

    def __init__(self, it, start=0):
        self._it, self._nextit = tee(iter(it))
        self.index = start - 1
        self._advance()

    def _advance(self):
        self.lookahead = next(self._nextit, None)
        self.index = self.index + 1

    def __next__(self):
        self._advance()
        return next(self._it)

    def __bool__(self):
        return self.lookahead is not None


class DocReader(Iterator):
    """Reader for database_documents.tsv format."""
    def __init__(self, stream):
        self.stream = stream
        self.iter = LookaheadIterator(stream, start=1)

    def current_doc_id(self):
        """Return id of document at the current position of the stream."""
        if self.iter.lookahead is None:
            return None
        else:
            return self.iter.lookahead.split()[0]

    def __next__(self):
        ln = self.iter.index
        line = next(self.iter)
        try:
            doc = parse_stringdb_input_line(line)
        except:
            raise ValueError(f'error parsing {self.stream.name} line {ln}: '
                             f'{line}')
        return doc


class SpanReader:
    """Reader for all_matches.tsv format."""

    def __init__(self, stream, source=None, raise_on_error=False,
                 no_type_mapping=False):
        self.stream = stream
        self.source = source
        self.raise_on_error = raise_on_error
        self.no_type_mapping = no_type_mapping
        self.iter = LookaheadIterator(stream, start=1)
        self.errors = 0

    def current_doc_id(self):
        """Return id of document at the current position of the stream."""
        if self.iter.lookahead is None:
            return None
        else:
            return self.iter.lookahead.split()[0]

    def document_lines(self, doc_id):
        """Return lines for document doc_id and advance past them."""
        spans = []
        while self.current_doc_id() == doc_id:
            spans.append(next(self.iter))
        return spans

    def document_spans(self, doc_id):
        """Return spans for document doc_id and advance past them.

        If doc_id does not match the current position of the stream, returns
        an empty list without advancing in the stream.
        """
        spans = []
        while self.current_doc_id() == doc_id:
            try:
                line = self.iter.lookahead.rstrip('\n')
                span = parse_stringdb_span_line(
                    line,
                    source=self.source,
                    no_type_mapping=self.no_type_mapping
                )
                span.line_no = self.iter.index
                spans.append(span)
            except Exception as e:
                self.errors += 1
                print(f'error parsing {self.stream.name} line '
                      f'{self.iter.index}: {e}: {line}', file=sys.stderr)
                if self.raise_on_error:
                    raise
            next(self.iter)
        return spans


def stringdb_escape_text(text):
    """Escape text for database_documents.tsv format."""
    return text.replace('\\', '\\\\').replace('\t', '\\t')


def stringdb_unescape_text(text):
    """Unescape text field in database_documents.tsv format."""
    unescaped = []
    pair_iter = zip_longest(text, text[1:])
    for char, next_ in pair_iter:
        if char == '\\' and next_ == '\\':
            # Double backslash -> single backslash
            unescaped.append('\\')
            next(pair_iter)
        elif char == '\\' and next_ == 't':
            # Backslash + t -> tab character
            unescaped.append('\t')
            next(pair_iter)
        else:
            unescaped.append(char)
    return ''.join(unescaped)


def parse_stringdb_input_line(line):
    """Parse line in database_documents.tsv format, return StringDocument."""
    line = line.rstrip('\n')
    fields = line.split('\t',  maxsplit=5) # with max split 6 the entire last field should be the text with title
    doc_id, other_ids, authors, forum, year, text = fields
    return StringDocument(doc_id, other_ids, authors, forum, year, text)


def parse_stringdb_span_line(line, source=None, no_type_mapping=False):
    """Parse line in all_matches.tsv format, return StringSpan."""
    line = line.rstrip('\n')
    fields = line.split('\t')
    doc_id, par_num, sent_num, start, end, text, type_, serial = fields[:8]
    if source is True:    # source as separate field
        source = fields[8]
        assert len(fields) == 9
    else:
        assert len(fields) == 8
    start, end = int(start), int(end)
    return StringSpan(
        doc_id, par_num, sent_num, start, end, text, type_, serial,
        source=source, no_type_mapping=no_type_mapping
    )


def stream_documents(fn):
    with open(fn) as f:
        for ln, l in enumerate(f, start=1):
            try:
                document = parse_stringdb_input_line(l)
            except Exception as e:
                raise ValueError('failed to parse {} line {}'.format(fn, ln))
            yield document


def open_file(fn, mode, options):
    if options.char_offsets:
        return open(fn, mode)
    else:
        # https://www.python.org/dev/peps/pep-0383/ (Python 3.1+)
        return open(fn, mode, encoding='ascii', errors='surrogateescape')


def load_ids(fn, options):
    ids = set()
    with open_file(fn, 'r', options) as f:
        for ln, l in enumerate(f, start=1):
            l = l.strip()
            ids.add(l)
    print(f'read {len(ids)} ids from {fn}', file=sys.stderr)
    return ids


def safe_str(string):
    # workaround for 'utf-8' codec can't encode [...]: surrogates not allowed
    return string.encode('utf-8', 'replace').decode()


def unique(sequence):
    """Return unique items in sequence, preserving order."""
    # https://www.peterbe.com/plog/fastest-way-to-uniquify-a-list-in-python-3.6
    return list(dict.fromkeys(sequence))


if __name__ == '__main__':
    import sys

    # Test I/O
    for fn in sys.argv[1:]:
        for doc in stream_documents(fn):
            print(doc.doc_id, len(doc.text.split()), 'tokens')
