# metadataXtractor - extracts technical metadata from mzML files
# by Lev Levitsky and Magnus Palmblad

from pyteomics import mzml
import sys

# Ensure an argument is passed
if len(sys.argv) < 2:
    print("Usage: python metadataXtractor.py <filename>")
    sys.exit(1)

# Get the filename from command-line argument
fname = sys.argv[1]

# create reader
reader = mzml.MzML(fname, use_index=True)

# extract instrument (works for Thermo and Bruker from msconvert)
group = next(reader.iterfind('referenceableParamGroup[@id="commonInstrumentParams" or @id="CommonInstrumentParams"]'))
#print(group)
instrument = next((key for key, value in group.items() if value == ''), None)
print("filename:", fname, "instrument:", instrument)

# Extract instrument model from <instrumentConfiguration> (Waters)
# group = next(reader.iterfind('instrumentConfigurationList/instrumentConfiguration'))
# instrument = next((key for key, value in group.items() if value == ''), None)
# print(instrument)

# extract ionization method
reader.reset()
group = next(reader.iterfind('instrumentConfiguration/componentList/source'))
source = next((key for key, value in group.items() if value == ''), None)
print("filename:", fname, "ionization method:", source)

# gradient length; only works with indexing
reader.reset()
def start_time(scan):
    return scan['scanList']['scan'][0]['scan start time']

# extract gradient time unit (not working)
# def start_time_unit(scan):
#     return scan['scanList']['scan'][0]['unitName']

gradientTime = start_time(reader[-1]) - start_time(reader[0])
# gradientTimeUnit = start_time_unit(reader[0])

print("filename:", fname, "gradient time:", gradientTime)

exit(0)


# all types of fragmentation used in the file
reader.reset()
keys = set()

for s in reader:
    if s['ms level'] > 1:
        keys.update(s['precursorList']['precursor'][0]['activation'])

keys  # still need to remove extra keys like "collision energy"

reader.reset()
keys = set()

for d in reader.iterfind('activation'):
        keys.update(s['precursorList']['precursor'][0]['activation'])

keys   # still need to remove extra keys like "collision energy"
