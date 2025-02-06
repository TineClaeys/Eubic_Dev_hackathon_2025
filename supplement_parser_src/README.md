### File input
Supplement parser is able to open and extract tables from .tsv, .csv, .docx, .xlsx, .xls and .txt files.

### Entity names
We have created a dictionary of established entity names and their possible synonyms which may appear in literature.

Example:
| Entity name | Synonyms |
| ----------- | -------- |
| ... | ... |
| GradientTime | gradient duration, chromatographic gradient |
| GrowthRate | growth rate, proliferation rate |
| Instrument | instrument, mass spectrometer |
| ... | ... |

### Extraction of sample relevant information
After loading the supplementary tables, columns whose names match entity names or allowed synonyms are extracted. Entity name synonyms are replaced with the entity names (e.g. *mass spectrometer* will be replaced with *Instrument*). Unique sample identifiers are expected to be located and extracted from the first table column.

Extracted information are saved into a .json file.

### Limitation
This supplement parser is currently only able to extract information from tables. Information included in descriptive text passages cannot be parsed.
