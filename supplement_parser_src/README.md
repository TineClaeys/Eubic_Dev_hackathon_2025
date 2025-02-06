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
After loading the supplementary tables, columns whose names match entity names or allowed synonyms are extracted. Entity name synonyms are replaced with the entity names (e.g. *mass spectrometer* will be replaced with *Instrument*)
Extracted information are saved into a .json file.
