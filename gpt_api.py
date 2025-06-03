from openai import OpenAI
import argparse
import os 
import openai_messages_token_helper 
import logging

MODEL = "gpt-4.1-nano-2025-04-14" 

OUTPUT_DIR = f"sdrf_tagged_api/{MODEL}"  # Directory to save output files

# Create output directory if it doesn't exist
os.makedirs("sdrf_tagged_api", exist_ok=True)
os.makedirs(f"sdrf_tagged_api/{MODEL}", exist_ok=True)


# Set up logging to track API usage and errors
logging.basicConfig(
    filename=f'{MODEL}_gpt_api.log',
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s'
)


client = OpenAI(
  api_key= os.getenv("OPENAI_API_KEY"),  # Set your OpenAI API key in the environment variables 
)

def extract_metadata_from_manuscript(text, filename=None):
    messages=[
        {"role": "system", "content": """GPT Instructions: Extract SDRF-Proteomics Metadata in BRAT Format (Coordinate-Free)

    Task
    Extract SDRF-Proteomics metadata entities from a `.txt` manuscript. Return each identified entity with its label and exact matched phrase. Do not include character offsets.

    Input
    You are given:
    - A `.txt` file with a scientific manuscript.

    Annotate From
    - Title
    - Abstract
    - Materials & Methods (explicit or inferred using keywords: e.g., "digested", "mass spectrometry", "sample preparation", "enrichment").

    Do Not Annotate From
    - Introduction, Results, Discussion
    - Supplementary data, tables, captions, acknowledgments, references
    - Any non-specified section

    Output Format
    Return a list of entity mentions in the following format:
    <Label>: <MatchedText>

    Example:
    Organism: mouse
    OrganismPart: liver
    Instrument: Orbitrap mass spectrometer

    Allowed Labels
    Use only the official SDRF-Proteomics metadata terms (full list provided with input). Define this list before starting annotation.

    Labeling Rules
    - Annotate exact, explicit mentions only — no paraphrases.
    - Use the most specific applicable label.
    - If multiple labels apply:
    1. Use context to disambiguate.
    2. Prefer specific over general.
    3. If unsure, skip.

    Do not normalize or invent text. Annotate as-is.

    Span Requirements
    - Annotate full, grammatical spans.
    - No overlapping, nested, or partial-word matches.
    - Allow case-insensitive, plural mentions.

    Redundancy
    - Annotate all valid mentions, even if repeated.
    - collapse or merge duplicates.

    Validation Steps (Required for Each Entity)
    1. Grammatical Check – Is the phrase valid and self-contained?
    2. Biological Context Check – Does the surrounding context support the label semantically and biologically?

    If any test fails, do not include the entity. Summarize all included entities in the final output.

    Few-Shot Guidance (Optional)
    If `.ann` examples are provided:
    - Learn from label usage and phrasing patterns.
    - Never copy directly. Always verify from current text.
        Extract SDRF labels from the provided txt, these are the options: AcquisitionMethod, Age, AlkylationReagent, AnatomicSiteTumor, AncestryCategory, AssayName, Bait, BMI, BiologicalReplicate, CellLine, CellPart, CellType, CleavageAgent, CollisionEnergy, Compound, ConcentrationOfCompound, Depletion, DevelopmentalStage, Disease, DiseaseTreatment, EnrichmentMethod, Experiment, FactorValue, FlowRateChromatogram, FractionationMethod, FractionIdentifier, FractionationFraction, FragmentationMethod, FragmentMassTolerance, GeneticModification, Genotype, GradientTime, GrowthRate, Instrument, IonizationType, Label, MaterialType, Modification, MS2MassAnalyzer, NumberOfMissedCleavages, NumberOfFractions, NumberOfTechnicalReplicates, NumberOfSamples, NumberOfBiologicalReplicates, Organism, OrganismPart, OriginSiteDisease, PrecursorMassTolerance, PooledSample, ReductionReagent, SamplingTime, SampleTreatment, Separation, Sex, SourceName, Specimen, SpikedCompound, Staining, Strain, SupplementaryFile, SyntheticPeptide, TumorCellularity, TumorGrade, TumorSize, TumorSite, TumorStage, Time, Temperature, Treatment, TechnicalReplicate. Keep in mind that FactorValue is the study variable and can be a double annotation with another label"""},
        {"role": "user", "content": text },
    ]

    TOKEN_COUNT_MODEL = "gpt-4"  # Fallback to standard gpt-4 for token counting
    
    # Fixed: Count tokens for all messages
    total_tokens = 0
    for message in messages:
        token_count = openai_messages_token_helper.count_tokens_for_message(TOKEN_COUNT_MODEL, message)
        total_tokens += token_count
    
    print(f"Token count for this request: {total_tokens}")
    if filename:
        logging.info(f"Processing file: {filename}, Token count: {total_tokens}")
    else:
        logging.info(f"Token count: {total_tokens}")

    
    print(f"Token count for this request: {total_tokens}")
    if filename:
        logging.info(f"Processing file: {filename}, Token count: {total_tokens}")
    else:
        logging.info(f"Token count: {total_tokens}")

    # Fixed: Added missing messages parameter
    completion = client.chat.completions.create(
        model=MODEL,
        messages=messages,  
        store=True,
    )

    return completion.choices[0].message.content

def main():
    parser = argparse.ArgumentParser(description="Extract metadata in SDRF format from a manuscript using OpenAI API")
    parser.add_argument("--input_folder", type=str, required=True, help="Folder with manuscript text files")
    args = parser.parse_args()
    input_folder = args.input_folder
    
    for filename in os.listdir(input_folder):
        if filename.endswith(".txt"):
            file_path = os.path.join(input_folder, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()  
            metadata = extract_metadata_from_manuscript(text, filename=filename)
            output_filename = f"{MODEL}_{filename.replace('.txt', '.sdrf').lstrip('_')}"
            output_path = os.path.join(f"{OUTPUT_DIR}", output_filename)
            with open(output_path, 'w', encoding='utf-8') as output_file:
                output_file.write(metadata)
            print(f"Processed {filename}, extracted metadata saved to {output_filename}")
            logging.info(f"Processed {filename}, output: {output_filename}")

if __name__ == "__main__":
    main()