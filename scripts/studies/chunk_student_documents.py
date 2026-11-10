
import json
import re
import hashlib
import unicodedata

from pathlib import Path

import fitz
from tqdm import tqdm
from transformers import AutoTokenizer


# --------------------------------------------------
# Configuration
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent.parent

INPUT_DIR = BASE_DIR / "data/studies/docs"

OUTPUT_DIR = BASE_DIR / "data/studies/chunks"

OUTPUT_FILE = OUTPUT_DIR / "segments.jsonl"

MODEL_PATH = (
    "/home/maximos/.cache/huggingface/hub/"
    "models--BAAI--bge-m3/snapshots/"
    "5617a9f61b028005a4858fdac845db406aefb181"
)

TARGET_TOKENS = 500
MAX_TOKENS = 750
OVERLAP_TOKENS = 80

print('INPUT_DIR: ', INPUT_DIR)

# --------------------------------------------------
# Tokenizer
# --------------------------------------------------

print("Loading BGE-M3 tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    local_files_only=True
)


def count_tokens(text):
    return len(
        tokenizer.encode(
            text,
            add_special_tokens=False
        )
    )


# --------------------------------------------------
# Document identification
# --------------------------------------------------

def make_document_id(filename):

    normalized = unicodedata.normalize(
        "NFKD",
        filename
    )

    normalized = "".join(
        c for c in normalized
        if not unicodedata.combining(c)
    )

    slug = re.sub(
        r"[^a-zA-Z0-9]+",
        "_",
        normalized
    ).strip("_").lower()

    # Keep the ID stable and avoid filename collisions.
    digest = hashlib.sha1(
        filename.encode("utf-8")
    ).hexdigest()[:8]

    return f"student_{slug[:60]}_{digest}"


def get_document_title(pdf):

    metadata = pdf.metadata or {}

    title = (metadata.get("title") or "").strip()

    # PDF metadata is sometimes missing or uninformative.
    if not title or title.lower() in (
        "untitled",
        "microsoft word",
        "document"
    ):
        return None

    return title


# --------------------------------------------------
# PDF outline / bookmarks
# --------------------------------------------------

def extract_outline(pdf):

    """
    Returns a mapping:

        page_number -> hierarchical section path

    PDF page numbers are 1-based.
    """

    toc = pdf.get_toc()

    if not toc:
        return {}

    outline = {}
    hierarchy = []

    entries = sorted(
        toc,
        key=lambda x: x[2]
    )

    for level, title, page in entries:

        hierarchy = hierarchy[:level - 1]

        hierarchy.append(title.strip())

        outline[page] = hierarchy.copy()

    # Expand section paths to all pages.
    page_paths = {}

    current_path = []

    for page_number in range(1, len(pdf) + 1):

        if page_number in outline:
            current_path = outline[page_number]

        page_paths[page_number] = current_path.copy()

    return page_paths


# --------------------------------------------------
# Heading detection
# --------------------------------------------------

HEADING_PATTERNS = [

    r"^Άρθρο\s+\d+.*",

    r"^ΚΕΦΑΛΑΙΟ\s+[Α-ΩA-Z0-9IVX]+.*",

    r"^ΜΕΡΟΣ\s+[Α-ΩA-Z0-9IVX]+.*",

    r"^ΠΑΡΑΡΤΗΜΑ\s+[Α-ΩA-Z0-9IVX]+.*",

    r"^ΕΝΟΤΗΤΑ\s+[Α-ΩA-Z0-9IVX]+.*",

    r"^ΠΙΝΑΚΑΣ\s+.*",

]

HEADING_REGEX = re.compile(
    "|".join(
        f"(?:{pattern})"
        for pattern in HEADING_PATTERNS
    ),
    re.IGNORECASE
)


def is_heading(line):

    line = line.strip()

    if not line or len(line) > 160:
        return False

    if HEADING_REGEX.match(line):
        return True

    # Numbered headings, e.g. "1. Γενικές Διατάξεις".
    if re.match(
        r"^\d+(?:\.\d+)*\.?\s+\S",
        line
    ):
        if len(line) < 120 and not line.endswith(
            (".", ",", ";", ":")
        ):
            return True

    return False


# --------------------------------------------------
# Extract paragraphs and section information
# --------------------------------------------------

def extract_pdf_content(pdf_path):

    pdf = fitz.open(pdf_path)

    metadata_title = get_document_title(pdf)

    document_title = (
        metadata_title
        if metadata_title
        else pdf_path.stem.replace("_", " ")
    )

    document_id = make_document_id(
        pdf_path.name
    )

    outline_paths = extract_outline(pdf)

    records = []

    current_section_path = []

    for page_number, page in enumerate(
        pdf,
        start=1
    ):

        page_text = page.get_text(
            "text",
            sort=True
        )

        if not page_text.strip():
            print(
                f"Warning: No extractable text in "
                f"{pdf_path.name}, page {page_number}"
            )
            continue

        # Start with the PDF outline information.
        if outline_paths.get(page_number):
            current_section_path = (
                outline_paths[page_number].copy()
            )

        # Remove common PDF line-break hyphenation.
        page_text = re.sub(
            r"(\w)-\n(\w)",
            r"\1\2",
            page_text
        )

        lines = page_text.splitlines()

        paragraph_lines = []

        def flush_paragraph():

            if not paragraph_lines:
                return

            text = " ".join(
                line.strip()
                for line in paragraph_lines
            ).strip()

            if text:
                records.append({

                    "text": text,

                    "page": page_number,

                    "section_path":
                        current_section_path.copy()

                })

            paragraph_lines.clear()

        for line in lines:

            line = line.strip()

            if not line:
                flush_paragraph()
                continue

            if is_heading(line):

                flush_paragraph()

                current_section_path = (
                    current_section_path + [line]
                )

                # Store the heading as context as well.
                records.append({

                    "text": line,

                    "page": page_number,

                    "section_path":
                        current_section_path.copy()

                })

            else:

                paragraph_lines.append(line)

        flush_paragraph()

    pdf.close()

    return {
        "document_id": document_id,
        "document_title": document_title,
        "source_file": pdf_path.name,
        "records": records
    }


# --------------------------------------------------
# Token-aware chunking
# --------------------------------------------------

def chunk_section(records):

    """
    Pack paragraphs into chunks.

    A new chunk is normally created around TARGET_TOKENS.
    MAX_TOKENS is a hard limit.
    The next chunk inherits OVERLAP_TOKENS from the
    previous chunk.
    """

    chunks = []

    current_ids = []

    current_pages = []

    current_section_path = []

    def emit_chunk():

        nonlocal current_ids
        nonlocal current_pages

        if not current_ids:
            return

        text = tokenizer.decode(
            current_ids,
            skip_special_tokens=True
        ).strip()

        if text:

            chunks.append({

                "text": text,

                "page_start": min(current_pages),

                "page_end": max(current_pages),

                "section_path":
                    current_section_path.copy(),

                "token_count": len(current_ids)

            })

        # Preserve trailing tokens for overlap.
        current_ids = current_ids[
            -OVERLAP_TOKENS:
        ]

        current_pages = current_pages[
            -OVERLAP_TOKENS:
        ]

    for record in records:

        text = record["text"]

        page = record["page"]

        ids = tokenizer.encode(
            text,
            add_special_tokens=False
        )

        if not ids:
            continue

        current_section_path = (
            record["section_path"].copy()
        )

        # Handle paragraphs larger than MAX_TOKENS.
        if len(ids) > MAX_TOKENS - OVERLAP_TOKENS:

            if current_ids:
                emit_chunk()

            current_ids = []
            current_pages = []

            for start in range(
                0,
                len(ids),
                MAX_TOKENS - OVERLAP_TOKENS
            ):

                part = ids[
                    start:start + MAX_TOKENS
                ]

                chunks.append({

                    "text": tokenizer.decode(
                        part,
                        skip_special_tokens=True
                    ).strip(),

                    "page_start": page,

                    "page_end": page,

                    "section_path":
                        current_section_path.copy(),

                    "token_count": len(part)

                })

            current_ids = []
            current_pages = []

            continue

        # Add paragraph separator.
        paragraph_ids = (
            tokenizer.encode(
                "\n",
                add_special_tokens=False
            ) + ids
        )

        # Flush when the target length is reached.
        if (
            current_ids
            and len(current_ids) + len(paragraph_ids)
            > TARGET_TOKENS
        ):

            emit_chunk()

        # Enforce hard maximum.
        if (
            len(current_ids) + len(paragraph_ids)
            > MAX_TOKENS
        ):

            emit_chunk()

        current_ids.extend(paragraph_ids)

        current_pages.extend(
            [page] * len(paragraph_ids)
        )

    if current_ids:

        # Remove overlap-only remnants.
        if len(current_ids) > OVERLAP_TOKENS:

            text = tokenizer.decode(
                current_ids,
                skip_special_tokens=True
            ).strip()

            chunks.append({

                "text": text,

                "page_start": min(current_pages),

                "page_end": max(current_pages),

                "section_path":
                    current_section_path.copy(),

                "token_count": len(current_ids)

            })

    return chunks


# --------------------------------------------------
# Process individual documents
# --------------------------------------------------

def process_document(pdf_path):

    document = extract_pdf_content(pdf_path)

    records = document["records"]

    if not records:
        print(
            f"Warning: No text extracted from "
            f"{pdf_path.name}"
        )
        return []

    # Group consecutive records by section.
    groups = []

    current_group = []

    current_key = None

    for record in records:

        key = tuple(record["section_path"])

        if current_group and key != current_key:

            groups.append(current_group)

            current_group = []

        current_group.append(record)

        current_key = key

    if current_group:
        groups.append(current_group)

    chunks = []

    for group in groups:

        chunks.extend(
            chunk_section(group)
        )

    # Remove empty chunks.
    chunks = [
        chunk for chunk in chunks
        if chunk["text"].strip()
    ]

    segment_count = len(chunks)

    for index, chunk in enumerate(
        chunks,
        start=1
    ):

        chunk_id = (
            f"{document['document_id']}_"
            f"chunk_{index:05d}"
        )

        chunk.update({

            "document_id":
                document["document_id"],

            "document_title":
                document["document_title"],

            "source_file":
                document["source_file"],

            "section":
                " > ".join(chunk["section_path"]),

            "chunk_id": chunk_id,

            "segment_id": chunk_id,

            "segment_index": index,

            "segment_count": segment_count,

        })

    return chunks


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    pdf_files = sorted(
        INPUT_DIR.rglob("*.pdf")
    )

    print(f"Found {len(pdf_files)} PDF files.")

    all_segments = []

    for pdf_path in tqdm(
        pdf_files,
        desc="Processing PDFs"
    ):

        try:

            segments = process_document(
                pdf_path
            )

            all_segments.extend(segments)

            print(
                f"\n{pdf_path.name}: "
                f"{len(segments)} segments"
            )

        except Exception as e:

            print(
                f"\nERROR processing {pdf_path.name}: {e}"
            )

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        for segment in all_segments:

            f.write(
                json.dumps(
                    segment,
                    ensure_ascii=False
                ) + "\n"
            )

    print("\nProcessing completed.")

    print(f"PDF files: {len(pdf_files)}")

    print(f"Total segments: {len(all_segments)}")

    print(f"Output: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
