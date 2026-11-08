
import json
import re
import hashlib

from pathlib import Path
from urllib.parse import urlparse

import yaml
import tiktoken


# --------------------------------------------------
# Configuration
# --------------------------------------------------

INPUT_DIR = Path(
    "data/odigos_dioikitikon_diadikasion/articles"
)

OUTPUT_DIR = Path(
    "data/odigos_dioikitikon_diadikasion/chunks"
)

OUTPUT_FILE = OUTPUT_DIR / "chunks.jsonl"

TARGET_TOKENS = 500
MAX_TOKENS = 750
OVERLAP_TOKENS = 80

# Approximate tokenizer for initial experiments.
# Replace with the actual Krikri tokenizer later.
try:
    tokenizer = tiktoken.get_encoding("cl100k_base")

    def count_tokens(text):
        return len(tokenizer.encode(text))

except Exception:

    def count_tokens(text):
        return len(re.findall(r"\w+|[^\w\s]", text))


# --------------------------------------------------
# Metadata
# --------------------------------------------------

def read_document(path):

    content = path.read_text(encoding="utf-8")

    metadata = {}

    # Extract YAML frontmatter.
    match = re.match(
        r"\A---\s*\n(.*?)\n---\s*\n",
        content,
        re.DOTALL,
    )

    if match:

        metadata = yaml.safe_load(match.group(1)) or {}

        content = content[match.end():]

    return metadata, content


def get_document_metadata(path, metadata):

    title = metadata.get("title", path.stem)

    article_number = metadata.get("article_number")

    if article_number is None:

        match = re.search(r"(\d+)", path.stem)

        if match:
            article_number = int(match.group(1))

    document_id = (
        f"hmu_odigos_arthro_{article_number:02d}"
        if article_number is not None
        else f"hmu_odigos_{path.stem}"
    )

    return {
        "document_id": document_id,
        "document_title": title,
        "article_number": article_number,
        "source_url": (
            metadata.get("source_url")
            or metadata.get("source")
        ),
    }


# --------------------------------------------------
# Markdown block parser
# --------------------------------------------------

def is_table_line(line):

    return (
        line.strip().startswith("|")
        and line.strip().endswith("|")
    )


def is_table_separator(line):

    return bool(
        re.match(
            r"^\s*\|?[\s:|-]+\|?\s*$",
            line
        )
        and "-" in line
    )


def is_list_line(line):

    return bool(
        re.match(
            r"^\s*(?:[-*+]|\d+[.)])\s+",
            line
        )
    )


def is_heading(line):

    match = re.match(r"^(#{1,6})\s+(.+?)\s*#*\s*$", line)

    if match:
        return len(match.group(1)), match.group(2)

    return None


def parse_markdown_blocks(markdown):

    """
    Split Markdown into structural blocks.

    Each block retains its original line range.
    """

    lines = markdown.splitlines()

    blocks = []

    current = []
    current_type = "paragraph"
    start_line = 1

    def flush(end_line):

        nonlocal current

        if current:

            text = "\n".join(current).strip()

            if text:

                blocks.append({
                    "type": current_type,
                    "text": text,
                    "line_start": start_line,
                    "line_end": end_line,
                })

        current = []

    i = 0

    while i < len(lines):

        line = lines[i]
        line_number = i + 1

        if not line.strip():

            flush(i)
            i += 1
            continue

        # Headings are independent structural elements.
        heading = is_heading(line)

        if heading:

            flush(i)

            blocks.append({
                "type": "heading",
                "level": heading[0],
                "text": line.strip(),
                "line_start": line_number,
                "line_end": line_number,
            })

            i += 1
            continue

        # Fenced code blocks.
        if line.strip().startswith("```"):

            flush(i)

            start_line = line_number
            fence = line.strip()[:3]
            code_lines = [line]

            i += 1

            while i < len(lines):

                code_lines.append(lines[i])

                if lines[i].strip().startswith(fence):
                    break

                i += 1

            blocks.append({
                "type": "code",
                "text": "\n".join(code_lines),
                "line_start": start_line,
                "line_end": i + 1,
            })

            i += 1
            continue

        # Tables.
        if (
            is_table_line(line)
            and i + 1 < len(lines)
            and is_table_separator(lines[i + 1])
        ):

            flush(i)

            start_line = line_number
            table_lines = [line, lines[i + 1]]

            i += 2

            while i < len(lines) and is_table_line(lines[i]):

                table_lines.append(lines[i])
                i += 1

            blocks.append({
                "type": "table",
                "text": "\n".join(table_lines),
                "line_start": start_line,
                "line_end": i,
            })

            continue

        # Lists.
        if is_list_line(line):

            flush(i)

            start_line = line_number
            list_lines = []

            while i < len(lines):

                current_line = lines[i]

                if (
                    not current_line.strip()
                    or is_heading(current_line)
                    or current_line.strip().startswith("```")
                ):
                    break

                list_lines.append(current_line)
                i += 1

            blocks.append({
                "type": "list",
                "text": "\n".join(list_lines),
                "line_start": start_line,
                "line_end": i,
            })

            continue

        # Ordinary paragraph.
        if not current:
            start_line = line_number
            current_type = "paragraph"

        current.append(line)

        i += 1

    flush(len(lines))

    return blocks


# --------------------------------------------------
# Section hierarchy
# --------------------------------------------------

def organize_sections(blocks):

    """
    Associate content with its heading hierarchy.
    """

    sections = []

    heading_stack = []
    current_blocks = []

    def flush():

        nonlocal current_blocks

        if current_blocks:

            sections.append({
                "section_path": list(heading_stack),
                "blocks": current_blocks,
            })

        current_blocks = []

    for block in blocks:

        if block["type"] == "heading":

            level = block["level"]

            flush()

            heading_stack[:] = [
                item for item in heading_stack
                if item["level"] < level
            ]

            heading_stack.append({
                "level": level,
                "title": re.sub(
                    r"^#{1,6}\s+", "",
                    block["text"]
                ),
            })

        else:

            current_blocks.append(block)

    flush()

    return sections


# --------------------------------------------------
# Hyperlink extraction
# --------------------------------------------------

def extract_links(text):

    pattern = r"\[([^\]]+)\]\((https?://[^)]+)\)"

    links = []

    for label, url in re.findall(pattern, text):

        links.append({
            "label": label,
            "url": url,
        })

    # Deduplicate.
    unique = {}

    for link in links:
        unique[link["url"]] = link

    return list(unique.values())


# --------------------------------------------------
# Oversized block splitting
# --------------------------------------------------

def split_table(block):

    """
    Split oversized tables by rows, repeating the header.
    """

    lines = block["text"].splitlines()

    if len(lines) < 3:
        return [block]

    header = lines[:2]
    rows = lines[2:]

    parts = []
    current = header[:]
    part_start = block["line_start"]

    for index, row in enumerate(rows):

        candidate = current + [row]

        if (
            count_tokens("\n".join(candidate)) > MAX_TOKENS
            and len(current) > 2
        ):

            parts.append({
                **block,
                "text": "\n".join(current),
                "line_start": part_start,
            })

            current = header + [row]

            # Approximate original row location.
            part_start = block["line_start"] + index + 2

        else:

            current.append(row)

    if current:
        parts.append({
            **block,
            "text": "\n".join(current),
            "line_start": part_start,
        })

    return parts


def split_large_block(block):

    """
    Fallback for an exceptionally large paragraph,
    list or code block.
    """

    if block["type"] == "table":
        return split_table(block)

    lines = block["text"].splitlines()

    parts = []
    current = []

    for line in lines:

        candidate = "\n".join(current + [line])

        if (
            count_tokens(candidate) > MAX_TOKENS
            and current
        ):

            parts.append({
                **block,
                "text": "\n".join(current),
            })

            current = [line]

        else:

            current.append(line)

    if current:

        parts.append({
            **block,
            "text": "\n".join(current),
        })

    return parts


# --------------------------------------------------
# Chunk construction
# --------------------------------------------------

def create_chunks(sections, doc_metadata):

    chunks = []

    chunk_index = 0

    for section in sections:

        section_path = [
            item["title"]
            for item in section["section_path"]
        ]

        heading_context = "\n".join(
            section_path
        )

        blocks = []

        for block in section["blocks"]:

            if count_tokens(block["text"]) > MAX_TOKENS:

                blocks.extend(
                    split_large_block(block)
                )

            else:

                blocks.append(block)

        current_blocks = []

        def emit_chunk(block_list):

            nonlocal chunk_index

            if not block_list:
                return

            body = "\n\n".join(
                block["text"]
                for block in block_list
            )

            prefix = (
                f"# {doc_metadata['document_title']}\n\n"
                f"{heading_context}\n\n"
                if heading_context else
                f"# {doc_metadata['document_title']}\n\n"
            )

            full_text = prefix + body

            first_line = min(
                block["line_start"]
                for block in block_list
            )

            last_line = max(
                block["line_end"]
                for block in block_list
            )

            chunk_id = (
                f"{doc_metadata['document_id']}_"
                f"{chunk_index:04d}"
            )

            chunks.append({

                "chunk_id": chunk_id,

                **doc_metadata,

                "section": (
                    section_path[-1]
                    if section_path else None
                ),

                "section_path": [
                    doc_metadata["document_title"]
                ] + section_path,

                "chunk_index": chunk_index,

                "line_start": first_line,
                "line_end": last_line,

                "token_count": count_tokens(full_text),

                "source_links": extract_links(body),

                "text": full_text,
            })

            chunk_index += 1

        for block in blocks:

            candidate = current_blocks + [block]

            candidate_text = "\n\n".join(
                item["text"] for item in candidate
            )

            candidate_tokens = count_tokens(
                candidate_text
            )

            if (
                candidate_tokens > TARGET_TOKENS
                and current_blocks
            ):

                previous = current_blocks[:]

                emit_chunk(previous)

                # Carry over the last block if it is small
                # enough to provide useful context.
                overlap = []

                if previous:

                    last = previous[-1]

                    if (
                        count_tokens(last["text"])
                        <= OVERLAP_TOKENS
                    ):
                        overlap = [last]

                current_blocks = overlap

            current_blocks.append(block)

            # A single oversized block can exceed target.
            # It has already been split using MAX_TOKENS.
            if (
                len(current_blocks) == 1
                and count_tokens(
                    current_blocks[0]["text"]
                ) > MAX_TOKENS
            ):

                emit_chunk(current_blocks)
                current_blocks = []

        emit_chunk(current_blocks)

    return chunks


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    all_chunks = []

    files = sorted(INPUT_DIR.glob("*.md"))

    print(f"Found {len(files)} Markdown documents.")

    for index, path in enumerate(files, start=1):

        print(
            f"[{index}/{len(files)}] "
            f"Processing {path.name}"
        )

        metadata, markdown = read_document(path)

        doc_metadata = get_document_metadata(
            path,
            metadata,
        )

        blocks = parse_markdown_blocks(markdown)

        sections = organize_sections(blocks)

        chunks = create_chunks(
            sections,
            doc_metadata,
        )

        all_chunks.extend(chunks)

        print(
            f"  Blocks: {len(blocks)} | "
            f"Chunks: {len(chunks)}"
        )

    with OUTPUT_FILE.open(
        "w",
        encoding="utf-8",
    ) as f:

        for chunk in all_chunks:

            f.write(
                json.dumps(
                    chunk,
                    ensure_ascii=False,
                ) + "\n"
            )

    print("\nChunking completed.")

    print(f"Documents: {len(files)}")
    print(f"Total chunks: {len(all_chunks)}")

    if all_chunks:

        counts = [
            chunk["token_count"]
            for chunk in all_chunks
        ]

        print(
            f"Average tokens: "
            f"{sum(counts) / len(counts):.1f}"
        )

        print(f"Maximum tokens: {max(counts)}")
        print(f"Minimum tokens: {min(counts)}")

    print(f"Output: {OUTPUT_FILE.resolve()}")


if __name__ == "__main__":
    main()
