
import json
import re

from pathlib import Path
import tiktoken


# --------------------------------------------------
# Configuration
# --------------------------------------------------

INPUT_FILE = Path(
    "data/odigos_dioikitikon_diadikasion/chunks/chunks.jsonl"
)

OUTPUT_FILE = Path(
    "data/odigos_dioikitikon_diadikasion/chunks/segments.jsonl"
)

MAX_TOKENS = 700
OVERLAP_TOKENS = 80

ENCODING = tiktoken.get_encoding("cl100k_base")


def count_tokens(text):
    return len(ENCODING.encode(text))


# --------------------------------------------------
# Find a safe character boundary
# --------------------------------------------------

def largest_fitting_end(text, start, max_tokens):
    """
    Find the furthest character position whose substring
    fits within max_tokens.

    Binary search operates on character positions,
    so Unicode characters are never split.
    """

    low = start + 1
    high = len(text)

    best = start

    while low <= high:

        mid = (low + high) // 2

        tokens = count_tokens(text[start:mid])

        if tokens <= max_tokens:

            best = mid
            low = mid + 1

        else:

            high = mid - 1

    return best


def find_overlap_start(text, previous_end, overlap_tokens):
    """
    Find the earliest character position that provides
    approximately overlap_tokens of overlap.
    """

    low = 0
    high = previous_end

    best = previous_end

    while low <= high:

        mid = (low + high) // 2

        tokens = count_tokens(
            text[mid:previous_end]
        )

        if tokens <= overlap_tokens:

            best = mid
            high = mid - 1

        else:

            low = mid + 1

    return best


# --------------------------------------------------
# Prefer natural text boundaries
# --------------------------------------------------

def improve_boundary(text, start, end, min_position):
    """
    Move the segment end backwards to a natural boundary
    when this doesn't sacrifice too much content.

    Priority:
        1. Paragraph boundary
        2. Line boundary
        3. Sentence boundary
        4. Word boundary
        5. Original boundary
    """

    minimum = max(
        start + 1,
        end - max(1, (end - start) // 5)
    )

    minimum = max(minimum, min_position)

    candidates = [
        text.rfind("\n\n", minimum, end),
        text.rfind("\n", minimum, end),
        text.rfind(". ", minimum, end),
        text.rfind(" ", minimum, end),
    ]

    for position in candidates:

        if position > start:

            candidate_end = position

            if (
                count_tokens(text[start:candidate_end])
                <= MAX_TOKENS
            ):

                return candidate_end

    return end


# --------------------------------------------------
# Split text into overlapping segments
# --------------------------------------------------

def split_text(text):

    text = text.strip()

    if not text:
        return []

    if count_tokens(text) <= MAX_TOKENS:
        return [text]

    segments = []

    start = 0
    text_length = len(text)

    while start < text_length:

        end = largest_fitting_end(
            text,
            start,
            MAX_TOKENS,
        )

        if end <= start:
            raise ValueError(
                "Unable to create a valid segment."
            )

        # Avoid unnecessary mid-paragraph splitting.
        end = improve_boundary(
            text,
            start,
            end,
            start + 1,
        )

        segment = text[start:end].strip()

        if segment:
            segments.append(segment)

        if end >= text_length:
            break

        # Determine where the next segment begins.
        next_start = find_overlap_start(
            text,
            end,
            OVERLAP_TOKENS,
        )

        # Guarantee forward progress.
        if next_start <= start:
            next_start = end

        start = next_start

    return segments


# --------------------------------------------------
# Process existing chunks
# --------------------------------------------------

def process_chunk(chunk):

    original_text = chunk["text"]

    parts = split_text(original_text)

    results = []

    for index, part in enumerate(parts):

        segment = dict(chunk)

        parent_id = chunk["chunk_id"]

        segment["parent_chunk_id"] = parent_id

        segment["segment_id"] = (
            f"{parent_id}_seg_{index:03d}"
        )

        segment["segment_index"] = index

        segment["segment_count"] = len(parts)

        segment["token_count"] = count_tokens(part)

        segment["text"] = part

        results.append(segment)

    return results


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    total_chunks = 0
    total_segments = 0
    oversized_chunks = 0

    all_counts = []

    with INPUT_FILE.open(
        "r",
        encoding="utf-8",
    ) as infile, OUTPUT_FILE.open(
        "w",
        encoding="utf-8",
    ) as outfile:

        for line in infile:

            if not line.strip():
                continue

            chunk = json.loads(line)

            total_chunks += 1

            original_tokens = count_tokens(
                chunk["text"]
            )

            if original_tokens > MAX_TOKENS:
                oversized_chunks += 1

            segments = process_chunk(chunk)

            for segment in segments:

                token_count = segment["token_count"]

                if token_count > MAX_TOKENS:

                    raise ValueError(
                        f"Segment exceeds maximum: "
                        f"{segment['segment_id']} "
                        f"({token_count} tokens)"
                    )

                outfile.write(
                    json.dumps(
                        segment,
                        ensure_ascii=False,
                    ) + "\n"
                )

                total_segments += 1
                all_counts.append(token_count)

    print("\nSegmentation completed.")

    print(f"Original chunks: {total_chunks}")

    print(f"Oversized chunks: {oversized_chunks}")

    print(f"Final segments: {total_segments}")

    if all_counts:

        print(
            f"Average tokens: "
            f"{sum(all_counts) / len(all_counts):.1f}"
        )

        print(f"Maximum tokens: {max(all_counts)}")

        print(f"Minimum tokens: {min(all_counts)}")

    print(f"Output: {OUTPUT_FILE.resolve()}")


if __name__ == "__main__":
    main()
