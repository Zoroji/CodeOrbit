
from pathlib import Path
import re

# Find the transcripts folder, or use the current folder
BASE_DIR = Path(__file__).parent
TRANSCRIPT_DIR = BASE_DIR / "transcripts"

if not TRANSCRIPT_DIR.exists():
    TRANSCRIPT_DIR = BASE_DIR


def convert_vtt_to_srt(vtt_path: Path) -> Path:
    content = vtt_path.read_text(
        encoding="utf-8-sig",
        errors="replace",
    )

    blocks = re.split(r"\n\s*\n", content.strip())
    srt_blocks = []

    for block in blocks:
        lines = block.strip().splitlines()

        # Find the subtitle timestamp line
        time_index = next(
            (i for i, line in enumerate(lines) if "-->" in line),
            None,
        )

        if time_index is None:
            continue

        match = re.match(
            r"\s*((?:\d{2}:)?\d{2}:\d{2}\.\d{3})"
            r"\s*-->\s*"
            r"((?:\d{2}:)?\d{2}:\d{2}\.\d{3})",
            lines[time_index],
        )

        if not match:
            continue

        start, end = match.groups()

        # Normalize timestamps to SRT format
        if start.count(":") == 1:
            start = "00:" + start
        if end.count(":") == 1:
            end = "00:" + end

        start = start.replace(".", ",")
        end = end.replace(".", ",")

        # Keep subtitle text and remove VTT markup
        text = "\n".join(lines[time_index + 1:])
        text = re.sub(r"<[^>]+>", "", text)
        text = text.strip()

        if not text:
            continue

        srt_blocks.append(
            f"{len(srt_blocks) + 1}\n"
            f"{start} --> {end}\n"
            f"{text}"
        )

    output_path = vtt_path.with_suffix(".srt")
    output_path.write_text(
        "\n\n".join(srt_blocks) + "\n",
        encoding="utf-8",
    )

    return output_path


# Convert the standard English subtitle files only
files = list(TRANSCRIPT_DIR.glob("*.en.vtt"))

if not files:
    print(f"No *.en.vtt files found in: {TRANSCRIPT_DIR}")
else:
    for file in files:
        output = convert_vtt_to_srt(file)
        print(f"Converted: {file.name} -> {output.name}")
