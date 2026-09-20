import re
from pathlib import Path

from pydantic import BaseModel

TITLE_RE = re.compile(r"^#\s+(.+?)\s+[—-]\s+Version\s+(\S+)")
HEADING_RE = re.compile(r"^##\s+(\d+)\.\s+(.+)$")
SLUG = "expense-policy"


class PolicySection(BaseModel):
    chunk_id: str
    document: str
    version: str
    section: str
    section_title: str
    text: str


def parse_policy(path: Path) -> list[PolicySection]:
    document: str | None = None
    version: str | None = None
    sections: list[PolicySection] = []
    current: dict | None = None

    def flush(current: dict | None) -> None:
        if current is None: return
        if document is None or version is None:
            raise ValueError(f"Could not parse title/version from {path}")

        text = "\n".join(current["body"]).strip()
        if text:
            sections.append(
                PolicySection(
                    chunk_id=f"{SLUG}:v{version}:section-{current['section']}",
                    document=document,
                    version=version,
                    section=current["section"],
                    section_title=current["section_title"],
                    text=text,
                )
            )

    for line in path.read_text().splitlines():
        if m := TITLE_RE.match(line):
            document, version = m.group(1).strip(), m.group(2).strip()
            continue
        if m := HEADING_RE.match(line):
            flush(current)
            current = {
                "section": m.group(1),
                "section_title": m.group(2).strip(),
                "body": [],
            }
            continue
        if current is not None:
            current["body"].append(line)
    
    flush(current)

    if document is None or version is None:
        raise ValueError(f"Could not parse title/version from {path}")
    if not sections:
        raise ValueError(f"No numbered sections found in {path}")
    return sections


if __name__ == "__main__":
    for section in parse_policy(Path("policy.md")):
        print(section.chunk_id, section.section_title)
        print(section.text)
        print()
