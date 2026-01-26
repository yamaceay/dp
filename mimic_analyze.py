import re
from pathlib import Path
from dp.loaders.mimic import MIMICDatasetAdapter

# def find_all_sections(text: str) -> dict:
#     section_pattern = r"([A-Z]+(?:\s+[A-Z]+)*):\s*(.*?)(?=\n\s*(?:[A-Z]+(?:\s+[A-Z]+)*):|$)"
#     sections = {}
#     for match in re.finditer(section_pattern, text, re.DOTALL | re.IGNORECASE):
#         section_name = match.group(1).strip()
#         section_content = match.group(2).strip()
#         sections[section_name] = section_content
#     return sections

def bound_sections(text: str, sep: str = "\n\n") -> dict[str, tuple[int, int, str]]:
    parts = text.split(sep)
    sections: dict[str, tuple[int, int, str]] = {}
    current_offset = 0
    
    for i, part in enumerate(parts):
        part_start = text.find(part, current_offset)
        part_end = part_start + len(part)
        current_offset = part_end
        
        match = re.match(r"^([A-Z]+(?:\s+[A-Z]+)*):\s*", part, re.IGNORECASE)
        if match:
            section_name = match.group(1).strip()
            content = part
            content_len = len(content)
            
            if content_len >= 150:
                sections[section_name] = (part_start, part_end, content)
            else:
                if sections:
                    last_section = list(sections.keys())[-1]
                    prev_start, _, prev_content = sections[last_section]
                    sections[last_section] = (prev_start, part_end, text[prev_start:part_end])
                else:
                    sections[section_name] = (part_start, part_end, content)
            continue
        
        if sections and len(part.strip()) >= 150:
            last_section = list(sections.keys())[-1]
            prev_start, _, _ = sections[last_section]
            sections[last_section] = (prev_start, part_end, text[prev_start:part_end])
        elif part.strip():
            if "UNKNOWN" in sections:
                prev_start, _, _ = sections["UNKNOWN"]
                sections["UNKNOWN"] = (prev_start, part_end, text[prev_start:part_end])
            else:
                sections["UNKNOWN"] = (part_start, part_end, part)

    return sections


# def extract_section(section_name: str, text: str) -> str:
#     section_pattern = rf"{re.escape(section_name)}:\s*(.*?)(?=\n\s*(?:[A-Z]+(?:\s+[A-Z]+)*):|$)"
    
#     match = re.search(section_pattern, text, re.DOTALL | re.IGNORECASE)
#     if match:
#         return match.group(1).strip()
#     return ""

if __name__ == "__main__":
    data_in = Path("data/mimic/splitted/test.csv")
    adapter = MIMICDatasetAdapter(data_in=str(data_in), max_records=10)

    for record in adapter.iter_records():
        text = record.text
        category = record.metadata.get("category")
        if category == "Discharge summary":
            sections = bound_sections(text)
            for section, (start, end, content) in sections.items():
                length = end - start
                print(f"Section: {section}\nLength: {length}\nContent: {content}\n{'-'*40}\n")
        break