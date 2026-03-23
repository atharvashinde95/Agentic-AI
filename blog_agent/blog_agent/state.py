from typing import TypedDict, List

class BlogState(TypedDict):
    # User inputs
    topic: str
    keywords: str
    tone: str
    length: str
    num_images: int
    language: str

    # Agent outputs
    outline: dict          # {title, slug, meta, keywords, sections}
    raw_content: str
    seo_content: str
    image_prompts: list    # [{heading, prompt}]
    image_paths: list      # [file_path, ...]
    final_markdown: str
    final_html: str

    # Pipeline metadata
    logs: List[str]
    error: str
