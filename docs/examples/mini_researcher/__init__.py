from dataclasses import dataclass


@dataclass
class RAGDocument:
    title: str
    source: str
    content: str
