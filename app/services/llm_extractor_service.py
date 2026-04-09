"""
LLM-based content extraction service.

Sends the PDF file directly to a vision-capable LLM in a single API call
to extract rich, structured text — especially useful for visual documents
like pitch decks where PyPDF2 extracts little to no useful text.

Enable via env var: LLM_EXTRACTION_ENABLED=true
"""

import base64
import logging
from typing import Optional

from openai import OpenAI

from app.configurations.config import OPENAI_API_KEY, LLM_EXTRACTION_ENABLED, LLM_EXTRACTION_MODEL, LLM_EXTRACTION_MAX_PAGES

logger = logging.getLogger(__name__)

EXTRACTION_PROMPT = """You are a document content extractor. Extract ALL text and visual content from this PDF into clean, structured text.

Rules:
- Extract every piece of text you can see, including titles, bullet points, labels, captions, and small print.
- Separate each page with a clear marker: [Page 1], [Page 2], etc.
- For charts, graphs, or tables: describe the data, axes, values, and key takeaways.
- For images or icons: briefly describe what they represent in context.
- Preserve the logical structure: use headings, bullet points, and paragraphs.
- If text is in Spanish, keep it in Spanish. If in English, keep it in English.
- Do NOT add commentary, opinions, or interpretation. Only extract what is there.
- Output clean text, no markdown formatting."""


class LLMExtractorService:
    """Extracts text from PDFs by sending the file directly to a vision-capable LLM."""

    def __init__(self):
        self.enabled = LLM_EXTRACTION_ENABLED
        self.model = LLM_EXTRACTION_MODEL
        self.max_pages = LLM_EXTRACTION_MAX_PAGES
        self.client = OpenAI(api_key=OPENAI_API_KEY) if self.enabled else None

    def is_enabled(self) -> bool:
        return self.enabled

    def extract_from_url(self, file_url: str) -> Optional[str]:
        """Extract text from a PDF URL in a single LLM call.

        Preferred method — sends the URL directly to the model, no local processing needed.
        Returns extracted text or None if extraction fails.
        """
        if not self.enabled or not self.client:
            return None

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": EXTRACTION_PROMPT},
                            {
                                "type": "file",
                                "file": {
                                    "url": file_url,
                                },
                            },
                        ],
                    }
                ],
                max_completion_tokens=16384,
                temperature=0,
            )
            text = response.choices[0].message.content
            if text and text.strip():
                return text.strip()
            return None
        except Exception as e:
            logger.warning("LLM extraction from URL failed: %s — will try local file", e)
            return None

    def extract_from_file(self, file_path: str) -> Optional[str]:
        """Extract text from a local PDF file by sending it as base64 in a single LLM call.

        Fallback method — used when URL extraction fails or file is local.
        Returns extracted text or None if extraction fails.
        """
        if not self.enabled or not self.client:
            return None

        try:
            with open(file_path, "rb") as f:
                b64_pdf = base64.b64encode(f.read()).decode("utf-8")

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": EXTRACTION_PROMPT},
                            {
                                "type": "file",
                                "file": {
                                    "filename": "document.pdf",
                                    "file_data": f"data:application/pdf;base64,{b64_pdf}",
                                },
                            },
                        ],
                    }
                ],
                max_completion_tokens=16384,
                temperature=0,
            )
            text = response.choices[0].message.content
            if text and text.strip():
                return text.strip()
            return None
        except Exception as e:
            logger.error("LLM extraction from file failed for %s: %s", file_path, e)
            return None
