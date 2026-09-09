import fitz
import re

from config import CHUNK_SIZE, CHUNK_OVERLAP


def extract_text_from_pdf(pdf_bytes):

    document = fitz.open(stream=pdf_bytes, filetype="pdf")

    pages = []

    for page_number, page in enumerate(document):

        text = page.get_text("text")

        if text.strip():

            pages.append({
                "page": page_number + 1,
                "text": text
            })

    document.close()

    return pages


def clean_text(text):

    text = re.sub(r"\s+", " ", text)

    text = re.sub(r"-\s+", "", text)

    return text.strip()


def create_chunks(pages):

    chunks = []

    for page in pages:

        text = clean_text(page["text"])

        words = text.split()

        start = 0

        while start < len(words):

            end = start + CHUNK_SIZE

            chunk_words = words[start:end]

            chunk = " ".join(chunk_words)

            if chunk.strip():

                chunks.append({
                    "text": chunk,
                    "page": page["page"]
                })

            start += CHUNK_SIZE - CHUNK_OVERLAP

    return chunks
