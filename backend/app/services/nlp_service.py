import re
from typing import List, Dict, Any

try:
    import tiktoken
    encoding = tiktoken.get_encoding("cl100k_base")
except ImportError:
    encoding = None

class NLPService:
    @staticmethod
    def clean_text(text: str) -> str:
        """
        Normalize whitespace, strip control characters, and clean raw text.
        """
        if not text:
            return ""
        # Replace multiple newlines and carriage returns with clean spacing
        text = re.sub(r'\r\n|\r', '\n', text)
        # Replace non-printable characters
        text = re.sub(r'[^\x09\x0A\x0D\x20-\x7E\xA0-\xFF]', '', text)
        # Consolidate multiple spaces
        text = re.sub(r'[ \t]+', ' ', text)
        return text.strip()

    @staticmethod
    def count_characters(text: str) -> int:
        return len(text)

    @staticmethod
    def count_words(text: str) -> int:
        if not text.strip():
            return 0
        return len(re.findall(r'\b\w+\b', text))

    @staticmethod
    def count_tokens(text: str) -> int:
        """
        Count approximate or exact tokens using tiktoken (cl100k_base) or fallback word multiplier.
        """
        if not text:
            return 0
        if encoding:
            try:
                return len(encoding.encode(text))
            except Exception:
                pass
        # Fallback estimation: average ~4 characters or ~0.75 words per token
        words = NLPService.count_words(text)
        return int(words * 1.3) if words > 0 else 0

    @staticmethod
    def chunk_text(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> List[Dict[str, Any]]:
        """
        Splits clean text into overlapping chunks.
        Returns a list of dicts containing chunk_id, content, char_count, word_count, and token_count.
        """
        clean = NLPService.clean_text(text)
        if not clean:
            return []

        if len(clean) <= chunk_size:
            return [{
                "chunk_id": 0,
                "content": clean,
                "char_count": NLPService.count_characters(clean),
                "word_count": NLPService.count_words(clean),
                "token_count": NLPService.count_tokens(clean),
                "start_char": 0,
                "end_char": len(clean)
            }]

        chunks = []
        step = chunk_size - chunk_overlap
        if step <= 0:
            step = chunk_size // 2

        chunk_idx = 0
        start = 0
        text_length = len(clean)

        while start < text_length:
            end = min(start + chunk_size, text_length)
            
            # If not at the end of the text, try to split at a newline or sentence end
            if end < text_length:
                break_point = max(
                    clean.rfind('\n', start, end),
                    clean.rfind('. ', start, end)
                )
                if break_point > start + (chunk_size // 2):
                    end = break_point + 1

            chunk_content = clean[start:end].strip()
            if chunk_content:
                chunks.append({
                    "chunk_id": chunk_idx,
                    "content": chunk_content,
                    "char_count": NLPService.count_characters(chunk_content),
                    "word_count": NLPService.count_words(chunk_content),
                    "token_count": NLPService.count_tokens(chunk_content),
                    "start_char": start,
                    "end_char": end
                })
                chunk_idx += 1

            start += step

        return chunks
