import os
import pandas as pd
from typing import Dict, Any, List, Tuple
from backend.app.services.nlp_service import NLPService

class DocumentService:
    @staticmethod
    def extract_text_from_file(file_path: str, file_type: str) -> str:
        """
        Extract raw text content from PDF, DOCX, TXT, or Markdown files.
        """
        file_type = file_type.lower().strip(".")
        
        if file_type == "pdf":
            return DocumentService._extract_pdf(file_path)
        elif file_type in ["docx", "doc"]:
            return DocumentService._extract_docx(file_path)
        elif file_type in ["txt", "md", "markdown"]:
            return DocumentService._extract_text_file(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_type}")

    @staticmethod
    def _extract_pdf(file_path: str) -> str:
        text = ""
        try:
            import pypdf
            reader = pypdf.PdfReader(file_path)
            for page_idx, page in enumerate(reader.pages):
                extracted = page.extract_text()
                if extracted:
                    text += f"\n--- Page {page_idx + 1} ---\n" + extracted
        except Exception as e:
            # Fallback to pdfplumber or plain reading if needed
            try:
                import pdfplumber
                with pdfplumber.open(file_path) as pdf:
                    for page_idx, page in enumerate(pdf.pages):
                        extracted = page.extract_text()
                        if extracted:
                            text += f"\n--- Page {page_idx + 1} ---\n" + extracted
            except Exception as inner_e:
                raise RuntimeError(f"Failed to extract text from PDF: {str(e)} | Fallback error: {str(inner_e)}")
        return text

    @staticmethod
    def _extract_docx(file_path: str) -> str:
        try:
            import docx
            doc = docx.Document(file_path)
            paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
            return "\n".join(paragraphs)
        except Exception as e:
            raise RuntimeError(f"Failed to extract text from DOCX: {str(e)}")

    @staticmethod
    def _extract_text_file(file_path: str) -> str:
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
        except Exception as e:
            raise RuntimeError(f"Failed to read text file: {str(e)}")

    @staticmethod
    def generate_pandas_document_analytics(documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Uses Pandas to compute tabular document analytics & summaries.
        Demonstrates practical usage of Pandas DataFrames for file metrics.
        """
        if not documents:
            return {
                "total_documents": 0,
                "total_chunks": 0,
                "total_characters": 0,
                "total_words": 0,
                "total_tokens": 0,
                "average_chunk_length": 0.0,
                "file_type_distribution": {}
            }

        df = pd.DataFrame(documents)
        
        # Ensure necessary columns exist
        for col in ["chunk_count", "char_count", "word_count", "token_count", "file_type"]:
            if col not in df.columns:
                df[col] = 0

        total_docs = len(df)
        total_chunks = int(df["chunk_count"].sum())
        total_chars = int(df["char_count"].sum())
        total_words = int(df["word_count"].sum())
        total_tokens = int(df["token_count"].sum())
        
        avg_chunk_length = float(total_chars / total_chunks) if total_chunks > 0 else 0.0
        
        file_type_counts = df["file_type"].value_counts().to_dict()

        return {
            "total_documents": total_docs,
            "total_chunks": total_chunks,
            "total_characters": total_chars,
            "total_words": total_words,
            "total_tokens": total_tokens,
            "average_chunk_length": round(avg_chunk_length, 2),
            "file_type_distribution": file_type_counts
        }
