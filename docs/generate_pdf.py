import os
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

def generate_pdf():
    pdf_path = os.path.join(os.path.dirname(__file__), "Fresher_GenAI_Interview_Questions.pdf")
    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=letter,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40
    )

    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=20,
        leading=24,
        textColor=colors.HexColor('#1E293B'),
        spaceAfter=15
    )

    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=10,
        leading=14,
        textColor=colors.HexColor('#64748B'),
        spaceAfter=20
    )

    q_style = ParagraphStyle(
        'QuestionStyle',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=12,
        leading=16,
        textColor=colors.HexColor('#0F172A'),
        spaceBefore=12,
        spaceAfter=6
    )

    ans_style = ParagraphStyle(
        'AnswerStyle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9.5,
        leading=13.5,
        textColor=colors.HexColor('#334155'),
        spaceAfter=6
    )

    story = []
    
    story.append(Paragraph("PersonalAI — 30 Fresher GenAI Interview Questions & Answers", title_style))
    story.append(Paragraph("A comprehensive technical preparation guide covering RAG, FAISS, Embeddings, FastAPI, SQLite, and Transformer concepts.", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#E2E8F0'), spaceAfter=15))

    # Read markdown interview questions file
    md_path = os.path.join(os.path.dirname(__file__), "interview_questions.md")
    with open(md_path, "r", encoding="utf-8") as f:
        content = f.read()

    blocks = content.split("---")
    for block in blocks:
        lines = [line.strip() for line in block.strip().split("\n") if line.strip()]
        if not lines:
            continue
        
        for line in lines:
            if line.startswith("### Q"):
                clean_q = line.replace("### ", "").replace("**", "")
                story.append(Paragraph(clean_q, q_style))
            elif line.startswith("- **Simple Answer**:") or line.startswith("Simple Answer:"):
                clean_ans = line.replace("- **Simple Answer**:", "<b>Simple Answer:</b>").replace("Simple Answer:", "<b>Simple Answer:</b>")
                story.append(Paragraph(clean_ans, ans_style))
            elif line.startswith("- **Technical Explanation**:") or line.startswith("Technical Explanation:"):
                clean_ans = line.replace("- **Technical Explanation**:", "<b>Technical Explanation:</b>").replace("Technical Explanation:", "<b>Technical Explanation:</b>")
                story.append(Paragraph(clean_ans, ans_style))
            elif line.startswith("- **Small Example**:") or line.startswith("Small Example:"):
                clean_ans = line.replace("- **Small Example**:", "<b>Small Example:</b>").replace("Small Example:", "<b>Small Example:</b>")
                story.append(Paragraph(clean_ans, ans_style))

    doc.build(story)
    print(f"Successfully generated PDF at {pdf_path}")

if __name__ == "__main__":
    generate_pdf()
