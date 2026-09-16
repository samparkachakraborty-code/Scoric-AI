import argparse
import os
import re
from google import genai
from google.genai import types
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors

def extract_id(filepath: str) -> str:
    """Extracts the numeric ID from the filename (e.g., as968850.pdf -> 968850)."""
    filename = os.path.basename(filepath)
    match = re.search(r'as(\d+)\.pdf$', filename, re.IGNORECASE)
    if not match:
        raise ValueError(f"Invalid filename format: '{filename}'. Expected format like 'as968850.pdf'.")
    return match.group(1)

def grade_answer_script(question_paper_path: str, answer_script_path: str, mode: str) -> str:
    """Sends the Question Paper and Answer Script to Gemini for grading based on evaluation mode."""
    client = genai.Client()

    # Upload PDF files to Gemini API
    print("[1/3] Uploading Question Paper and Answer Script to Gemini...")
    qp_file = client.files.upload(file=question_paper_path)
    as_file = client.files.upload(file=answer_script_path)

    # Prompt engineering based on grading mode
    mode_instructions = {
        "hard": "Strict grading. Deduct marks for minor logical gaps, lack of detail, or missing key terms.",
        "medium": "Balanced grading. Give standard partial credit for partially correct answers and clear intent.",
        "low": "Lenient grading. Focus on core concepts and attempt effort. Give full credit if the main idea is present."
    }

    prompt = f"""
    You are an automated university exam evaluator.
    
    Grading Rigor Mode: {mode.upper()} - {mode_instructions.get(mode.lower(), mode_instructions['medium'])}

    Tasks:
    1. Extract all questions and their assigned total marks from the Question Paper.
    2. Read and analyze the handwritten Answer Script.
    3. For every question:
       - Transcribe/summarize the student's answer.
       - Assign marks obtained vs maximum marks.
       - Provide specific feedback highlighting correct parts and errors/omissions.
    4. Provide a Final Summary including Total Marks Obtained, Percentage, and Overall Remarks.

    Format the output cleanly using Markdown headers (## Question X) and bold text for easy parsing.
    """

    print(f"[2/3] Evaluating answer script (Mode: {mode.upper()})...")
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[qp_file, as_file, prompt]
    )

    # Cleanup remote files
    client.files.delete(name=qp_file.name)
    client.files.delete(name=as_file.name)

    return response.text

def generate_pdf_report(evaluation_text: str, script_id: str, output_path: str):
    """Generates a structured final result PDF report."""
    print(f"[3/3] Generating final report PDF at: {output_path}")
    doc = SimpleDocTemplate(output_path, pagesize=letter)
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        'TitleStyle',
        parent=styles['Heading1'],
        fontSize=18,
        textColor=colors.HexColor('#1A365D'),
        spaceAfter=12
    )
    
    body_style = ParagraphStyle(
        'BodyStyle',
        parent=styles['Normal'],
        fontSize=10,
        leading=14,
        spaceAfter=8
    )

    story = []
    story.append(Paragraph(f"<b>Final Result Report - Script ID: {script_id}</b>", title_style))
    story.append(Spacer(1, 12))

    # Process markdown lines into ReportLab Paragraphs
    lines = evaluation_text.split('\n')
    for line in lines:
        clean_line = line.strip()
        if not clean_line:
            continue
        
        # Format headings
        if clean_line.startswith('## '):
            p = Paragraph(f"<b>{clean_line[3:]}</b>", styles['Heading2'])
        elif clean_line.startswith('# '):
            p = Paragraph(f"<b>{clean_line[2:]}</b>", styles['Heading1'])
        else:
            # Simple markdown bold convert
            formatted = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', clean_line)
            p = Paragraph(formatted, body_style)
        
        story.append(p)

    doc.build(story)

def main():
    parser = argparse.ArgumentParser(description="Smart Teacher AI - Automated Answer Script Evaluation")
    parser.add_argument("--qp", required=True, help="Path to Question Paper PDF")
    parser.add_argument("--as", required=True, dest="ans_script", help="Path to Handwritten Answer Script PDF (e.g. as968850.pdf)")
    parser.add_argument("--mode", choices=["hard", "medium", "low"], default="medium", help="Grading strictness mode")
    
    args = parser.parse_args()

    try:
        script_id = extract_id(args.ans_script)
        
        # Run Evaluation
        result_text = grade_answer_script(args.qp, args.ans_script, args.mode)
        
        # Save Output PDF
        output_filename = f"fr{script_id}.pdf"
        output_dir = os.path.join(os.environ['USERPROFILE'], 'OneDrive', 'Documents')
        os.makedirs(output_dir, exist_ok=True)
        output_filepath = os.path.join(output_dir, output_filename)
        
        generate_pdf_report(result_text, script_id, output_filepath)
        
        print(f"\nSuccessfully evaluated! Final report saved to: {output_filepath}")

    except Exception as e:
        print(f"\nError: {e}")

if __name__ == "__main__":
    main()