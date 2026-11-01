"""
DSC5001 Project: Generate Appendix PDF with all source code
"""

import os
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Preformatted
from reportlab.lib.colors import HexColor

SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'output'))

# Source files in order
SOURCE_FILES = [
    ('01_download_data.py', 'Step 1: Data Download'),
    ('02_feature_engineering.py', 'Step 2: Data Processing & Feature Engineering'),
    ('03_model_training.py', 'Step 3: Model Training & Evaluation'),
    ('04_visualizations.py', 'Step 4: Visualization Generation'),
    ('05_generate_slides.py', 'Step 5: Slide Generation'),
]

output_path = os.path.join(OUTPUT_DIR, 'DSC5001_Project_Appendix.pdf')

doc = SimpleDocTemplate(
    output_path,
    pagesize=A4,
    leftMargin=1.5*cm,
    rightMargin=1.5*cm,
    topMargin=2*cm,
    bottomMargin=2*cm,
)

styles = getSampleStyleSheet()

# Custom styles
title_style = ParagraphStyle(
    'CustomTitle',
    parent=styles['Title'],
    fontSize=18,
    textColor=HexColor('#0D47A1'),
    spaceAfter=12,
    alignment=1,  # Center
)

subtitle_style = ParagraphStyle(
    'CustomSubtitle',
    parent=styles['Normal'],
    fontSize=12,
    textColor=HexColor('#666666'),
    spaceAfter=6,
    alignment=1,
)

section_style = ParagraphStyle(
    'SectionHeader',
    parent=styles['Heading1'],
    fontSize=14,
    textColor=HexColor('#0D47A1'),
    spaceBefore=20,
    spaceAfter=10,
)

code_style = ParagraphStyle(
    'CodeStyle',
    parent=styles['Code'],
    fontSize=7,
    textColor=HexColor('#212121'),
    backColor=HexColor('#F5F7FA'),
    leftIndent=10,
    rightIndent=10,
    spaceBefore=5,
    spaceAfter=5,
    fontName='Courier',
)

story = []

# Title page
story.append(Spacer(1, 3*cm))
story.append(Paragraph("DSC5001 Project Appendix", title_style))
story.append(Paragraph("Source Codes", subtitle_style))
story.append(Spacer(1, 1*cm))
story.append(Paragraph("Hong Kong AQHI High-Risk Hour Early Warning", 
             ParagraphStyle('ProjTitle', parent=styles['Normal'], fontSize=14, 
                          textColor=HexColor('#333333'), alignment=1, spaceAfter=6)))
story.append(Paragraph("A Comparative Study of Classification Models on Hourly Monitoring Data",
             ParagraphStyle('ProjSub', parent=styles['Normal'], fontSize=11, 
                          textColor=HexColor('#666666'), alignment=1, spaceAfter=20)))
story.append(Paragraph("Team-XX", 
             ParagraphStyle('Team', parent=styles['Normal'], fontSize=12, 
                          textColor=HexColor('#333333'), alignment=1, spaceAfter=6)))

# File listing
story.append(Spacer(1, 1*cm))
story.append(Paragraph("Contents:", section_style))
for i, (filename, title) in enumerate(SOURCE_FILES, 1):
    story.append(Paragraph(f"{i}. {title} ({filename})", 
                 ParagraphStyle('FileItem', parent=styles['Normal'], fontSize=11, 
                              leftIndent=20, spaceAfter=4)))

story.append(PageBreak())

# Add each source file
for filename, title in SOURCE_FILES:
    filepath = os.path.join(SRC_DIR, filename)
    
    story.append(Paragraph(title, section_style))
    story.append(Paragraph(f"File: {filename}", 
                 ParagraphStyle('FileLabel', parent=styles['Normal'], fontSize=10, 
                              textColor=HexColor('#666666'), spaceAfter=8)))
    
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            code = f.read()
        
        # Split code into chunks that fit on pages
        lines = code.split('\n')
        current_chunk = []
        chunk_size = 55  # lines per page
        
        for i, line in enumerate(lines):
            current_chunk.append(line)
            if len(current_chunk) >= chunk_size or i == len(lines) - 1:
                code_text = '\n'.join(current_chunk)
                # Escape special characters for reportlab
                code_text = code_text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                story.append(Preformatted(code_text, code_style))
                current_chunk = []
                if i < len(lines) - 1:
                    story.append(Spacer(1, 0.3*cm))
    
    story.append(PageBreak())

doc.build(story)
print(f"Appendix PDF saved to: {output_path}")
