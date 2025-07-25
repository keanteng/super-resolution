export PANDOC_PATH="/C/Users/Khor Kean Teng/Downloads/pandoc-3.7.0.2-windows-x86_64/pandoc-3.7.0.2"
export PATH="$PATH:$PANDOC_PATH"

cd "slides-2"

# Generate PDF with LaTeX for A4 paper size
pandoc "presentation-version-2.md" -o "presentation-version-2.pdf" \
  --pdf-engine=pdflatex \
  -V geometry:a4paper \
  -V geometry:margin=0.5in \
  -V mainfont="Arial" \
  --include-in-header=watermark.tex