from PyPDF2 import PdfReader
from Ramesses_rag_stone import preprocess_text  

def inspect_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    print(f"=== PDF STRUCTURE ===")
    print(f"Pages: {len(reader.pages)}\n")

    #critical pages
    for page_num in [0, 5, -1]:
        text = reader.pages[page_num].extract_text()
        print(f"--- Page {page_num} (Original) ---\n{text[:300]}...\n")
        print(f"--- Page {page_num} (Cleaned) ---\n{preprocess_text(text[:300])}...\n\n")

if __name__ == "__main__":
    inspect_pdf("the_life_and_times_of_Ramesses_II.pdf")