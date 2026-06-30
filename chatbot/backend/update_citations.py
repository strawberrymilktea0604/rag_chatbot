import csv
import re

def update_citations_in_csv(input_file, output_file):
    """
    Updates citation numbers in gold_answer to match the first page_hint value.
    
    Example:
    - Input: gold_answer="...answer[cite: 1]..." page_hint="90|104|114"
    - Output: gold_answer="...answer[cite: 90]..." page_hint="90|104|114"
    """
    
    rows = []
    citations_updated = 0
    
    with open(input_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        
        for row in reader:
            page_hint = row.get('page_hint', '').strip()
            gold_answer = row.get('gold_answer', '').strip()
            
            if page_hint and gold_answer:
                # Extract first page_hint value
                first_page = page_hint.split('|')[0]
                
                # Replace all [cite: N] with [cite: first_page]
                updated_answer = re.sub(
                    r'\[cite:\s*\d+\]',
                    f'[cite: {first_page}]',
                    gold_answer
                )
                
                if updated_answer != gold_answer:
                    citations_updated += 1
                    row['gold_answer'] = updated_answer
            
            rows.append(row)
    
    # Write updated CSV
    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"✓ Updated {citations_updated} citation(s)")
    print(f"✓ Output saved to: {output_file}")

if __name__ == '__main__':
    input_file = 'd:\\rag_chatbot\\chatbot\\backend\\qa_gold.csv'
    output_file = 'd:\\rag_chatbot\\chatbot\\backend\\qa_gold_updated.csv'
    
    update_citations_in_csv(input_file, output_file)
    
    