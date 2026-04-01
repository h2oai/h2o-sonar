
# BEGIN: user added these matplotlib lines to ensure any plots do not pop-up in their UI
import matplotlib
matplotlib.use('Agg')  # Set the backend to non-interactive
import matplotlib.pyplot as plt
plt.ioff()
import os
os.environ['TERM'] = 'dumb'
# END: user added these matplotlib lines to ensure any plots do not pop-up in their UI
# filename: check_full_document.py
# execution: true
import os

# Check if the full document context exists
if os.path.exists("document_context.txt"):
    print("Reading full document context...")
    with open("document_context.txt", 'r') as f:
        content = f.read()
        # Look for sections specifically about criminal records
        criminal_record_sections = []
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if "criminal" in line.lower() or "felony" in line.lower() or "conviction" in line.lower() or "disqualif" in line.lower():
                start = max(0, i-5)
                end = min(len(lines), i+10)
                context = "\n".join(lines[start:end])
                criminal_record_sections.append(context)
        
        # Print relevant sections
        if criminal_record_sections:
            print("\n--- Relevant sections about criminal records ---")
            for i, section in enumerate(criminal_record_sections):
                print(f"\nSection {i+1}:")
                print(section)
                print("-"*50)
        else:
            print("No additional sections specifically mentioning criminal records found.")
else:
    print("Full document context file not found.")