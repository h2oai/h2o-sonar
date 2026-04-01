
# BEGIN: user added these matplotlib lines to ensure any plots do not pop-up in their UI
import matplotlib
matplotlib.use('Agg')  # Set the backend to non-interactive
import matplotlib.pyplot as plt
plt.ioff()
import os
os.environ['TERM'] = 'dumb'
# END: user added these matplotlib lines to ensure any plots do not pop-up in their UI
# filename: read_fbi_guide_chunks.py
# execution: true
# Let's read the document chunks to get more detailed information
import os

# Read the first set of document chunks
chunks = [
    "Guides_FBI_Jobs_Eligibility_Guide_2022_page_1_chunk_0.txt",
    "Guides_FBI_Jobs_Eligibility_Guide_2022_page_1_chunk_1.txt",
    "Guides_FBI_Jobs_Eligibility_Guide_2022_page_2_chunk_0.txt",
    "Guides_FBI_Jobs_Eligibility_Guide_2022_page_2_chunk_1.txt"
]

for chunk_file in chunks:
    if os.path.exists(chunk_file):
        print(f"\n--- Content of {chunk_file} ---")
        with open(chunk_file, 'r') as f:
            content = f.read()
            print(content)
    else:
        print(f"File {chunk_file} does not exist")