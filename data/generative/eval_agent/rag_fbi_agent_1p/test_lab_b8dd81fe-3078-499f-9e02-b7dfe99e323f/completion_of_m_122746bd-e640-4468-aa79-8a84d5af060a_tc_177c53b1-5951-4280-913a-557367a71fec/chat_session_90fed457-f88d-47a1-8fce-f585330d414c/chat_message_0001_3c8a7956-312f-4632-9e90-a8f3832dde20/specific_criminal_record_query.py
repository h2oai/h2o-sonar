
# BEGIN: user added these matplotlib lines to ensure any plots do not pop-up in their UI
import matplotlib
matplotlib.use('Agg')  # Set the backend to non-interactive
import matplotlib.pyplot as plt
plt.ioff()
import os
os.environ['TERM'] = 'dumb'
# END: user added these matplotlib lines to ensure any plots do not pop-up in their UI
# filename: specific_criminal_record_query.py
# execution: true
from api_server.agent_tools.ask_question_about_documents import ask_question_about_documents

query = """
Based on the FBI Jobs Eligibility Guide:
1. Is there any distinction between types of felonies that might disqualify candidates?
2. Are there any exceptions to the felony conviction disqualifier?
3. For non-special agent positions, what specific criminal convictions are disqualifying?
4. Is there a timeframe after which a criminal conviction might no longer be disqualifying?
"""

csv_answer, json_answer, rag_answer, advanced_rag_answers = ask_question_about_documents(query=query, files=["Guides_FBI_Jobs_Eligibility_Guide_2022_page_1_chunk_0.txt", "Guides_FBI_Jobs_Eligibility_Guide_2022_page_1_chunk_1.txt", "Guides_FBI_Jobs_Eligibility_Guide_2022_page_2_chunk_0.txt", "Guides_FBI_Jobs_Eligibility_Guide_2022_page_2_chunk_1.txt"])