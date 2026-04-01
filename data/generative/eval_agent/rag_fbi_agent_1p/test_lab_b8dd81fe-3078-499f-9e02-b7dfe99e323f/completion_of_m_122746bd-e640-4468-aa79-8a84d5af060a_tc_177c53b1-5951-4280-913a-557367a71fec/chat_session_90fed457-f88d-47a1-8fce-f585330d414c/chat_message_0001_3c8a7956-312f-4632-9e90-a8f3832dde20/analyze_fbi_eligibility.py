
# BEGIN: user added these matplotlib lines to ensure any plots do not pop-up in their UI
import matplotlib
matplotlib.use('Agg')  # Set the backend to non-interactive
import matplotlib.pyplot as plt
plt.ioff()
import os
os.environ['TERM'] = 'dumb'
# END: user added these matplotlib lines to ensure any plots do not pop-up in their UI
# filename: analyze_fbi_eligibility.py
# execution: true
from api_server.agent_tools.ask_question_about_documents import ask_question_about_documents

query = "Can a candidate apply for any FBI position if they have a criminal record? What are the specific rules regarding criminal history for FBI applicants?"
csv_answer, json_answer, rag_answer, advanced_rag_answers = ask_question_about_documents(query=query, baseline=True)