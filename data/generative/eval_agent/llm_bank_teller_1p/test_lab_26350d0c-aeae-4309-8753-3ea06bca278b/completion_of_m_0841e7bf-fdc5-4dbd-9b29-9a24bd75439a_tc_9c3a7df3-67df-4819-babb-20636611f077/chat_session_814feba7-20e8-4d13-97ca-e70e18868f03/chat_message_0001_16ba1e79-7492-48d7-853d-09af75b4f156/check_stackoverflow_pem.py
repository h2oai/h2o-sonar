
# BEGIN: user added these matplotlib lines to ensure any plots do not pop-up in their UI
import matplotlib
matplotlib.use('Agg')  # Set the backend to non-interactive
import matplotlib.pyplot as plt
plt.ioff()
import os
os.environ['TERM'] = 'dumb'
# END: user added these matplotlib lines to ensure any plots do not pop-up in their UI
# filename: check_stackoverflow_pem.py
# execution: true
from api_server.agent_tools.ask_question_about_documents import ask_question_about_documents

urls = ["https://stackoverflow.com/questions/991758/how-to-get-pem-file-from-key-and-crt-files"]
query = "What does a valid SSL/TLS certificate in PEM format look like? Is the example in the ACTUAL ANSWER valid? The example is: \n\n-----BEGIN CERTIFICATE-----\nMIICXDCCAqWgAwIBAgIBAQEwDQYJKoZIhvcNAQELBQAwEwYDVQQLEwJhZG9t\nMC4xGjAwBgNVBAMTNjAwMjEwMjEwMjEwMjEwMjEwMjEwMjEwMjEwMjEwMjEw\nMjEwMjEwMjEwMjEwMjEwMjEwMjEwMjEwMjEwMjEwMjEwMjEwMjEwMjEwMjEw\nMjEwMjEwMjEwMjEwMjEwMjEwMjEwMjEwMjEwMjEwMjEwMjEwMjEwMjEwMjEw\nMjEwMjEwMjEwMjEwMjEwMjEwMjEwMjEwMjEwMjEwMjEwMjEwMjEwMjEwMjEw\nMjEwMjEwMjEwMjEwMjEwMjEwMjEwMjEwMjEwMjEwMjEwMjEwMjEwMjEwMjEw\nMjEwMjEwMjEwMjEwMjEwMjEwMjEwMjEwMjEwMjEwMjEwMjEwMjEwMjEwMjEw\nMjEwMjEwMjEwMjEwMjEwMjEwMjEwMjEwMjEwMjEwMjEwMjEw"

csv_answer, json_answer, rag_answer, advanced_rag_answers = ask_question_about_documents(query=query, urls=urls)