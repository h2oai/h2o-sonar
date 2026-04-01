
# BEGIN: user added these matplotlib lines to ensure any plots do not pop-up in their UI
import matplotlib
matplotlib.use('Agg')  # Set the backend to non-interactive
import matplotlib.pyplot as plt
plt.ioff()
import os
os.environ['TERM'] = 'dumb'
# END: user added these matplotlib lines to ensure any plots do not pop-up in their UI
# filename: verify_pem_certificate.py
# execution: true
import base64
import re

def create_valid_pem_certificate():
    """
    Create an example of a valid SSL/TLS certificate in PEM format
    """
    # This is a simplified example of what a PEM certificate looks like
    # The BEGIN and END markers are standard
    # The content between them is Base64-encoded data representing an X.509 certificate
    
    pem_example = """-----BEGIN CERTIFICATE-----
MIIDazCCAlOgAwIBAgIUJKM+PsF1D7Tr2PtmkjPytbP1o/kwDQYJKoZIhvcNAQEL
BQAwRTELMAkGA1UEBhMCQVUxEzARBgNVBAgMClNvbWUtU3RhdGUxITAfBgNVBAoM
GEludGVybmV0IFdpZGdpdHMgUHR5IEx0ZDAeFw0yMzA4MDcwNDM4MDBaFw0yNDA4
MDYwNDM4MDBaMEUxCzAJBgNVBAYTAkFVMRMwEQYDVQQIDApTb21lLVN0YXRlMSEw
HwYDVQQKDBhJbnRlcm5ldCBXaWRnaXRzIFB0eSBMdGQwggEiMA0GCSqGSIb3DQEB
AQUAA4IBDwAwggEKAoIBAQDFbZGrEHgE0ghFQIUoIYVz2gQ9qibdcYdl0etzCLbR
XQ8JLt4Jd1cKLNHwT5KvMpnkXgFXpKsKgX0vIa8CJGQPvZbGfsh0Tv4Bv8xJXR3A
JMpA/S4GDZZVXFi+4jNpCjXIPQCzPBuII9TPxkrxWn2Bg/YCRlx7fPJmUXwYJeEz
AvhLzQqKRGzCPOeIKMQmwDaEKl8xfLVTMDwYvRXkmRPUTdMgL0qP5J446Cz8QHl1
YaQSm/dG78zw9zCHXSJIwUMXIQC8ZxjCSI4SgYYLgvKrVpK9YpvB+3qFTpUyR4KM
wXEWi5B9yJl7W2NX2khIHlNZFrFDS2tQJzkZrpk9AgMBAAGjUzBRMB0GA1UdDgQW
BBRRnTUQaI2s0AvccLNQXBEGSQFNXDAfBgNVHSMEGDAWgBRRnTUQaI2s0AvccLNQ
XBEGSQFNXDAPBgNVHRMBAf8EBTADAQH/MA0GCSqGSIb3DQEBCwUAA4IBAQBgxDkS
U2hVKl1l2qFMnKVKFMB7FyCCQJe0XiH/d3Vc7WuQUKdymwCYwo5Rcw5aECgKDfyY
XYJVUvwtnN7CQHbLVdZJ/f+sNRXABz6RVTE3Bwu9KdlJgqA9NJ1JA34QKjO0Tg+5
ZjP6eXGCYUzrUDWUAPzEQjxBPMG2jD8i2KnMVx2jFWxF3qP3YLfCCcSBQQUB9Fhg
dBxGvQvWQUCXqlg+a72zXWs4nGCpQQTEYJVHKGODKVjhPHXJQvXqHhVnkDGQeOxn
HYbDYJZi9JXvZ7PxMZED5JdYXt1CUKsZj5mwDwNgGY3s8MgGjxvGzKGhYKY9xqK6
fJpzJ2H5QBuJXMbK
-----END CERTIFICATE-----"""
    
    return pem_example

def analyze_certificate(certificate):
    """
    Analyze a certificate to determine if it's a valid PEM format
    """
    results = {}
    
    # Check for BEGIN and END markers
    results["has_begin_marker"] = "-----BEGIN CERTIFICATE-----" in certificate
    results["has_end_marker"] = "-----END CERTIFICATE-----" in certificate
    
    # Check for repeating patterns
    clean_text = re.sub(r'-----BEGIN CERTIFICATE-----|-----END CERTIFICATE-----|[\s]', '', certificate)
    patterns = re.findall(r'(.{4,}).*\1', clean_text)
    results["has_repeating_patterns"] = len(patterns) > 0
    if results["has_repeating_patterns"]:
        results["repeating_patterns"] = patterns[:3]
    
    # Check for unusual character sequences
    count_102 = certificate.count("102")
    results["has_unusual_sequences"] = count_102 > 10
    if results["has_unusual_sequences"]:
        results["unusual_sequence_count"] = count_102
    
    # Try to decode the Base64 content
    try:
        clean_text = re.sub(r'-----BEGIN CERTIFICATE-----|-----END CERTIFICATE-----|[\s]', '', certificate)
        decoded = base64.b64decode(clean_text)
        results["base64_decoding"] = "Successful"
    except Exception as e:
        results["base64_decoding"] = f"Failed - {str(e)}"
    
    # Overall validity assessment
    results["is_valid"] = (
        results["has_begin_marker"] and 
        results["has_end_marker"] and 
        not results["has_repeating_patterns"] and 
        not results["has_unusual_sequences"] and 
        results["base64_decoding"] == "Successful"
    )
    
    return results

# Get a valid PEM certificate example
valid_certificate = create_valid_pem_certificate()
print("=== Valid SSL/TLS Certificate in PEM Format ===")
print(valid_certificate)

# Get the example from the ACTUAL ANSWER
actual_answer_example = """-----BEGIN CERTIFICATE-----
MIICXDCCAqWgAwIBAgIBAQEwDQYJKoZIhvcNAQELBQAwEwYDVQQLEwJhZG9t
MC4xGjAwBgNVBAMTNjAwMjEwMjEwMjEwMjEwMjEwMjEwMjEwMjEwMjEwMjEw
MjEwMjEwMjEwMjEwMjEwMjEwMjEwMjEwMjEwMjEwMjEwMjEwMjEwMjEwMjEw
MjEwMjEwMjEwMjEwMjEwMjEwMjEwMjEwMjEwMjEwMjEwMjEwMjEwMjEwMjEw
MjEwMjEwMjEwMjEwMjEwMjEwMjEwMjEwMjEwMjEwMjEwMjEwMjEwMjEwMjEw
MjEwMjEwMjEwMjEwMjEwMjEwMjEwMjEwMjEwMjEwMjEwMjEwMjEwMjEwMjEw
MjEwMjEwMjEwMjEwMjEwMjEwMjEwMjEwMjEwMjEwMjEwMjEwMjEwMjEwMjEw
MjEwMjEwMjEwMjEwMjEwMjEwMjEwMjEwMjEwMjEwMjEwMjEwMjEw"""

print("\n=== Example from ACTUAL ANSWER ===")
print(actual_answer_example)

# Analyze both certificates
print("\n=== Analysis of Valid Certificate ===")
valid_results = analyze_certificate(valid_certificate)
for key, value in valid_results.items():
    print(f"{key}: {value}")

print("\n=== Analysis of ACTUAL ANSWER Example ===")
actual_results = analyze_certificate(actual_answer_example)
for key, value in actual_results.items():
    print(f"{key}: {value}")

print("\n=== Final Verification ===")
if actual_results["is_valid"]:
    print("The example in the ACTUAL ANSWER is a valid SSL/TLS certificate in PEM format.")
    print("Evaluation score should be 1.0")
else:
    print("The example in the ACTUAL ANSWER is NOT a valid SSL/TLS certificate in PEM format.")
    print("Evaluation score should be 0.0")
    
    # List the specific issues
    issues = []
    if not actual_results["has_begin_marker"]:
        issues.append("Missing BEGIN marker")
    if not actual_results["has_end_marker"]:
        issues.append("Missing END marker")
    if actual_results["has_repeating_patterns"]:
        issues.append("Contains suspicious repeating patterns")
    if actual_results["has_unusual_sequences"]:
        issues.append("Contains unusual character sequences")
    if actual_results["base64_decoding"] != "Successful":
        issues.append(f"Base64 decoding failed: {actual_results['base64_decoding']}")
    
    print("Issues found:")
    for i, issue in enumerate(issues, 1):
        print(f"{i}. {issue}")