import json
import csv
import io
import xml.etree.ElementTree as ET
from xml.dom import minidom

# --- PII Placeholders Legend ---
PLACEHOLDERS = {
    "ip": "{{IP_ADDRESS}}",
    "date": "{{DATE}}",
    "time": "{{TIME}}",
    "timestamp": "{{TIMESTAMP_ISO}}",
    "user_agent": "{{USER_AGENT}}",
    "username": "{{USERNAME}}",
    "first_name": "{{FIRST_NAME}}",
    "last_name": "{{LAST_NAME}}",
}

# --- Available contexts for log generation ---
CONTEXTS = {
    "web_server_access": {
        "timestamp": PLACEHOLDERS["timestamp"],
        "client_ip": PLACEHOLDERS["ip"],
        "http_method": "GET",
        "request_path": "/profile/dashboard",
        "status_code": 200,
        "user_agent": PLACEHOLDERS["user_agent"],
        "user": PLACEHOLDERS["username"]
    },
    "auth_system_login": {
        "event_ts": PLACEHOLDERS["timestamp"],
        "event_type": "LOGIN_SUCCESS",
        "source_ip": PLACEHOLDERS["ip"],
        "username": PLACEHOLDERS["username"],
        "description": "User successfully authenticated."
    },
    "user_registration": {
        "registration_date": PLACEHOLDERS["date"],
        "registration_time": PLACEHOLDERS["time"],
        "username": PLACEHOLDERS["username"],
        "first_name": PLACEHOLDERS["first_name"],
        "last_name": PLACEHOLDERS["last_name"],
        "registration_ip": PLACEHOLDERS["ip"]
    },
    "apache_combined_log": {
        # This is a special context for raw-text format
        "format_string": f'{PLACEHOLDERS["ip"]} - {PLACEHOLDERS["username"]} [{PLACEHOLDERS["date"]}:{PLACEHOLDERS["time"]} +0000] "GET /api/v1/user HTTP/1.1" 200 1500 "http://example.com/referrer" "{PLACEHOLDERS["user_agent"]}"'
    },
    "application_error": {
        "log_time": PLACEHOLDERS["timestamp"],
        "severity": "ERROR",
        "module": "payment_processor.py",
        "message": "Transaction failed for user.",
        "context_user": PLACEHOLDERS["username"],
        "client_ip": PLACEHOLDERS["ip"]
    },
    "file_download": {
        "download_ts": PLACEHOLDERS["timestamp"],
        "source_ip": PLACEHOLDERS["ip"],
        "username": PLACEHOLDERS["username"],
        "filename": "confidential_report_q3.pdf",
        "filesize_bytes": 450123
    }
}

# --- Generator Functions ---

def generate_raw_text(context_data):
    """Generates a simple raw text log."""
    # Special case for pre-formatted strings like Apache logs
    if "format_string" in context_data:
        return context_data["format_string"]
        
    log_items = [f"{k}=\"{v}\"" for k, v in context_data.items()]
    return " ".join(log_items)

def generate_csv(context_data):
    """Generates a CSV formatted log."""
    output = io.StringIO()
    header = list(context_data.keys())
    writer = csv.DictWriter(output, fieldnames=header)
    
    writer.writeheader()
    writer.writerow(context_data)
        
    return output.getvalue()

def generate_json(context_data):
    """Generates a JSON formatted log."""
    return json.dumps(context_data, indent=2)

def generate_xml(context_data):
    """Generates an XML formatted log."""
    root = ET.Element("log")
    for key, value in context_data.items():
        field = ET.SubElement(root, key)
        field.text = str(value)

    rough_string = ET.tostring(root, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")


def generate_log_template(log_format, context):
    """Main function to generate log templates."""
    if context not in CONTEXTS:
        return "Error: Invalid context specified."

    context_data = CONTEXTS[context]
    
    if log_format == 'raw-text':
        return generate_raw_text(context_data)
    elif log_format == 'csv':
        return generate_csv(context_data)
    elif log_format == 'json':
        return generate_json(context_data)
    elif log_format == 'xml':
        return generate_xml(context_data)
    else:
        return "Error: Invalid format specified."

# --- Example Usage ---
if __name__ == "__main__":
    print("--- EXAMPLE 1: JSON User Registration Log ---")
    print(generate_log_template(log_format='json', context='user_registration'))

    print("\n--- EXAMPLE 2: Raw-Text Apache Combined Log ---")
    print(generate_log_template(log_format='raw-text', context='apache_combined_log'))
    
    print("\n--- EXAMPLE 3: XML Web Server Access ---")
    print(generate_log_template(log_format='xml', context='web_server_access'))

    print("\n--- EXAMPLE 4: CSV Application Error ---")
    print(generate_log_template(log_format='csv', context='application_error'))
