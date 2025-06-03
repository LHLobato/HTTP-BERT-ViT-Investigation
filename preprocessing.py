from sklearn.model_selection import train_test_split 
from sklearn.feature_extraction.text import TfidfVectorizer
import io
import urllib.parse
import json

normal_file_raw = 'normalTrafficTraining.txt'
anomaly_file_raw = 'anomalousTrafficTest.txt'
normal_file_parse = 'normalRequestTraining2.txt'
anomaly_file_parse = 'anomalousRequestTest2.txt'

def parse_file(file_in, file_out):
    fin = open(file_in, 'r', encoding='utf-8', errors='ignore')
    fout = io.open(file_out, "w", encoding="utf-8")
    lines = fin.readlines()
    res = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith(("GET ", "POST ", "PUT ", "DELETE ", "PATCH ")):
            method, path, _ = line.split(' ', 2)
            url_parts = urllib.parse.urlparse(path)
            query = url_parts.query
            
            # Process headers and body for POST/PUT/PATCH
            body = ""
            content_type = ""
            if method in ["POST", "PUT", "PATCH"]:
                while i + 1 < len(lines) and not lines[i + 1].startswith("Content-Length:"):
                    if lines[i + 1].startswith("Content-Type:"):
                        content_type = lines[i + 1].split(":", 1)[1].strip().lower()
                    i += 1
                
                if i + 1 < len(lines) and lines[i + 1].startswith("Content-Length:"):
                    content_length = int(lines[i + 1].split(":", 1)[1].strip())
                    i += 2
                    # Skip empty lines until body
                    while i < len(lines) and lines[i].strip() == "":
                        i += 1
                    if i < len(lines):
                        body = lines[i].strip()
                        # For JSON, keep raw body
                        if "application/json" in content_type:
                            try:
                                body_json = json.loads(body)
                                body = json.dumps(body_json, separators=(',', ':'))  # Minify JSON
                            except json.JSONDecodeError:
                                pass  # Keep original if invalid JSON
                        elif "form-data" not in content_type:  # Skip multipart (complex)
                            body = urllib.parse.unquote_plus(body)
            
            # Rebuild the normalized request
            normalized_path = urllib.parse.quote(url_parts.path)  # Re-encode path (optional)
            if body and "application/json" not in content_type:
                query = f"{query}&{body}" if query else body
            query = urllib.parse.unquote_plus(query)  # Decode but keep & intact
            
            # Combine method, path, and query
            request = f"{method.lower()}{normalized_path}"
            if query:
                request += f"?{query}"
            res.append(request)
        i += 1
    
    # Write results
    for line in res:
        fout.write(line + '\n')
    print(f"Finished parsing {len(res)} requests from {file_in}")
    fin.close()
    fout.close()

def parse_file2(file_in, file_out):
    fin = open(file_in)
    fout = io.open(file_out, "w", encoding="utf-8")
    lines = fin.readlines()
    res = []
    for i in range(len(lines)):
        line = lines[i].strip()
        if line.startswith("GET"):
            res.append("GET" + line.split(" ")[1])
        elif line.startswith("POST") or line.startswith("PUT"):
            url = line.split(' ')[0] + line.split(' ')[1]
            j = 1
            while True:
                if lines[i + j].startswith("Content-Length"):
                    break
                j += 1
            j += 1
            data = lines[i + j + 1].strip()
            url += '?' + data
            res.append(url)
    for line in res:
        line = urllib.parse.unquote(line).replace('\n','').lower()
        fout.writelines(line + '\n')
    print ("finished parse ",len(res)," requests")
    fout.close()
    fin.close()

parse_file(normal_file_raw,normal_file_parse)
parse_file(anomaly_file_raw,anomaly_file_parse)
