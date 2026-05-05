import requests

# Test the CSV upload
with open('data/test_upload.csv', 'rb') as f:
    files = {'file': f}
    response = requests.post('http://127.0.0.1:8000/api/audit/upload', files=files)
    
print(f'Status: {response.status_code}')
if response.status_code == 200:
    data = response.json()
    print('✅ CSV uploaded successfully!')
    print(f'Violations found: {len(data.get("violations", []))}')
    print(f'Risk level: {data.get("risk_level", "N/A")}')
else:
    print(f'❌ Error: {response.text}')
