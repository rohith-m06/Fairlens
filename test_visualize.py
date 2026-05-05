import requests
import json

# First upload data
with open('data/test_upload.csv', 'rb') as f:
    files = {'file': f}
    response = requests.post('http://127.0.0.1:8000/api/audit/upload', files=files)
    
audit_data = response.json()
print("Upload response keys:", audit_data.keys())

# Now test visualizations
viz_response = requests.post('http://127.0.0.1:8000/api/visualize', 
                             json=audit_data,
                             headers={'Content-Type': 'application/json'})

print("\nVisualize response status:", viz_response.status_code)
result = viz_response.json()
print("Visualize response keys:", result.keys())
print("Charts:", list(result.get('charts', {}).keys()))
print("Errors:", result.get('errors'))

if result.get('charts'):
    for chart_name, chart_json in result['charts'].items():
        print(f"\n{chart_name}: {len(chart_json)} characters")
        try:
            chart_data = json.loads(chart_json)
            print(f"  - Valid JSON, keys: {list(chart_data.keys())}")
        except Exception as e:
            print(f"  - Invalid JSON: {e}")
