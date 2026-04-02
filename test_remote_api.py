import requests
import base64
import os
from io import BytesIO
from PIL import Image

def test_remote_server():
    # 1. Define the remote server URL
    vision_url = "https://vision-server-qvem3ril2q-uc.a.run.app/embed_media"
    
    # 2. Get API key from ai_engine/.env
    api_key = "eyekoidhnkldhlihijYYDhhshSycsHDhhUYykjskwehcjnh"
    
    print(f"Testing the Live Server Endpoint: {vision_url}")
    
    # 3. Create a clean mock image natively and convert to base64
    img = Image.new('RGB', (224, 224), color='#f43f5e')
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    img_b64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    # 4. Craft the correct Payload format using image_base64 array
    payload = {
        "image_base64": [img_b64]
    }
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    try:
        # 5. Execute 
        print("Executing POST request...")
        response = requests.post(vision_url, json=payload, headers=headers, timeout=60)
        
        # 6. Evaluate
        if response.status_code == 200:
            data = response.json()
            synth_score = data.get("synthetic_prob", [])
            print("\n✅ API Success! Code 200")
            if len(synth_score) > 0:
                 print(f"Server Derived Synthetic Probability: {synth_score[0]*100:.2f}%")
            else:
                 print("Zero predictions returned (empty array)")
        else:
            print(f"\n❌ SERVER ERROR: {response.status_code}")
            print(response.text)
            
    except Exception as e:
        print(f"\n❌ CONNECTION FAILED: {e}")

if __name__ == "__main__":
    test_remote_server()
