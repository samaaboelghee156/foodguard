# FoodGuard AI Backend

FastAPI backend for real food image classification. The default model is [`nateraw/food`](https://huggingface.co/nateraw/food), a ViT model fine-tuned on Food-101.

## Setup

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Android emulator default mobile URL:

```bash
flutter run --dart-define=FOOD_ANALYSIS_API_URL=http://10.0.2.2:8000
```

Physical Android device on the same Wi-Fi:

```bash
flutter run --dart-define=FOOD_ANALYSIS_API_URL=http://YOUR_COMPUTER_LAN_IP:8000
```

For APKs shared with other phones, `10.0.2.2` will not work. Use a public server URL, a LAN IP reachable from the phone, or a tunneling URL such as ngrok, then save that URL in the app Profile screen.

The model downloads from Hugging Face on first startup. For production, deploy this backend behind HTTPS and cache the model in the container image.
