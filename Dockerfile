# पायथन 3.11 इमेज
FROM python:3.11-slim

# FFmpeg और ज़रूरी टूल्स इंस्टॉल करो
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libmagic-dev \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# वर्किंग डायरेक्टरी
WORKDIR /app

# पहले requirements.txt कॉपी करो ताकि कैश का इस्तेमाल हो सके
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# अब सारा कोड कॉपी करो
COPY . .

# पोर्ट 8000 ओपन करो
EXPOSE 8000

# सर्वर चलाने का सही कमांड (main.py बाहर है इसलिए main:app)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
