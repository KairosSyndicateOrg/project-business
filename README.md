# project-business
An Android based AI powered inventory management system for small shops

## Setup

    pip install -r requirements.txt
    # On Windows, install tesseract OCR manually and add to PATH. Restart terminal before continuing
    # Debian/Ubuntu also needs the tesseract binary:
    sudo apt-get install tesseract-ocr

## Run

    python app.py             # scan-architecture debug shell (camera + pipeline internals)
    python app_consumer.py    # consumer shop app (Analytics / Inventory / AI Chat / Options)
