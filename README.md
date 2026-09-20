# project-business
An Android based AI powered inventory management system for small shops

## Setup

    pip install -r requirements.txt
    # Debian/Ubuntu also needs the tesseract binary:
    sudo apt-get install tesseract-ocr

## Run

    python app.py             # scan-architecture debug shell (camera + pipeline internals)
    python app_consumer.py    # consumer shop app (Analytics / Inventory / AI Chat / Options)
