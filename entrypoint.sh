#!/bin/sh
python3 -m uvicorn wsgi:app --reload --port 8000 --host 0.0.0.0