FROM svizor/zoomcamp-model:mlops-3.10.0-slim 

WORKDIR /app

COPY [ "starter.py", "requirements.txt",  "./" ]

RUN pip install -r requirements.txt

ENTRYPOINT ["python", "starter.py"]



