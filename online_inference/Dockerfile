FROM python:3.8-slim
COPY requirements.txt ./requirements.txt
RUN pip install -r requirements.txt

RUN mkdir -p /models
COPY models/model.pickle /models/model.pickle

COPY app.py /app.py

RUN mkdir -p /src/common
COPY src/common/__init__.py /src/common/__init__.py
COPY src/common/timing.py /src/common/timing.py

RUN mkdir -p /src/requests
COPY src/requests/make_request.py /src/requests/make_request

WORKDIR .

ENV PROJECT_ROOT .
ENV PYTHONPATH .

RUN mkdir ./logs

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]