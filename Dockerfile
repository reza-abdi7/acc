FROM python:3.12-slim-bookworm

RUN mkdir -p app
COPY requirements-app.txt /app/requirements.txt
COPY ./.streamlit/ /app/.streamlit/
COPY ./ui /app/ui
COPY ./src /app/src
WORKDIR /app

RUN pip3 install cython
RUN pip3 install -r requirements.txt --no-cache-dir
RUN pip3 install uvicorn arq asyncpg

RUN useradd -ms /bin/bash appuser
USER appuser
