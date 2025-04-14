FROM public.ecr.aws/lambda/python:3.12

WORKDIR /layer
COPY requirements.txt .
RUN pip install --no-cache-dir --target ./python/lib/python3.12/site-packages -r requirements.txt


