FROM python:3.10-slim

RUN apt-get update && apt-get install gcc g++ git make poppler-utils tesseract-ocr tesseract-ocr-jpn -y && apt-get clean \
	&& rm -rf /var/lib/apt/lists/*
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
	PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

COPY --chown=user . $HOME/app

RUN pip install langflow>==0.0.86 -U --user
RUN pip install packaging==21.3 -U --user
RUN pip install pdf2image==1.16.0 -U --user
RUN pip install Pillow==9.1.1 --user
RUN pip install pyparsing==3.0.9 -U --user
RUN pip install pytesseract==0.3.9 -U --user
RUN pip install PyPDF2
RUN pip install kor

CMD ["python", "-m", "langflow", "--host", "0.0.0.0", "--port", "7860"]
