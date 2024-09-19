FROM langchain/langchain:latest

RUN apt update && apt-get install -y pulseaudio