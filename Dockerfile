FROM joshualevy44/trace_app:latest

COPY . /trace_app/
RUN cd /trace_app && pip install . --force-reinstall --no-deps && cd -

WORKDIR /workdir