FROM rasa/rasa-pro:3.17.3

# Install as root only for system packages / pip; runtime drops privileges
USER root

COPY --chown=1001:1001 custom-component custom-component
COPY --chown=1001:1001 credentials.yml credentials.yml

RUN apt-get update --fix-missing \
    && apt-get install -y --no-install-recommends \
        build-essential \
        python3-dev \
    && pip install --no-cache-dir -r custom-component/requirements.txt \
    && apt-get purge -y --auto-remove build-essential python3-dev \
    && rm -rf /var/lib/apt/lists/* /tmp/* /root/.cache/pip

# Match base image: do not run the container as root
USER 1001
