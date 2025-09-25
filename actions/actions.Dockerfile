# specify the version of rasa-sdk image
ARG VERSION=3.12.1

# start from rasa-sdk image
FROM rasa/rasa-sdk:${VERSION}

# Copy the actions folder, endpoints.yml, db json files
COPY ./actions /app/actions
# COPY ./endpoints.yml /app/endpoints.yml
COPY ./db /app/db


USER root
# install required packages
RUN pip install -U pip
RUN pip install --no-cache-dir -r /app/actions/requirements.txt

# Expose the port the app runs on
EXPOSE 5055

# Command to run when the container starts
USER 1001
CMD ["start", "--actions", "actions"]