# Set the base image
FROM python:3.10-slim-buster

# Set the working directory
WORKDIR /airpoll

RUN apt-get update && apt-get install libgomp1 && apt-get install -y cron && touch /var/log/cron.log

# Install the required packages
RUN pip install --upgrade pip && \
    pip install lightgbm~=3.3.5 numpy~=1.24.2 pandas~=2.0.0 matplotlib~=3.7.1 \
                seaborn~=0.12.2 requests~=2.28.2 pathspec fastapi~=0.95.0 pydantic~=1.10.7 \
                uvicorn PyYAML scipy==1.10.1 scikit-learn==1.2.2 progress~=1.6 \
                optuna==3.1.1 onnxmltools~=1.11.2 skl2onnx onnx onnxruntime

# Copy the local code to the container
COPY ./data_loaders /airpoll/data_loaders
COPY ./data_preprocessing /airpoll/data_preprocessing
COPY ./datasets /airpoll/datasets
COPY ./deployment/airpol_update_data.sh /airpoll/
COPY ./deployment/params.yaml /airpoll/
COPY ./experiments_results /airpoll/experiments_results
COPY ./model_tune_helpers /airpoll/model_tune_helpers
COPY ./settings /airpoll/settings
COPY ./web_service /airpoll/web_service
COPY  *.py /airpoll/

COPY ./deployment/data-upd-night-scheduler /etc/cron.d/

# Give execution rights on the script file
RUN chmod +x /airpoll/airpol_update_data.sh
# Give execution rights on the cron job
RUN chmod 0644 /etc/cron.d/data-upd-night-scheduler
# Apply cron job
RUN crontab /etc/cron.d/data-upd-night-scheduler


# Set the entrypoint command
#CMD /airpoll/airpol_update_data.sh >> /var/log/cron.log && cat /var/log/cron.log && \
# python3 -m uvicorn web_service.aqi_predictor_service:app --host 0.0.0.0 --port 8041 & bash

CMD /airpoll/airpol_update_data.sh && \
 python3 -m uvicorn web_service.aqi_predictor_service:app --host 0.0.0.0 --port 8041 & bash
