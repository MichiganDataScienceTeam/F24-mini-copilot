FROM public.ecr.aws/lambda/python:3.12

# Copy requirements.txt
COPY lambda_requirements.txt ${LAMBDA_TASK_ROOT}

# Install the specified packages
RUN pip install --compile --no-cache-dir -r lambda_requirements.txt

# Copy model
COPY trained/gpt2-728 ${LAMBDA_TASK_ROOT}/trained/gpt2-728

# Copy function code
COPY lambda.py ${LAMBDA_TASK_ROOT}

# Set the CMD to your handler
CMD [ "lambda.handler" ]
