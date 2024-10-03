import sagemaker
import boto3
from sagemaker.huggingface import HuggingFaceModel

# Role and credentials
try:
    role = "*****************"
except ValueError:
    iam = boto3.client('iam')
    role = iam.get_role(RoleName='sagemaker_execution_role')['Role']['Arn']
    
aws_access_key_id = '***************'
aws_secret_access_key = '***************************'
aws_region = '******************'

# Hugging Face Model Environment Variables
hub = {
    'HF_MODEL_ID': '************',
    'HF_TASK': 'automatic-speech-recognition',
    'HF_API_TOKEN': "****************",
    'MMS_JOB_QUEUE_SIZE': "400",
}

# Batching Configuration for TensorFlow Serving
batching_config = '''
{
  "batch_size": 4,
  "max_batch_size": 16,  # Increased batch size to improve GPU utilization
  "batch_timeout_micros": 100000,  # 100 ms (adjust based on your latency requirements)
  "max_enqueued_batches": 50,
  "num_batch_threads": 8
}
'''

# Model Environment Configurations
environment = {
    'SAGEMAKER_MULTI_MODEL': 'true',  # Enable multi-model on this endpoint
    'SAGEMAKER_BATCH': 'true',        # Enable batching
    'MAX_BATCH_SIZE': '16',           # Configure max batch size (adjust as needed)
    'BATCH_TIMEOUT': '100',           # Timeout for collecting a batch (in ms)
    # 'SAGEMAKER_MODEL_SERVER_WORKERS': '4',        # Use 4 workers for handling requests
    # 'MAX_BATCH_SIZE': '16',                       # Process up to 16 requests per batch
    # 'SAGEMAKER_MULTI_MODEL': 'true',
    # 'SAGEMAKER_GUNICORN_WORKERS': '9',            # 9 Gunicorn workers for handling requests in parallel
    # 'OMP_NUM_THREADS': '4',                       # Use 4 threads for parallel processing
    # 'SAGEMAKER_TFS_ENABLE_BATCHING': 'true',      # Enable TensorFlow Serving batching
    # 'SAGEMAKER_TFS_BATCHING_CONFIG': batching_config,  # Apply batching config
    # 'SAGEMAKER_TFS_INSTANCE_COUNT': '1',          # Keep single instance for now
    # 'SAGEMAKER_TFS_INTER_OP_PARALLELISM': '1',    # Inter-op parallelism
    # 'SAGEMAKER_TFS_INTRA_OP_PARALLELISM': '1',    # Intra-op parallelism
}

# Hugging Face Model Definition
huggingface_model = HuggingFaceModel(
    transformers_version='4.37.0',  # Ensure compatible version with SageMaker
    pytorch_version='2.1.0',        # Ensure compatible PyTorch version
    py_version='py310',             # Use compatible Python version
    env=hub,                        # Use environment variables from the hub
    role=role,                      # IAM Role
    sagemaker_session=sagemaker.Session(boto_session=boto3.Session(
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        region_name=aws_region
    ))
)

# Custom serializers and deserializers for audio and text
from sagemaker.serializers import DataSerializer
from sagemaker.deserializers import StringDeserializer

class Utf8Deserializer(StringDeserializer):
    def deserialize(self, stream, content_type):
        return stream.read().decode('utf-8')

# Set serializer for audio input and deserializer for text output
wav_serializer = DataSerializer(content_type='audio/x-audio')
utf8_deserializer = Utf8Deserializer()

# Endpoint Name
new_endpoint_name = 'doccy-ml-g5-2xlarge'

# Deploy the model with batching enabled
predictor = huggingface_model.deploy(
    initial_instance_count=1,
    instance_type='ml.g5.2xlarge',  # GPU instance for ASR model
    endpoint_name=new_endpoint_name,
    serializer=wav_serializer,
    deserializer=utf8_deserializer,
    environment=environment  # Pass environment for batching and concurrency
)