from sagemaker.tensorflow import TensorFlowModel
from sagemaker import get_execution_role, Session

role = get_execution_role()
sagemaker_session = Session()

model = TensorFlowModel(
    model_data='s3://brain-model-h5/brain_tumor_model.h5',
    role=role,
    entry_point='inference.py',        # ✅ This points to the local file
    framework_version='2.10',
    py_version='py39',
    sagemaker_session=sagemaker_session
)

predictor = model.deploy(
    instance_type='ml.m5.large',
    endpoint_name='brain-tumor-endpoint'
)
