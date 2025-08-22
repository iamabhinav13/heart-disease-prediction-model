from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from azure.ai.ml.entities import Model, ManagedOnlineEndpoint, ManagedOnlineDeployment, CodeConfiguration, Environment

# --- 1. Connect to your Azure ML Workspace ---
# You can find these details on the overview page of your Azure ML Workspace
subscription_id = "a5974998-4722-4b26-85f0-7357d23440bb"
resource_group = "heart-disease-rg1"
workspace_name = "heart-disease-ws"

ml_client = MLClient(
    DefaultAzureCredential(), subscription_id, resource_group, workspace_name
)

# --- 2. Register the Model ---
# This uploads your .pkl file and creates a model asset in Azure
print("Registering the model...")
model_name = "heart-disease-rf-model"
model_path = "heart_disease_model.pkl"

model = Model(
    path=model_path,
    name=model_name,
    description="Random Forest model to predict heart disease.",
    type="custom_model" # or "mlflow_model", "triton_model"
)
registered_model = ml_client.models.create_or_update(model)
print(f"Model '{registered_model.name}' version {registered_model.version} registered.")

# --- 3. Define the Online Endpoint ---
# This is the stable URL where your model will be hosted.
endpoint_name = "heart-disease-endpoint" 

print(f"Creating endpoint '{endpoint_name}'...")
endpoint = ManagedOnlineEndpoint(
    name=endpoint_name,
    description="Online endpoint for heart disease prediction.",
    auth_mode="key"
)
# This command can take a few minutes
ml_client.online_endpoints.begin_create_or_update(endpoint).result()
print("Endpoint created successfully.")

# --- 4. Define the Deployment ---
# This configures the compute resources and links your model/code.
blue_deployment = ManagedOnlineDeployment(
    name="blue",
    endpoint_name=endpoint_name,
    model=registered_model,
    environment=Environment(
        name="rf-classifier-env",
        conda_file="conda.yml",
        image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04"
    ),
    code_configuration=CodeConfiguration(
        code="./",  # The directory containing score.py
        scoring_script="score.py"
    ),
    instance_type="Standard_DS2_v2",
    instance_count=1,
)

print("Creating the deployment...")
# This command can take 10-20 minutes as it provisions hardware
ml_client.online_deployments.begin_create_or_update(blue_deployment).result()
print("Deployment created successfully.")

# --- 5. Allocate Traffic to the Deployment ---
# Direct 100% of traffic to our new deployment
endpoint.traffic = {"blue": 100}
ml_client.online_endpoints.begin_create_or_update(endpoint).result()
print("Traffic allocated. Deployment is now live and ready to be tested.")