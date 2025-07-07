"""Example of a pipeline to demonstrate a simple real world data science workflow."""

import os
import json
import kfp.compiler
from dotenv import load_dotenv
from kfp import dsl
from kfp.dsl import component, Input, Artifact

load_dotenv(override=True)

kubeflow_endpoint = os.environ["KUBEFLOW_ENDPOINT"]
base_image = os.getenv("BASE_IMAGE", "")
data_science_image = os.getenv("DATA_SCIENCE_IMAGE", "")

@dsl.component(
    base_image=base_image,
    packages_to_install=["pandas", "scikit-learn"], 
)
def data_prep(
    x_train_file: dsl.Output[dsl.Dataset],
    x_test_file: dsl.Output[dsl.Dataset],
    y_train_file: dsl.Output[dsl.Dataset],
    y_test_file: dsl.Output[dsl.Dataset],
):
    import pickle

    import pandas as pd
    from sklearn import datasets
    from sklearn.model_selection import train_test_split

    def get_iris_data() -> pd.DataFrame:
        iris = datasets.load_iris()
        data = pd.DataFrame(
            {
                "sepalLength": iris.data[:, 0],
                "sepalWidth": iris.data[:, 1],
                "petalLength": iris.data[:, 2],
                "petalWidth": iris.data[:, 3],
                "species": iris.target,
            }
        )

        print("Initial Dataset:")
        print(data.head())

        return data

    def create_training_set(dataset: pd.DataFrame, test_size: float = 0.3):
        # Features
        x = dataset[["sepalLength", "sepalWidth", "petalLength", "petalWidth"]]
        # Labels
        y = dataset["species"]

        # Split dataset into training set and test set
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=11)

        return x_train, x_test, y_train, y_test

    def save_pickle(object_file, target_object):
        with open(object_file, "wb") as f:
            pickle.dump(target_object, f)

    dataset = get_iris_data()
    x_train, x_test, y_train, y_test = create_training_set(dataset)

    save_pickle(x_train_file.path, x_train)
    save_pickle(x_test_file.path, x_test)
    save_pickle(y_train_file.path, y_train)
    save_pickle(y_test_file.path, y_test)
    


@dsl.component(
    base_image=base_image,
    packages_to_install=["pandas", "scikit-learn"],
)
def validate_data():
    pass


@dsl.component(
    base_image=base_image,
    packages_to_install=["pandas", "scikit-learn"],
)
def train_model(
    x_train_file: dsl.Input[dsl.Dataset],
    y_train_file: dsl.Input[dsl.Dataset],
    model_file: dsl.Output[dsl.Model],
):
    import pickle

    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier

    def load_pickle(object_file):
        with open(object_file, "rb") as f:
            target_object = pickle.load(f)  # noqa: S301

        return target_object

    def save_pickle(object_file, target_object):
        with open(object_file, "wb") as f:
            pickle.dump(target_object, f)

    def train_iris(x_train: pd.DataFrame, y_train: pd.DataFrame):
        model = RandomForestClassifier(n_estimators=100)
        model.fit(x_train, y_train)

        return model

    x_train = load_pickle(x_train_file.path)
    y_train = load_pickle(y_train_file.path)

    model = train_iris(x_train, y_train)

    save_pickle(model_file.path, model)


@dsl.component(
    base_image=base_image,
    packages_to_install=["pandas", "scikit-learn"],
)
def evaluate_model(
    x_test_file: dsl.Input[dsl.Dataset],
    y_test_file: dsl.Input[dsl.Dataset],
    model_file: dsl.Input[dsl.Model],
    mlpipeline_metrics_file: dsl.Output[dsl.Metrics],
):
    import json
    import pickle

    from sklearn.metrics import accuracy_score

    def load_pickle(object_file):
        with open(object_file, "rb") as f:
            target_object = pickle.load(f)  # noqa: S301

        return target_object

    x_test = load_pickle(x_test_file.path)
    y_test = load_pickle(y_test_file.path)
    model = load_pickle(model_file.path)

    y_pred = model.predict(x_test)

    accuracy_score_metric = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy_score_metric}")

    metrics = {
        "metrics": [
            {
                "name": "accuracy-score",
                "numberValue": accuracy_score_metric,
                "format": "PERCENTAGE",
            },
        ]
    }

    with open(mlpipeline_metrics_file.path, "w") as f:
        json.dump(metrics, f)


@dsl.component(
    base_image=data_science_image,
    packages_to_install=["pandas", "skl2onnx"],
)
def model_to_onnx(model_file: dsl.Input[dsl.Model], onnx_model_file: dsl.Output[dsl.Model]):
    import pickle

    from skl2onnx import to_onnx
    from skl2onnx.common.data_types import FloatTensorType

    def load_pickle(object_file):
        with open(object_file, "rb") as f:
            target_object = pickle.load(f)  # noqa: S301

        return target_object

    model = load_pickle(model_file.path)

    initial_type = [("float_input", FloatTensorType([None, 4]))]
    onnx_model = to_onnx(model, initial_types=initial_type, options={id(model): {"zipmap": False}})

    with open(onnx_model_file.path, "wb") as f:
        f.write(onnx_model.SerializeToString())


@dsl.component(
    base_image=base_image,
    packages_to_install=["pandas", "onnxruntime"],
)
def validate_model(onnx_model_file: dsl.Input[dsl.Model]):
    import onnxruntime

    session = onnxruntime.InferenceSession(onnx_model_file.path)

    input_name = session.get_inputs()[0].name
    label_name = session.get_outputs()[0].name

    input_values = [[5, 3, 1.6, 0.2]]

    print(f"Performing test prediction on {input_values}")
    result = session.run([label_name], {input_name: input_values})[0]

    print(f"Response: {result}")


@component(
    base_image=data_science_image,
    packages_to_install=["kubernetes"],
)
def deploy_model(
    onnx_model_file: Input[Artifact],
    model_name: str,
    model_format: str,
    namespace: str,
    runtime: str,
    version: str
):
    """
    Deploys a model to ModelMesh using an InferenceService and the Kubernetes Python API.

    Parameters:
    - onnx_model_file: The ONNX model artifact (with .uri pointing to a Data Connection-backed location)
    - model_name: Name of the InferenceService (e.g., 'iris')
    - namespace: Kubernetes namespace where the service will be created
    - model_format: Model format (default 'onnx')
    - runtime: Name of the ServingRuntime to use (default 'iris')
    - version: Model format version (default '1')
    """
    from kubernetes import client, config
    from urllib.parse import urlparse
    import json

    # Load in-cluster Kubernetes configuration
    config.load_incluster_config()

    # Get the artifact URI, e.g. "s3://iris-data-conn/some/path/model.onnx"
    model_uri = onnx_model_file.uri
    print(f"Received model URI: {model_uri}")

    # Parse the URI to extract Data Connection key and path
    parsed = urlparse(model_uri)

    if parsed.scheme != "s3":
        raise ValueError(f"Unsupported URI scheme: {parsed.scheme}. Expected s3://")

    model_uri_key = parsed.netloc  # bucket name or data connection key
    model_uri_path = parsed.path.lstrip("/")  # remove leading slash

    print(f"Parsed storage key: {model_uri_key}")
    print(f"Parsed path: {model_uri_path}")

    # Define the Inference Server that will know about our models location in s3,
    # the format it is in (i.e. onnx for this example), and a reference to the serving runtime (which 
    # is responsible for executing the model and returning predictions when given inputs).
    inference_service = {
        "apiVersion": "serving.kserve.io/v1beta1",
        "kind": "InferenceService",
        "metadata": {
            "name": model_name,
            "namespace": namespace,
            "annotations": {
                "serving.kserve.io/deploymentMode": "ModelMesh",
                "openshift.io/display-name": model_name
            },
            "labels": {
                "opendatahub.io/dashboard": "true"
            }
        },
        "spec": {
            "predictor": {
                "model":    {
                    "runtime": runtime,
                    "modelFormat": {
                        "name": model_format,
                        "version": version
                    },
                    "storage": {
                        "key": model_uri_key,
                        "path": model_uri_path
                    }
                }
            }
        }
    }

    pretty_json = json.dumps(inference_service, indent=4)
    print('Applying Inference Service Configuration:')
    print(pretty_json)
    # Initialize the CustomObjects API
    api = client.CustomObjectsApi()

    try:
        # Check if InferenceService exists
        api.get_namespaced_custom_object(
            group="serving.kserve.io",
            version="v1beta1",
            namespace=namespace,
            plural="inferenceservices",
            name=model_name,
        )

        print("InferenceService already exists. Patching it...")
        api.patch_namespaced_custom_object(
            group="serving.kserve.io",
            version="v1beta1",
            namespace=namespace,
            plural="inferenceservices",
            name=model_name,
            body=inference_service
        )
    except client.exceptions.ApiException as e:
        if e.status == 404:
            print("InferenceService not found. Creating a new one...")
            api.create_namespaced_custom_object(
                group="serving.kserve.io",
                version="v1beta1",
                namespace=namespace,
                plural="inferenceservices",
                body=inference_service
            )
        else:
            raise



@kfp.dsl.pipeline(
    name="Iris Pipeline",
)
def iris_pipeline(
    model_obc: str = "iris-detect-model",
    model_format: str = "onnx",
    version: str = "1",
    namespace: str = "fharo-testing",
    runtime: str = "triton-multi-model-server",
):
    data_prep_task = data_prep()
    data_prep_task.set_caching_options(False)

    train_model_task = train_model(
        x_train_file=data_prep_task.outputs["x_train_file"],
        y_train_file=data_prep_task.outputs["y_train_file"],
    )
    train_model_task.set_caching_options(False)

    evaluate_model_task = evaluate_model(  # noqa: F841
        x_test_file=data_prep_task.outputs["x_test_file"],
        y_test_file=data_prep_task.outputs["y_test_file"],
        model_file=train_model_task.output,
    )
    evaluate_model_task.set_caching_options(False)

    model_to_onnx_task = model_to_onnx(  # noqa: F841
        model_file=train_model_task.output,
    )
    model_to_onnx_task.set_caching_options(False)

    validate_model_task = validate_model(onnx_model_file=model_to_onnx_task.output)  # noqa: F841
    validate_model_task.set_caching_options(False)

    deploy_model_task = deploy_model(
        onnx_model_file=model_to_onnx_task.output,
        model_name=model_obc,
        model_format=model_format,
        namespace=namespace,
        runtime=runtime,
        version=version
    )
    deploy_model_task.set_caching_options(False)
    deploy_model_task.after(validate_model_task)

def get_rhods_dashboard_url_from_env():
    """
    Returns the RHODS dashboard pipeline URL by parsing the NOTEBOOK_ARGS env variable.
    Returns None if not found or if the format is invalid.
    """
    args = os.environ.get("NOTEBOOK_ARGS", "")
    for part in args.split():
        if part.startswith("--ServerApp.tornado_settings="):
            try:
                settings_json = part.split("=", 1)[1]
                tornado_settings = json.loads(settings_json)
                hub_host = tornado_settings.get("hub_host")
                if hub_host:
                    return hub_host.rstrip("/")
            except Exception:
                pass  # Fail silently if malformed
    return None

def get_project_name_from_env():
    """
    Returns the data science project name this notebook is under.
    """
    args = os.environ.get("NOTEBOOK_ARGS", "")
    for part in args.split():
        if part.startswith("--ServerApp.tornado_settings="):
            try:
                settings_json = part.split("=", 1)[1]
                tornado_settings = json.loads(settings_json)
                hub_prefix = tornado_settings.get("hub_prefix")
                if hub_prefix:
                    return hub_prefix.removeprefix("/projects/")
            except Exception:
                pass  # Fail silently if malformed
    return None

# if __name__ == "__main__":
#     kfp.compiler.Compiler().compile(iris_pipeline, package_path=__file__.replace(".py", ".yaml"))

# OPTIONAL
# Instead of compiling to a Intermediate Representation (IR) yaml pipeline as above,
# One could call the kubeflow endpoint directly to submit the pipeline. This eliminates
# 1 manual step of transferring/importing the pipeline file to the pipeline server.
# If you would like to do so please comment the if conditional above and uncomment
# the if conditional below:

# Note: The environment variables BEARER_TOKEN and KUBEFLOW_ENDPOINT in the .env 
# file would have to be updated.

if __name__ == "__main__":
    print(f"Connecting to kfp: {kubeflow_endpoint}")
    sa_token_path = "/run/secrets/kubernetes.io/serviceaccount/token"  # noqa: S105
    if "BEARER_TOKEN" in os.environ:
        bearer_token = os.environ["BEARER_TOKEN"]
    elif os.path.isfile(sa_token_path):
        with open(sa_token_path) as f:
            bearer_token = f.read().rstrip()
    # Check if the script is running in a k8s pod
    # Get the CA from the service account if it is
    # Skip the CA if it is not
    sa_ca_cert = "/run/secrets/kubernetes.io/serviceaccount/service-ca.crt"
    if os.path.isfile(sa_ca_cert) and "svc" in kubeflow_endpoint:
        ssl_ca_cert = sa_ca_cert
    else:
        ssl_ca_cert = None
    print()
    print()
    client = kfp.Client(
        host=kubeflow_endpoint,
        existing_token=bearer_token,
        ssl_ca_cert=ssl_ca_cert,
    )
    experiment_name="Iris Experiment"
    rhods_dashboard_url = get_rhods_dashboard_url_from_env()
    datascience_project_name = get_project_name_from_env()
    result = client.create_run_from_pipeline_func(iris_pipeline, arguments={}, experiment_name=experiment_name)
    experiment_id = client.get_experiment(experiment_name=experiment_name,namespace=datascience_project_name).experiment_id
    print()
    print()
    print(f"Starting pipeline run with run_id: {result.run_id}")
    if rhods_dashboard_url and datascience_project_name and experiment_id and result.run_id:
        print(f"RHODS Dashboard Pipeline Run URL: {rhods_dashboard_url}/experiments/{datascience_project_name}/{experiment_id}/runs/{result.run_id}")
    else:
        print("Could not determine RHODS dashboard pipeline run URL. Please go to The RedHat Openshift AI Dashboard > Experiments > Experiments and Runs section to find your run.")
