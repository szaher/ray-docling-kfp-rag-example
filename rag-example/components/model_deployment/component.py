"""KFP component: Deploy a model on OpenShift AI using vLLM ServingRuntime.

Creates an InferenceService CR that deploys a model from a model registry
or direct path using the vLLM serving runtime.

Uses in-cluster Kubernetes config (service account token) — no
explicit API URL or token required.
"""

from kfp import dsl


@dsl.component(
    base_image="quay.io/modh/odh-generic-data-science-notebook:v3-20250827",
    packages_to_install=["kubernetes>=28.1.0"],
)
def model_deployment(
    model_name: str,
    namespace: str,
    runtime: str = "vllm",
    model_path: str = "",
    serving_runtime_name: str = "vllm-runtime",
    min_replicas: int = 1,
    max_replicas: int = 1,
    gpu_count: int = 1,
) -> str:
    """Deploy a model on OpenShift AI using an InferenceService.

    Args:
        model_name: Name/path of the model (e.g. 'mistralai/Mistral-7B-Instruct-v0.3').
        namespace: Namespace to deploy the InferenceService.
        runtime: Serving runtime to use (default: vllm).
        model_path: Optional PVC path to the model. If empty, the model is pulled from HuggingFace.
        serving_runtime_name: Name of the ServingRuntime CR.
        min_replicas: Minimum number of replicas.
        max_replicas: Maximum number of replicas.
        gpu_count: Number of GPUs per replica.

    Returns:
        The inference endpoint URL.
    """
    import time

    from kubernetes import client as kclient
    from kubernetes import config

    config.load_incluster_config()
    custom_api = kclient.CustomObjectsApi()

    isvc_name = model_name.split("/")[-1].lower().replace(".", "-")

    # Build the InferenceService spec
    predictor_spec = {
        "minReplicas": min_replicas,
        "maxReplicas": max_replicas,
        "model": {
            "modelFormat": {"name": runtime},
            "runtime": serving_runtime_name,
            "resources": {
                "requests": {"cpu": "2", "memory": "8Gi"},
                "limits": {"cpu": "4", "memory": "16Gi"},
            },
        },
    }

    if gpu_count > 0:
        predictor_spec["model"]["resources"]["requests"]["nvidia.com/gpu"] = str(gpu_count)
        predictor_spec["model"]["resources"]["limits"]["nvidia.com/gpu"] = str(gpu_count)

    if model_path:
        predictor_spec["model"]["storageUri"] = model_path
    else:
        predictor_spec["model"]["storageUri"] = f"hf://{model_name}"

    isvc = {
        "apiVersion": "serving.kserve.io/v1beta1",
        "kind": "InferenceService",
        "metadata": {
            "name": isvc_name,
            "namespace": namespace,
            "annotations": {
                "serving.kserve.io/deploymentMode": "RawDeployment",
            },
        },
        "spec": {
            "predictor": predictor_spec,
        },
    }

    # Create or update the InferenceService
    try:
        custom_api.get_namespaced_custom_object(
            group="serving.kserve.io",
            version="v1beta1",
            namespace=namespace,
            plural="inferenceservices",
            name=isvc_name,
        )
        custom_api.patch_namespaced_custom_object(
            group="serving.kserve.io",
            version="v1beta1",
            namespace=namespace,
            plural="inferenceservices",
            name=isvc_name,
            body=isvc,
        )
        print(f"InferenceService '{isvc_name}' updated.")
    except kclient.rest.ApiException as e:
        if e.status == 404:
            custom_api.create_namespaced_custom_object(
                group="serving.kserve.io",
                version="v1beta1",
                namespace=namespace,
                plural="inferenceservices",
                body=isvc,
            )
            print(f"InferenceService '{isvc_name}' created.")
        else:
            raise

    # Wait for the InferenceService to become ready
    print("Waiting for InferenceService to become ready...")
    for attempt in range(60):
        time.sleep(30)
        obj = custom_api.get_namespaced_custom_object(
            group="serving.kserve.io",
            version="v1beta1",
            namespace=namespace,
            plural="inferenceservices",
            name=isvc_name,
        )
        conditions = obj.get("status", {}).get("conditions", [])
        ready = any(
            c.get("type") == "Ready" and c.get("status") == "True"
            for c in conditions
        )
        if ready:
            url = obj.get("status", {}).get("url", "")
            print(f"InferenceService ready: {url}")
            return f"{url}/v1"

    raise TimeoutError(
        f"InferenceService '{isvc_name}' did not become ready within 30 minutes."
    )


if __name__ == "__main__":
    from kfp import compiler

    compiler.Compiler().compile(
        model_deployment,
        package_path=__file__.replace(".py", "_component.yaml"),
    )
