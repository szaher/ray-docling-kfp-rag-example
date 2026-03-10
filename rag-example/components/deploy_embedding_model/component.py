"""KFP component: Deploy a text embedding model on OpenShift AI.

Creates a KServe InferenceService with a TEI-compatible ServingRuntime
for serving embedding requests via an OpenAI-compatible API.
"""

from kfp import dsl


@dsl.component(
    base_image="quay.io/modh/odh-generic-data-science-notebook:v3-2025a-20250523",
    packages_to_install=["kubernetes>=28.1.0"],
)
def deploy_embedding_model(
    model_name: str,
    namespace: str,
    serving_runtime_name: str = "embedding-runtime",
    min_replicas: int = 1,
    max_replicas: int = 1,
    cpu_requests: str = "2",
    cpu_limits: str = "4",
    memory_requests: str = "4Gi",
    memory_limits: str = "8Gi",
    api_url: str = "",
    token: str = "",
) -> str:
    """Deploy a text embedding model using KServe InferenceService.

    Args:
        model_name: HuggingFace model ID (e.g. 'sentence-transformers/all-MiniLM-L6-v2').
        namespace: Namespace to deploy the InferenceService.
        serving_runtime_name: Name of the embedding ServingRuntime CR.
        min_replicas: Minimum number of replicas.
        max_replicas: Maximum number of replicas.
        cpu_requests: CPU requests per replica.
        cpu_limits: CPU limits per replica.
        memory_requests: Memory requests per replica.
        memory_limits: Memory limits per replica.
        api_url: OpenShift API URL.
        token: OpenShift auth token.

    Returns:
        The embedding service endpoint URL.
    """
    import time

    from kubernetes import client as kclient

    configuration = kclient.Configuration()
    configuration.host = api_url
    configuration.verify_ssl = True
    configuration.api_key["authorization"] = f"Bearer {token}"

    api_client = kclient.ApiClient(configuration)
    custom_api = kclient.CustomObjectsApi(api_client)

    # Derive InferenceService name from model name
    isvc_name = model_name.split("/")[-1].lower().replace("_", "-").replace(".", "-")

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
            "predictor": {
                "minReplicas": min_replicas,
                "maxReplicas": max_replicas,
                "model": {
                    "modelFormat": {"name": "embedding"},
                    "runtime": serving_runtime_name,
                    "storageUri": f"hf://{model_name}",
                    "resources": {
                        "requests": {
                            "cpu": cpu_requests,
                            "memory": memory_requests,
                        },
                        "limits": {
                            "cpu": cpu_limits,
                            "memory": memory_limits,
                        },
                    },
                },
            },
        },
    }

    # Create or update
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

    # Wait for Ready
    print("Waiting for embedding InferenceService to become ready...")
    for _ in range(60):
        time.sleep(15)
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
            print(f"Embedding InferenceService ready: {url}")
            return url

    raise TimeoutError(
        f"Embedding InferenceService '{isvc_name}' did not become ready "
        "within 15 minutes."
    )


if __name__ == "__main__":
    from kfp import compiler

    compiler.Compiler().compile(
        deploy_embedding_model,
        package_path=__file__.replace(".py", "_component.yaml"),
    )
