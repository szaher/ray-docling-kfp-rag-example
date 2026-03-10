# Milvus on OpenShift (OCP)

Deploy Milvus vector database on OpenShift Container Platform with proper security context constraints.

## Architecture

```
                    ┌──────────────┐
                    │  OCP Router  │
                    └──────┬───────┘
                           │
              ┌────────────┼────────────┐
              │            │            │
        ┌─────▼────┐ ┌────▼─────┐ ┌────▼─────┐
        │  Attu UI │ │  Milvus  │ │ MinIO UI │
        │ (port    │ │  gRPC    │ │ (port    │
        │  3000)   │ │ (19530)  │ │  9090)   │
        └──────────┘ └────┬─────┘ └──────────┘
                          │
              ┌───────────┼───────────┐
              │           │           │
        ┌─────▼────┐ ┌───▼────┐ ┌────▼─────┐
        │  MinIO   │ │  etcd  │ │  Milvus  │
        │ (S3 API) │ │        │ │Standalone│
        └──────────┘ └────────┘ └──────────┘
```

**Components:**

| Component | Purpose | Managed By |
|-----------|---------|------------|
| Milvus Operator | Lifecycle management of Milvus instances | Helm |
| Milvus Standalone | Vector database engine | Operator |
| etcd | Metadata storage | Operator (in-cluster) |
| MinIO | Object storage (S3-compatible) | Existing (default namespace) |
| Attu | Web UI for Milvus (optional) | Static manifests |

## Prerequisites

- OpenShift 4.12+ cluster with `cluster-admin` access
- `oc` CLI authenticated to your cluster
- `helm` CLI v3+
- `podman` or `docker` CLI (for mirroring images)
- A StorageClass that supports `ReadWriteOnce` PVCs
- MinIO already deployed in the `default` namespace (see `../minio.yaml`)

## File Overview

| File | Description |
|------|-------------|
| `00-namespace.yaml` | Creates the `milvus` namespace |
| `01-scc.yaml` | Custom SCC (optional fallback — not needed if using namespace UID range) |
| `03-milvus-operator.yaml` | Operator namespace (operator installed via Helm) |
| `04-milvus-instance.yaml` | MinIO secret mirror + Milvus CR (standalone mode) with OCP-safe security contexts |
| `05-milvus-route.yaml` | Routes for Milvus gRPC and Attu UI |
| `06-attu.yaml` | Attu web UI deployment and service (optional) |
| `07-networkpolicy.yaml` | NetworkPolicies for internal and ingress traffic |

> **Note:** MinIO is not included here — Milvus connects to the existing MinIO
> in the `default` namespace (`minio-service.default.svc.cluster.local:9000`).
> The MinIO credentials are mirrored into a Secret in the `milvus` namespace
> inside `04-milvus-instance.yaml`.

## Step 0: Mirror Images to Quay

OCP enforces **short-name mode** (CRI-O rejects images without a full registry prefix)
and Docker Hub has **rate limits** on unauthenticated pulls. All images must be mirrored
to a Quay.io org (or another accessible registry) before deploying.

```bash
# Milvus Operator
podman pull --platform linux/amd64 docker.io/milvusdb/milvus-operator:v1.3.6
podman tag docker.io/milvusdb/milvus-operator:v1.3.6 quay.io/rhoai-szaher/milvus-operator:v1.3.6
podman push quay.io/rhoai-szaher/milvus-operator:v1.3.6

# Milvus
podman pull --platform linux/amd64 docker.io/milvusdb/milvus:v2.6.11
podman tag docker.io/milvusdb/milvus:v2.6.11 quay.io/rhoai-szaher/milvus:v2.6.11
podman push quay.io/rhoai-szaher/milvus:v2.6.11

# etcd (bitnami chart — use registry/repository split)
podman pull --platform linux/amd64 docker.io/milvusdb/etcd:3.5.25-r1
podman tag docker.io/milvusdb/etcd:3.5.25-r1 quay.io/rhoai-szaher/milvus-etcd:3.5.25-r1
podman push quay.io/rhoai-szaher/milvus-etcd:3.5.25-r1

# Attu UI
podman pull --platform linux/amd64 docker.io/zilliz/attu:v2.4
podman tag docker.io/zilliz/attu:v2.4 quay.io/rhoai-szaher/attu:v2.4
podman push quay.io/rhoai-szaher/attu:v2.4
```

> **Important:** Make sure each Quay repository is set to **public**, or create an
> image pull secret and link it to the `default` service account:
>
> ```bash
> oc create secret docker-registry quay-pull-secret \
>   -n milvus \
>   --docker-server=quay.io \
>   --docker-username=<quay-user> \
>   --docker-password=<quay-password>
> oc secrets link default quay-pull-secret --for=pull -n milvus
> ```

> **Important:** Always use `--platform linux/amd64` when pulling. Pushing an
> `arm64` image will cause `Exec format error` on x86_64 nodes.

## Installation

### Step 1: Create namespace

```bash
oc apply -f 00-namespace.yaml
```

> **Note:** `01-scc.yaml` defines a custom SCC as a fallback but is **not required**
> for the default setup. The manifests use the namespace's UID range to satisfy
> the built-in `restricted-v2` SCC. Only apply `01-scc.yaml` if you encounter
> SCC issues that cannot be resolved with UID range adjustments.

### Step 2: Create the Milvus bucket in your existing MinIO

Milvus needs a bucket to store data. Create it in your existing MinIO (default namespace):

```bash
# Port-forward to your existing MinIO
oc port-forward svc/minio-service 9000:9000 -n default &

# Using the mc CLI (update credentials to match your minio-secret)
mc alias set ocp-minio http://localhost:9000 <MINIO_USER> <MINIO_PASSWORD>
mc mb ocp-minio/milvus-bucket

# Stop port-forward
kill %1
```

### Step 3: Install the Milvus Operator

```bash
# Apply the operator namespace
oc apply -f 03-milvus-operator.yaml

# Install the operator via Helm
helm repo add milvus-operator https://zilliztech.github.io/milvus-operator/
helm repo update

helm install milvus-operator milvus-operator/milvus-operator \
  -n milvus-operator \
  --set image.repository=quay.io/rhoai-szaher/milvus-operator \
  --set operator.securityContext.runAsNonRoot=true \
  --set operator.securityContext.allowPrivilegeEscalation=false \
  --set operator.securityContext.seccompProfile.type=RuntimeDefault
```

Wait for the operator to be ready:

```bash
oc wait --for=condition=available deployment/milvus-operator -n milvus-operator --timeout=180s
```

> **Note:** The operator container is named `manager`, not `milvus-operator`.
> If you need to patch the image manually:
> ```bash
> oc -n milvus-operator set image deployment/milvus-operator \
>   manager=quay.io/rhoai-szaher/milvus-operator:v1.3.6
> ```

### Step 4: Deploy the Milvus instance

```bash
oc apply -f 04-milvus-instance.yaml
```

Monitor the rollout:

```bash
# Watch all pods in the milvus namespace
oc get pods -n milvus -w

# Check the Milvus CR status
oc get milvus -n milvus
```

### Step 5: Expose Milvus and deploy Attu (optional)

```bash
oc apply -f 05-milvus-route.yaml
oc apply -f 06-attu.yaml
```

### Step 6: Apply NetworkPolicies (if needed)

Only apply this if your cluster enforces NetworkPolicy by default:

```bash
oc apply -f 07-networkpolicy.yaml
```

### Step 7: Verify the deployment

```bash
# All pods should be 1/1 Running
oc get pods -n milvus

# Milvus CR should show "Healthy"
oc get milvus milvus -n milvus -o jsonpath='{.status.status}'

# Get the Attu route URL
oc get route milvus-attu -n milvus -o jsonpath='{.spec.host}'
```

## Connecting to Milvus

### From Attu UI

Open the Attu route URL in your browser. When prompted for the Milvus address, use
the **internal service name** (Attu runs in the same namespace):

- **Milvus Address:** `milvus-milvus:19530`
- **Database:** `default`

> **Do not** use the external route URL from within Attu — it runs inside the
> cluster and should connect directly via the service.

### From within the cluster

```python
from pymilvus import connections

connections.connect(
    alias="default",
    host="milvus-milvus.milvus.svc.cluster.local",
    port="19530",
)
```

### From outside the cluster (via Route)

```python
from pymilvus import connections

connections.connect(
    alias="default",
    host="<milvus-grpc route host>",
    port="443",
    secure=True,
)
```

### Using port-forward (for development)

```bash
oc port-forward svc/milvus-milvus 19530:19530 -n milvus
```

```python
from pymilvus import connections

connections.connect(host="localhost", port="19530")
```

## OCP Security Fixes Explained

OpenShift enforces the `restricted-v2` SCC by default, which is stricter than
upstream Kubernetes. The manifests in this directory address the following issues:

### 1. RunAsUser — namespace UID range

**Problem:** Milvus, etcd, and MinIO images may attempt to run as root (UID 0) or
a specific UID not in the namespace's allowed range. The `restricted-v2` SCC only
allows UIDs within the range assigned to the namespace.

**Fix:** All pod and container security contexts use the namespace's UID range
(e.g. `1000960000`) for `runAsUser` and `fsGroup`. Find your namespace's range with:

```bash
oc get namespace milvus -o jsonpath='{.metadata.annotations.openshift\.io/sa\.scc\.uid-range}'
# Example output: 1000960000/10000
```

Update the `runAsUser` and `fsGroup` values in `04-milvus-instance.yaml` to match
the first UID in your range.

### 2. Capabilities and privilege escalation

**Problem:** The `restricted-v2` SCC drops all Linux capabilities and blocks
privilege escalation. Containers that don't explicitly set these will fail.

**Fix:** Every container in these manifests explicitly sets:

```yaml
securityContext:
  allowPrivilegeEscalation: false
  capabilities:
    drop:
      - ALL
  seccompProfile:
    type: RuntimeDefault
```

### 3. fsGroup and volume permissions

**Problem:** etcd requires write access to its data volume. Without a matching
`fsGroup`, the mounted PVC may be owned by root and inaccessible.

**Fix:** The etcd values in `04-milvus-instance.yaml` set `fsGroup` and
`runAsUser` to the namespace UID range so the etcd process can write to its volume.

### 4. Network isolation

**Problem:** OCP clusters with default-deny NetworkPolicies will block inter-pod
communication between Milvus components and to MinIO in the `default` namespace.

**Fix:** `07-networkpolicy.yaml` allows:
- All pods within the `milvus` namespace to talk to each other
- Egress to MinIO in the `default` namespace on port 9000
- DNS resolution
- Traffic from the OpenShift ingress controller (for Routes)

### 5. seccomp profiles

**Problem:** OCP 4.11+ requires explicit seccomp profile declarations under
Pod Security Admission.

**Fix:** All pod specs include `seccompProfile.type: RuntimeDefault`.

### 6. Image short names and Docker Hub rate limits

**Problem:** OCP's CRI-O runtime enforces short-name mode — images without a
full registry prefix (e.g. `milvusdb/milvus`) are rejected with
`ImageInspectError`. Docker Hub also rate-limits unauthenticated pulls.

**Fix:** All images are mirrored to `quay.io/rhoai-szaher/` and referenced with
fully qualified names. See [Step 0](#step-0-mirror-images-to-quay) for mirroring
instructions.

> **Important:** The bitnami etcd Helm chart has separate `image.registry` and
> `image.repository` fields. Setting the full `quay.io/org/repo` path in
> `repository` alone causes a double-prefix like `docker.io/quay.io/...`.
> Always split them: `registry: quay.io`, `repository: rhoai-szaher/milvus-etcd`.

## Troubleshooting

### `ImageInspectError` — short name mode enforcing

**Error:** `short name mode is enforcing, but image name ... returns ambiguous list`

**Fix:** Ensure all images use fully qualified names with a registry prefix.
See [Step 0](#step-0-mirror-images-to-quay).

### `Exec format error`

**Error:** `exec container process /manager: Exec format error`

**Cause:** The image was built for the wrong CPU architecture (e.g. `arm64` on
an `amd64` cluster).

**Fix:** Re-pull with `--platform linux/amd64`, re-tag, and re-push. Then force
a fresh pull by setting `imagePullPolicy: Always` or using a new image tag.

### `ImagePullBackOff` — Docker Hub rate limit

**Error:** `toomanyrequests: You have reached your unauthenticated pull rate limit`

**Fix:** Mirror images to Quay.io. See [Step 0](#step-0-mirror-images-to-quay).

### SCC rejection — UID not in allowed range

**Error:** `.containers[0].runAsUser: Invalid value: 1000: must be in the ranges: [1000960000, 1000969999]`

**Fix:** Update `runAsUser` and `fsGroup` in `04-milvus-instance.yaml` to match
your namespace's UID range:

```bash
oc get namespace milvus -o jsonpath='{.metadata.annotations.openshift\.io/sa\.scc\.uid-range}'
```

### Pods stuck in `CreateContainerConfigError`

Check SCC assignment and pod security context:

```bash
oc describe pod <pod-name> -n milvus
oc get pod <pod-name> -n milvus -o yaml | grep -A10 "securityContext"
```

### etcd PVC deletion error during reconciliation

**Error:** `delete pvc milvus/data-milvus-etcd-0 failed: persistentvolumeclaims not found`

**Fix:** Delete the Milvus CR, clean up orphaned resources, and re-apply:

```bash
oc delete milvus milvus -n milvus
oc delete statefulset -l app.kubernetes.io/instance=milvus-etcd -n milvus
oc delete pvc -l app.kubernetes.io/instance=milvus-etcd -n milvus
oc delete svc -l app.kubernetes.io/instance=milvus-etcd -n milvus
oc apply -f 04-milvus-instance.yaml
```

### Operator not reconciling

```bash
oc logs deployment/milvus-operator -n milvus-operator
oc get events -n milvus --sort-by='.lastTimestamp'
```

## Cleanup

```bash
# Delete in reverse order
oc delete -f 07-networkpolicy.yaml
oc delete -f 06-attu.yaml
oc delete -f 05-milvus-route.yaml
oc delete -f 04-milvus-instance.yaml

# Uninstall the operator
helm uninstall milvus-operator -n milvus-operator

oc delete -f 03-milvus-operator.yaml
oc delete -f 00-namespace.yaml
```

> **Note:** This does not remove your existing MinIO in the `default` namespace.
