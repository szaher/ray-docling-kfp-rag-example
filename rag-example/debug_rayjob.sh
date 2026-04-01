#!/bin/bash
# Debug RayJob failures
# Usage: ./debug_rayjob.sh <rayjob-name> [namespace]

RAYJOB_NAME=${1}
NAMESPACE=${2:-ray-docling}

if [ -z "$RAYJOB_NAME" ]; then
    echo "Usage: $0 <rayjob-name> [namespace]"
    echo ""
    echo "Recent RayJobs:"
    oc get rayjob -n $NAMESPACE --sort-by=.metadata.creationTimestamp | tail -10
    exit 1
fi

echo "==================================================================="
echo "Debugging RayJob: $RAYJOB_NAME in namespace: $NAMESPACE"
echo "==================================================================="
echo ""

echo "--- RayJob Status ---"
oc get rayjob $RAYJOB_NAME -n $NAMESPACE
echo ""

echo "--- RayJob Details ---"
oc get rayjob $RAYJOB_NAME -n $NAMESPACE -o yaml | grep -A 30 "status:"
echo ""

echo "--- Ray Cluster Pods ---"
CLUSTER_NAME=$(oc get rayjob $RAYJOB_NAME -n $NAMESPACE -o jsonpath='{.status.rayClusterName}')
oc get pods -n $NAMESPACE -l ray.io/cluster=$CLUSTER_NAME
echo ""

echo "--- RayJob Submitter Logs ---"
oc logs job/$RAYJOB_NAME -n $NAMESPACE --tail=50
echo ""

echo "--- Ray Head Pod Logs (last 50 lines) ---"
HEAD_POD=$(oc get pods -n $NAMESPACE -l ray.io/cluster=$CLUSTER_NAME,ray.io/node-type=head -o jsonpath='{.items[0].metadata.name}')
if [ -n "$HEAD_POD" ]; then
    oc logs $HEAD_POD -n $NAMESPACE -c ray-head --tail=50
else
    echo "No head pod found"
fi
echo ""

echo "--- Worker Pods Status ---"
oc get pods -n $NAMESPACE -l ray.io/cluster=$CLUSTER_NAME,ray.io/node-type=worker -o wide
echo ""

echo "--- Recent Events ---"
oc get events -n $NAMESPACE --sort-by='.lastTimestamp' | grep $RAYJOB_NAME | tail -20
echo ""

echo "==================================================================="
echo "Debug Summary for RayJob: $RAYJOB_NAME"
echo "==================================================================="
echo "Job Status: $(oc get rayjob $RAYJOB_NAME -n $NAMESPACE -o jsonpath='{.status.jobStatus}')"
echo "Cluster State: $(oc get rayjob $RAYJOB_NAME -n $NAMESPACE -o jsonpath='{.status.rayClusterStatus.state}')"
echo "Ready Workers: $(oc get rayjob $RAYJOB_NAME -n $NAMESPACE -o jsonpath='{.status.rayClusterStatus.readyWorkerReplicas}')"
echo "Desired Workers: $(oc get rayjob $RAYJOB_NAME -n $NAMESPACE -o jsonpath='{.status.rayClusterStatus.desiredWorkerReplicas}')"
echo ""

# Check for common issues
echo "--- Common Issues Check ---"
if [ "$(oc get rayjob $RAYJOB_NAME -n $NAMESPACE -o jsonpath='{.status.jobStatus}')" == "FAILED" ]; then
    echo "❌ Job FAILED"
    echo "   Check submitter logs above for error details"
fi

READY=$(oc get rayjob $RAYJOB_NAME -n $NAMESPACE -o jsonpath='{.status.rayClusterStatus.readyWorkerReplicas}')
DESIRED=$(oc get rayjob $RAYJOB_NAME -n $NAMESPACE -o jsonpath='{.status.rayClusterStatus.desiredWorkerReplicas}')
if [ "$READY" != "$DESIRED" ]; then
    echo "⚠️  Workers not ready: $READY/$DESIRED"
    echo "   Check worker pod status and logs"
fi

# Check if workers are pending
PENDING=$(oc get pods -n $NAMESPACE -l ray.io/cluster=$CLUSTER_NAME,ray.io/node-type=worker --field-selector=status.phase=Pending -o name | wc -l)
if [ "$PENDING" -gt 0 ]; then
    echo "⚠️  $PENDING worker pod(s) pending - possible resource constraints"
    echo "   Check: oc describe pod -n $NAMESPACE -l ray.io/cluster=$CLUSTER_NAME,ray.io/node-type=worker"
fi

echo ""
echo "For more details:"
echo "  Full RayJob YAML: oc get rayjob $RAYJOB_NAME -n $NAMESPACE -o yaml"
echo "  Head pod logs:    oc logs $HEAD_POD -n $NAMESPACE -c ray-head"
echo "  Worker logs:      oc logs -n $NAMESPACE -l ray.io/cluster=$CLUSTER_NAME,ray.io/node-type=worker"
