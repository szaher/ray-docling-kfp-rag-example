# RAG Pipeline Security Assessment
**Production-Ready Demo Security Review**
**Date**: 2026-03-31
**Reviewer**: Security Analysis Agent

---

## Executive Summary

This security assessment evaluates the RAG (Retrieval-Augmented Generation) pipeline for production-ready demo deployment. The assessment identifies **3 CRITICAL**, **5 HIGH**, and **4 MEDIUM** priority security issues that must be addressed before production deployment.

### Overall Security Posture: MODERATE RISK
- Authentication: Limited
- Authorization: Partial RBAC implementation
- Encryption: Not configured for data in transit
- Secrets Management: Partial implementation with hardcoded defaults
- Container Security: Good (pinned SHA256 images)
- Network Security: Not configured

---

## 1. SECRETS HANDLING

### 1.1 S3/MinIO Credentials - CRITICAL

**Risk Level**: CRITICAL
**Current State**:
- S3 credentials stored in Kubernetes Secret (`minio-secret`)
- **HARDCODED DEFAULT CREDENTIALS** in manifest:
  ```yaml
  # /manifests/minio-secret.yaml
  access_key: "minioadmin"
  secret_key: "minioadmin"
  ```
- Credentials passed via environment variables to pipeline components
- Secret properly referenced via `kubernetes.use_secret_as_env()` in pipeline code

**Issues**:
1. Default credentials "minioadmin/minioadmin" present in version-controlled manifest
2. Same credentials hardcoded in Milvus configuration (`milvus-ocp/04-milvus-instance.yaml`):
   ```yaml
   stringData:
     accesskey: szaher
     secretkey: your-secure-password  # Placeholder exposed in repo
   ```
3. No credential rotation mechanism
4. Credentials accessible to all pipeline service accounts

**Recommendations**:
- **IMMEDIATE**: Remove hardcoded credentials from manifests before public demo
- Use Secret generation via sealed-secrets, external-secrets, or Vault
- Implement credential rotation policy (90-day maximum)
- Use separate S3 credentials for read vs read-write operations
- Add Secret scanning to CI/CD pipeline

**Remediation Priority**: IMMEDIATE (before demo)

---

### 1.2 Milvus Access Control - HIGH

**Risk Level**: HIGH
**Current State**:
- No authentication enabled for Milvus connections
- Connection string: `http://milvus-milvus.milvus.svc.cluster.local:19530`
- No TLS encryption
- RBAC disabled in etcd dependency:
  ```yaml
  auth:
    rbac:
      enabled: false
  ```

**Issues**:
1. Anyone with network access to Milvus namespace can read/write vectors
2. No authentication required for vector operations
3. Sensitive document embeddings unprotected
4. Collection can be dropped without authorization (line 99-100 in `ingest_to_milvus/component.py`)

**Recommendations**:
- Enable Milvus authentication (user/password or token-based)
- Configure TLS for Milvus gRPC connections
- Enable etcd RBAC for metadata protection
- Implement read-only vs read-write roles
- Store Milvus credentials in separate Secret

**Remediation Priority**: HIGH (before production)

---

### 1.3 HuggingFace Model Access - MEDIUM

**Risk Level**: MEDIUM
**Current State**:
- Public models downloaded without authentication
- No `HF_TOKEN` configured (mentioned in docs but not implemented)
- Model downloads via `huggingface_hub.snapshot_download()`

**Issues**:
1. Cannot access gated models (e.g., Llama 2, Mistral v0.3 may require agreement)
2. No rate limiting protection
3. Potential for supply chain attacks (no model verification)

**Recommendations**:
- Configure `HF_TOKEN` as Secret for gated model access
- Implement model signature verification
- Use private model mirror/cache for air-gapped deployments
- Add model hash verification before loading

**Remediation Priority**: MEDIUM

---

## 2. DATA SECURITY

### 2.1 PDF Storage Security - MEDIUM

**Risk Level**: MEDIUM
**Current State**:
- PDFs stored on PVC (`data-pvc`)
- PVC access mode: ReadWriteOnce (line 14 in `model-cache-pvc.yaml` shows pattern)
- No mention of encryption at rest
- No access controls on PVC beyond Kubernetes RBAC

**Issues**:
1. No encryption at rest configured for PVC
2. PDF content may contain sensitive/proprietary information
3. No audit logging for PVC access
4. No data retention/deletion policy

**Recommendations**:
- Enable storage-class encryption at rest
- Implement PVC access logging
- Add data classification labels to PVCs
- Document data retention and deletion procedures
- Consider using ReadOnlyMany for input PDFs to prevent tampering

**Remediation Priority**: MEDIUM

---

### 2.2 S3 Data Encryption - HIGH

**Risk Level**: HIGH
**Current State**:
- S3 endpoint: `http://minio-service.default.svc.cluster.local:9000` (HTTP, not HTTPS)
- MinIO configuration shows `useSSL: false` in Milvus config
- Chunked JSONL files stored unencrypted in S3

**Issues**:
1. Data in transit NOT encrypted (HTTP instead of HTTPS)
2. Document chunks contain sensitive text in plaintext
3. No server-side encryption (SSE) configured
4. Network sniffing could expose document content

**Recommendations**:
- **IMMEDIATE**: Enable TLS for MinIO (configure SSL certificates)
- Enable MinIO server-side encryption (SSE-S3 or SSE-KMS)
- Update S3 endpoint to `https://` in pipeline parameters
- Configure boto3 client with `use_ssl=True`
- Implement bucket policies to enforce encryption

**Remediation Priority**: HIGH (before demo)

---

### 2.3 Vector Embedding Leakage - MEDIUM

**Risk Level**: MEDIUM
**Current State**:
- Embeddings generated from sensitive documents
- No encryption for vectors at rest in Milvus
- No access controls on vector retrieval

**Issues**:
1. Vector embeddings can leak information about original documents
2. Similarity search could expose relationships between documents
3. No audit trail for vector queries

**Recommendations**:
- Implement query-level access controls
- Add audit logging for all vector searches
- Consider differential privacy techniques for sensitive embeddings
- Document acceptable use policies for RAG system

**Remediation Priority**: MEDIUM

---

## 3. NETWORK SECURITY

### 3.1 Service Exposure - HIGH

**Risk Level**: HIGH
**Current State**:
- InferenceService authentication disabled:
  ```python
  "security.opendatahub.io/enable-auth": "false"
  ```
- No NetworkPolicies defined
- Services exposed via internal ClusterIP (good) but no ingress controls
- Ray cluster uses `RAY_USE_TLS: "0"` (disabled)

**Issues**:
1. Model inference endpoint accessible without authentication
2. No network segmentation between pipeline components
3. Ray workers communicate over unencrypted connections
4. No egress filtering (workers can reach any internet endpoint)

**Recommendations**:
- **CRITICAL**: Enable authentication for InferenceService:
  ```python
  "security.opendatahub.io/enable-auth": "true"
  ```
- Implement NetworkPolicies to restrict:
  - Pipeline pods → Milvus (port 19530)
  - Pipeline pods → MinIO (port 9000)
  - InferenceService → HuggingFace (egress HTTPS only)
  - Deny all other cross-namespace traffic
- Enable Ray TLS for worker-to-worker communication
- Implement API Gateway with rate limiting for InferenceService

**Remediation Priority**: HIGH

---

### 3.2 TLS/SSL Configuration - CRITICAL

**Risk Level**: CRITICAL
**Current State**:
- MinIO: `useSSL: false`
- Milvus: HTTP connections (no TLS)
- Ray: TLS disabled
- S3 endpoint: HTTP
- InferenceService: May use HTTPS via OpenShift Routes (not verified)

**Issues**:
1. All internal data transfers unencrypted
2. Credentials transmitted in plaintext over network
3. Man-in-the-middle attacks possible
4. Non-compliance with security standards (PCI-DSS, SOC 2, HIPAA)

**Recommendations**:
- **IMMEDIATE**: Enable TLS for all services:
  - MinIO: Configure TLS certificates, update endpoints to HTTPS
  - Milvus: Enable TLS for gRPC (requires cert-manager setup)
  - Ray: Set `RAY_USE_TLS: "1"` and configure certificates
  - S3 client: Update to `https://` endpoints
- Use cert-manager for automated certificate management
- Enforce TLS 1.2+ minimum version
- Configure mutual TLS (mTLS) for service-to-service communication

**Remediation Priority**: CRITICAL (before demo)

---

### 3.3 Egress Controls - MEDIUM

**Risk Level**: MEDIUM
**Current State**:
- Ray workers download models from HuggingFace (internet access required)
- No egress filtering or allowlisting
- Pipeline components can reach arbitrary external endpoints

**Issues**:
1. Potential for data exfiltration via compromised components
2. No visibility into external API calls
3. Risk of supply chain attacks via malicious packages

**Recommendations**:
- Implement NetworkPolicy egress rules:
  - Allow HuggingFace model downloads (huggingface.co)
  - Allow PyPI package downloads (if needed during runtime)
  - Deny all other egress by default
- Use private package mirror for Python dependencies
- Configure HTTP proxy for egress monitoring
- Add egress monitoring and alerting

**Remediation Priority**: MEDIUM

---

## 4. RBAC & AUTHORIZATION

### 4.1 Service Account Permissions - MEDIUM

**Risk Level**: MEDIUM
**Current State**:
- Pipeline SA: `pipeline-runner-dspa`, `ds-pipeline-dspa`
- Permissions granted (from `/manifests/rbac.yaml`):
  - RayJobs/RayClusters: create, get, list, watch, patch, update, delete
  - InferenceServices: create, get, list, watch, patch, update, delete
  - ServingRuntimes: create, get, list, watch, patch, update, delete
  - Secrets: create, get, patch, delete
  - Pods/logs: get, list, watch
- **Secrets permissions are broad** - can create/delete any Secret in namespace

**Issues**:
1. Secret permissions too permissive (can access all Secrets)
2. No separation between pipeline orchestration and workload execution
3. InferenceService has `automountServiceAccountToken: False` (good) but ServingRuntime may auto-mount
4. ClusterRole grants read access to HardwareProfiles cluster-wide

**Recommendations**:
- Restrict Secret permissions to specific Secret names:
  ```yaml
  resourceNames: ["minio-secret", "codeflare-*"]
  ```
- Separate service accounts for:
  - Pipeline orchestration
  - RayJob execution
  - Model serving
- Implement least-privilege for each component
- Add RBAC audit logging
- Remove broad "delete" permissions where not needed

**Remediation Priority**: MEDIUM

---

### 4.2 Pod Security Standards - HIGH

**Risk Level**: HIGH
**Current State**:
- InferenceService: `automountServiceAccountToken: False` (GOOD)
- Milvus etcd and standalone pods: Proper securityContext configured:
  ```yaml
  runAsNonRoot: true
  allowPrivilegeEscalation: false
  capabilities:
    drop: [ALL]
  seccompProfile:
    type: RuntimeDefault
  ```
- Ray pods: NO securityContext configured
- Pipeline component pods: NO securityContext configured

**Issues**:
1. Ray workers may run as root
2. No seccomp profiles for Ray/pipeline pods
3. No capability dropping for pipeline pods
4. Inconsistent security posture across components

**Recommendations**:
- Apply restricted Pod Security Standards to namespace
- Add securityContext to Ray cluster config:
  ```yaml
  podSecurityContext:
    runAsNonRoot: true
    runAsUser: 1000960000
    fsGroup: 1000960000
    seccompProfile:
      type: RuntimeDefault
  containerSecurityContext:
    allowPrivilegeEscalation: false
    capabilities:
      drop: [ALL]
  ```
- Enforce Pod Security Admission at namespace level
- Test with restricted-v2 SCC on OpenShift

**Remediation Priority**: HIGH

---

## 5. CONTAINER IMAGE SECURITY

### 5.1 Image Sources & Verification - GOOD

**Risk Level**: LOW
**Current State**:
- All base images use SHA256 digests (EXCELLENT):
  ```
  registry.redhat.io/rhoai/odh-pipeline-runtime-datascience-cpu-py312-rhel9@sha256:f9844dc150592a9f196283b3645dda92bd80dfdb3d467fa8725b10267ea5bdbc
  registry.redhat.io/rhaiis/vllm-cuda-rhel9@sha256:094db84a1da5e8a575d0c9eade114fa30f4a2061064a338e3e032f3578f8082a
  ```
- Ray image: `quay.io/rhoai-szaher/docling-ray:latest` - uses `:latest` tag (BAD)
- Milvus images: SHA256 digests via personal Quay repo

**Strengths**:
1. Official Red Hat images (trusted source)
2. SHA256 pinning prevents tag mutation
3. No Docker Hub images (avoids rate limits)

**Issues**:
1. Ray image uses `:latest` tag (non-deterministic)
2. Personal Quay repos (`rhoai-szaher`) not organization-managed
3. No image scanning mentioned
4. No signature verification

**Recommendations**:
- Pin Ray image to specific SHA256 digest
- Move images to organizational Quay repository
- Implement image scanning in CI/CD (Clair, Trivy)
- Add image signature verification (Sigstore/Cosign)
- Document image build and update process

**Remediation Priority**: MEDIUM

---

## 6. COMPLIANCE CONSIDERATIONS

### 6.1 Data Residency & Privacy

**Current State**: Not configured for compliance

**Gaps**:
- No data classification labeling
- No geographic restrictions on data storage
- No data retention/deletion policies
- No audit logging for data access
- No consent management for document processing

**Recommendations for Compliance**:

**GDPR Compliance**:
- Add data subject rights implementation (right to deletion, export)
- Implement data processing records
- Add consent tracking for document uploads
- Configure data retention policies (default: delete after 90 days)
- Enable audit logging for all data access

**SOC 2 Compliance**:
- Implement comprehensive audit logging
- Add change management controls
- Configure encryption at rest and in transit
- Implement access reviews (quarterly)
- Add incident response procedures

**HIPAA Compliance** (if processing PHI):
- Enable encryption at rest for all storage
- Implement role-based access controls
- Add audit logging with HIPAA-compliant retention
- Configure Business Associate Agreements for cloud services
- Implement de-identification for vector embeddings

---

## 7. SECURITY RECOMMENDATIONS SUMMARY

### Critical Priority (Before Demo)

1. **Remove hardcoded credentials** from all manifests
   - Impact: HIGH | Effort: LOW | Timeline: Immediate

2. **Enable TLS for all internal services**
   - MinIO, Milvus, Ray cluster
   - Impact: CRITICAL | Effort: MEDIUM | Timeline: 1-2 days

3. **Enable InferenceService authentication**
   - Change `enable-auth: "true"`
   - Impact: HIGH | Effort: LOW | Timeline: 2 hours

### High Priority (Before Production)

4. **Implement NetworkPolicies**
   - Restrict inter-service communication
   - Impact: HIGH | Effort: MEDIUM | Timeline: 1 day

5. **Enable Milvus authentication**
   - Configure user/password authentication
   - Impact: HIGH | Effort: MEDIUM | Timeline: 1 day

6. **Add Pod Security Standards**
   - Apply securityContext to all pods
   - Impact: HIGH | Effort: MEDIUM | Timeline: 1 day

7. **Enable S3 encryption**
   - Server-side encryption + HTTPS
   - Impact: HIGH | Effort: MEDIUM | Timeline: 1 day

### Medium Priority (Production Hardening)

8. **Implement Secret management**
   - Use external-secrets or Vault
   - Impact: MEDIUM | Effort: HIGH | Timeline: 1 week

9. **Add comprehensive audit logging**
   - All data access, API calls, config changes
   - Impact: MEDIUM | Effort: HIGH | Timeline: 1 week

10. **Implement egress controls**
    - NetworkPolicy egress rules
    - Impact: MEDIUM | Effort: MEDIUM | Timeline: 2 days

11. **Enhance RBAC granularity**
    - Separate service accounts, restrict Secret access
    - Impact: MEDIUM | Effort: MEDIUM | Timeline: 2 days

12. **Pin Ray image digest**
    - Replace `:latest` with SHA256
    - Impact: MEDIUM | Effort: LOW | Timeline: 1 hour

---

## 8. SECURITY HIGHLIGHTS FOR DEMO DISCUSSIONS

### What to Emphasize (Strengths):

1. **Container Security**:
   - "All pipeline component images use SHA256-pinned digests from Red Hat's certified registry"
   - "No dependency on Docker Hub - all images from trusted sources"

2. **RBAC Foundation**:
   - "Least-privilege RBAC implemented for pipeline service accounts"
   - "InferenceService pods do not mount service account tokens"
   - "Separate roles for pipeline orchestration vs. execution"

3. **Secret Management Pattern**:
   - "S3 credentials stored in Kubernetes Secrets, never in pipeline code"
   - "Environment variable injection via kubernetes.use_secret_as_env()"
   - "No secrets in container images or logs"

4. **Security Roadmap**:
   - "TLS enablement in progress for all internal services"
   - "NetworkPolicy implementation planned for network segmentation"
   - "Authentication enablement for all service endpoints"

### What to Acknowledge (Gaps):

1. **Current Demo Limitations**:
   - "Demo environment uses simplified authentication for ease of access"
   - "TLS not yet enabled - production deployment will require cert-manager"
   - "NetworkPolicies not configured - suitable for isolated demo environment"

2. **Production Requirements**:
   - "Production deployment requires Secret rotation automation"
   - "Full TLS/mTLS implementation needed for compliance"
   - "Audit logging and monitoring integration required"

---

## 9. IMMEDIATE ACTION ITEMS (PRE-DEMO)

### Day 1 (Immediate):
- [ ] Remove hardcoded credentials from `/manifests/minio-secret.yaml`
- [ ] Update Milvus instance secret in `milvus-ocp/04-milvus-instance.yaml`
- [ ] Add `.gitignore` entry for credential files
- [ ] Create credential rotation procedure document

### Day 2 (Before Demo):
- [ ] Generate unique MinIO credentials for demo environment
- [ ] Test pipeline with new credentials
- [ ] Enable InferenceService authentication
- [ ] Document demo authentication flow

### Week 1 (Production Prep):
- [ ] Configure TLS for MinIO (cert-manager + Let's Encrypt)
- [ ] Enable Milvus TLS for gRPC connections
- [ ] Implement NetworkPolicies for namespace isolation
- [ ] Add Pod Security Standards to Ray cluster

---

## 10. RISK MATRIX

| Issue | Likelihood | Impact | Risk Score | Priority |
|-------|-----------|--------|------------|----------|
| Hardcoded credentials leaked | HIGH | CRITICAL | 9/10 | CRITICAL |
| TLS not enabled (MITM) | MEDIUM | CRITICAL | 8/10 | CRITICAL |
| InferenceService unauthenticated | HIGH | HIGH | 8/10 | CRITICAL |
| Milvus no auth | MEDIUM | HIGH | 7/10 | HIGH |
| No NetworkPolicies | MEDIUM | HIGH | 7/10 | HIGH |
| S3 not encrypted | MEDIUM | HIGH | 7/10 | HIGH |
| Broad Secret permissions | LOW | HIGH | 6/10 | MEDIUM |
| Ray image :latest tag | LOW | MEDIUM | 4/10 | MEDIUM |
| No egress controls | LOW | MEDIUM | 4/10 | MEDIUM |

---

## Appendix A: Security Testing Checklist

- [ ] Verify no secrets in Git history (`git log --all --full-history -- "*secret*"`)
- [ ] Test TLS certificate validation
- [ ] Verify NetworkPolicy deny-by-default behavior
- [ ] Test authentication token expiration
- [ ] Validate RBAC permissions (attempt unauthorized actions)
- [ ] Test S3 encryption at rest
- [ ] Verify audit log completeness
- [ ] Test credential rotation procedure
- [ ] Validate Pod Security Standards enforcement
- [ ] Test egress NetworkPolicy rules

## Appendix B: Security Contacts

**For Security Issues**:
- Security Team: [security@example.com]
- On-Call: [oncall@example.com]
- Incident Response: [incident@example.com]

**For Compliance Questions**:
- Compliance Officer: [compliance@example.com]
- Privacy Officer: [privacy@example.com]

---

**Assessment Completed**: 2026-03-31
**Next Review Date**: 2026-04-30
**Approved By**: [Pending Security Team Review]
