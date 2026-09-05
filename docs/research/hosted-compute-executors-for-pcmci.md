# Hosted compute executors for portable PCMCI cases

Date: 2026-09-01

## Question

Which hosted services could supplement Phoenix and Kaggle for the portable
`density-pcmci-v2-saber` analysis bundle?

The useful executor must support unattended CPU-heavy Python work, private
inputs, multi-hour runs, durable job identity, status/cancel operations, and
retrievable output artifacts. The current bundle is about 15 MB and contains
derived inputs rather than raw SABER source files.

## Recommendation

1. **Prototype Hugging Face Jobs first.** It is the closest match to the
   existing executor contract: direct Python and CLI APIs for submit, inspect,
   logs, wait, and cancel; Docker images; mounted input/output storage; explicit
   timeouts; and unusually inexpensive CPU flavors. `cpu-upgrade` currently
   provides 8 vCPU and 32 GB for $0.03/hour. Confirm private bucket behavior and
   outbound-network policy in a disposable proof before authorizing real data.
2. **Prototype Modal second** when automatic fan-out is valuable. It has durable
   function-call IDs, asynchronous submission, polling, logs, cancellation,
   persistent volumes, a $30/month Starter credit, and per-second CPU/memory
   billing. Its 24-hour execution ceiling and default preemption mean jobs must
   be retry-safe; non-preemptible CPU execution costs three times list price.
3. **Use Google Cloud Batch as the durable paid fallback.** It is a true managed
   batch scheduler, has CLI/API control, provisions arbitrary supported VM
   shapes, supports Spot VMs and retries, and can block external network access.
   Batch itself has no surcharge; Compute Engine, storage, logging, and network
   resources are billed normally.
4. **Consider Oracle Cloud Always Free as an opportunistic static worker, not a
   batch backend.** The current free allowance is 2 Ampere A1 OCPUs and 12 GB
   RAM with persistent storage. It could use the existing SSH-style adapter,
   but is Arm-based, capacity can be unavailable, and idle free instances can be
   reclaimed.

Do not prioritize Google Colab or standard GitHub-hosted Actions for the current
unsharded jobs. Colab is interactive and non-guaranteed; Actions has a six-hour
job limit and only 2 vCPU/8 GB for a private repository.

Adding an executor will not resolve the current Phoenix failure: both
`hasdm_all-*` cases reached Tigramite and failed with `ValueError: No valid
samples`. That is an analysis/data-design issue and should be resolved before
spending paid compute.

## Comparison

| Service | Automation | CPU / memory | Runtime | Private artifacts | Cost model | Fit |
|---|---|---|---|---|---|---|
| Hugging Face Jobs | Python API and `hf jobs` submit/list/inspect/log/wait/cancel | 2/16 GB, 8/32 GB, 16/124 GB, 32/256 GB | 30-minute default; caller sets longer seconds/minutes/hours/days timeout; published maximum not found | Hub repositories or buckets can be mounted; local directories can sync to input/output buckets | Per minute; CPU currently $0.01, $0.03, $1.00, or $1.90/hour | **Best first adapter** |
| Modal | Python API/CLI, deployed functions, detached `spawn`, durable call ID, logs/get/cancel | Fractional/configurable CPU and memory | 5-minute default, maximum 24 hours per attempt | Modal Volumes or cloud bucket mounts | Per second; Starter includes $30/month credit | **Best fan-out adapter** |
| Google Cloud Batch | REST API, SDK, `gcloud`; status, logs, cancel, retries | Broad Compute Engine general, compute, and memory-optimized families | Configurable task timeout; suitable for long VM jobs | Cloud Storage, IAM/service accounts, Secret Manager; external access can be blocked | No Batch surcharge; underlying resources billed; Spot available | **Best durable paid fallback** |
| AWS Batch | REST/SDK/CLI; managed queues, job status, logs, cancellation, retries | EC2, Fargate, ECS Managed Instances, and EKS options | No default or maximum Batch timeout; Fargate should not be expected beyond 14 days | S3/IAM/VPC ecosystem | Batch control plane plus underlying AWS resources; EC2 Spot supported | Strong but more setup than HF/Modal |
| Azure Batch | API, CLI, client libraries; pools/jobs/tasks and monitoring | Broad Azure VM catalog, autoscale and Spot nodes | VM/task model supports long jobs | Azure Storage and regional data residency | No Batch surcharge; VM/storage/network billed | Strong, similar complexity to AWS/GCP |
| Oracle Always Free VM | OCI API/CLI plus SSH/systemd worker | 2 Arm OCPUs and 12 GB total; 200 GB combined free block storage | Persistent VM | Private subnet and persistent disk/object storage | Always Free allowance; capacity may be unavailable | Useful free static worker if Arm-compatible |
| Runpod CPU Pod | REST API creates/gets/stops Pods; custom image/command | User-selected CPU flavor, vCPU count, RAM and storage | Pod lifecycle, not a batch-job timeout | Persistent/network volumes; Secure or Community cloud | Per-second compute/storage; interruptible option | Viable, but adapter must implement more lifecycle logic |
| GitHub Actions | Workflow-dispatch REST/CLI, run status/cancel, logs and artifacts | Private repo standard Linux: 2 vCPU, 8 GB RAM, 14 GB SSD | 6 hours per hosted job | Private repository plus workflow artifacts | Free plan: 2,000 min/month and 500 MB artifacts; then per-minute billing | Only after reliable sub-six-hour sharding |
| Google Colab | Notebook UI; no supported unattended batch API identified | Variable and unpublished | Usually at most 12 hours; Pro+ up to 24 hours with sufficient units | Private account VM/Drive, but ephemeral | Free or compute-unit plans | **Reject for scheduler automation** |

## Executor notes

### Hugging Face Jobs

The Hub client exposes `run_job`, `inspect_job`, `list_jobs`,
`fetch_job_logs`, `wait_for_job`, and `cancel_job`; equivalent `hf jobs`
commands are available. A job is a Docker image plus command and hardware
flavor. Mounted repositories and buckets support read-only inputs and writable
outputs, and `sync_job_volume` / `sync_bucket` provide local transfer. This maps
cleanly onto `submit`, `status`, `retrieve`, and `cancel` adapter operations.

Open questions for a proof:

- verify that the automatically created `jobs-artifacts` bucket is private to
  the account by default, or create an explicitly private repository/bucket;
- determine whether outbound internet can be disabled or restricted;
- benchmark Tigramite on `cpu-upgrade`, including whether all eight vCPUs help;
- verify the practical maximum custom timeout and artifact retention.

### Modal

`Function.spawn()` returns a durable `FunctionCall`; its object ID can be saved
and reconstructed later with `FunctionCall.from_id`. Calls expose logs,
non-blocking `get(timeout=0)`, and cancellation. Detached batch invocation and
persistent Volumes fit the current immutable-input/atomic-output design.

Modal Functions are preemptible by default and automatically restart on the
same input. That behavior is acceptable only because the current runner is
case-level idempotent. A 24-hour maximum per attempt is the main risk. If a case
can exceed it, the scientific runner needs valid internal checkpointing or
smaller declared cases before Modal is suitable.

### Managed cloud batch

Google Cloud Batch, AWS Batch, and Azure Batch are the most operationally
complete options. All provide APIs/CLIs, queues, VM-backed jobs, logs, private
object storage, IAM, and lower-cost interruptible capacity. They add account,
billing, IAM, container/image, and network setup absent from Hugging Face Jobs
and Modal.

Google Cloud Batch is the preferred one-cloud proof because its official job
schema supports scripts or containers, task arrays, retries, timeouts, custom
service accounts, and both VM-level and container-level external-network
blocking. AWS Batch is equally capable and has no maximum Batch timeout for EC2
jobs. Azure Batch closely matches traditional HPC pool/job/task semantics.

### Free and CI-oriented options

Oracle Always Free is not elastic batch compute, but a small persistent worker
could be addressed through an OCI or SSH adapter. The free A1 shape is Arm, so
NumPy/SciPy/Tigramite wheel availability and performance must be tested. Oracle
also warns that Always Free capacity can be unavailable and idle instances can
be reclaimed.

GitHub Actions is easy to automate and private-repository inputs can remain
private, but standard private Linux runners are only 2 vCPU/8 GB and stop after
six hours. The monthly free allowance is about 33 runner-hours. It becomes
reasonable only if cases are partitioned into independently valid, resumable
sub-six-hour shards.

Colab explicitly prioritizes active notebook use, does not guarantee resources,
does not publish stable usage limits, and generally limits runtimes to 12 hours.
Those properties conflict with a durable unattended scheduler adapter.

## Suggested adoption sequence

1. Resolve and regression-test the `No valid samples` failure locally.
2. Record peak RSS, wall time, and CPU utilization for one valid selected case.
3. Build a disposable Hugging Face Jobs adapter proof using synthetic data.
4. Require the proof to demonstrate private upload, submit, durable ID, poll,
   logs, cancel, retry without duplicate submission, artifact hash validation,
   and budget/time limits.
5. Compare one identical synthetic case on HF `cpu-upgrade` and Modal.
6. Add only the winning executor to the canonical plan; keep Google Cloud Batch
   as the next option if a case exceeds hosted-service limits.

## Primary sources

- Hugging Face, [Run and manage Jobs](https://huggingface.co/docs/huggingface_hub/en/guides/jobs)
- Hugging Face, [Jobs pricing and billing](https://huggingface.co/docs/hub/jobs-pricing)
- Modal, [Batch processing](https://modal.com/docs/guide/batch-processing)
- Modal, [Timeouts](https://modal.com/docs/guide/timeouts)
- Modal, [Preemption](https://modal.com/docs/guide/preemption)
- Modal, [FunctionCall API](https://modal.com/docs/reference/modal.FunctionCall)
- Modal, [Pricing](https://modal.com/pricing)
- Google Cloud, [Get started with Batch](https://cloud.google.com/batch/docs/get-started)
- Google Cloud, [Batch pricing](https://cloud.google.com/batch/pricing)
- Google Cloud, [Block external access for a Batch job](https://cloud.google.com/batch/docs/job-without-external-access)
- AWS, [What is AWS Batch?](https://docs.aws.amazon.com/batch/latest/userguide/what-is-batch.html)
- AWS, [Job timeouts](https://docs.aws.amazon.com/batch/latest/userguide/job_timeouts.html)
- Azure, [Batch technical overview](https://learn.microsoft.com/en-us/azure/batch/batch-technical-overview)
- Azure, [Batch pricing](https://azure.microsoft.com/en-us/pricing/details/batch/)
- Oracle, [Always Free resources](https://docs.oracle.com/en-us/iaas/Content/FreeTier/freetier_topic-Always_Free_Resources.htm)
- Runpod, [Create a Pod REST API](https://docs.runpod.io/api-reference/pods/POST/pods)
- Runpod, [Pod pricing](https://docs.runpod.io/pods/pricing)
- GitHub, [Actions limits](https://docs.github.com/en/actions/reference/limits)
- GitHub, [Hosted runner specifications](https://docs.github.com/en/actions/reference/runners/github-hosted-runners)
- GitHub, [Actions billing](https://docs.github.com/en/billing/concepts/product-billing/github-actions)
- Google, [Colab FAQ](https://research.google.com/colaboratory/faq.html)
