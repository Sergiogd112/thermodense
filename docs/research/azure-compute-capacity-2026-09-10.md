# Azure compute capacity assessment — 2026-09-10

## Decision

Use the Azure for Students subscription only for a bounded CPU pilot after explicit
deployment approval. Do not move the raw 36-node run there yet. No GPU SKU is
usable under the current quota, and the only exactly priced CPU options are
burstable Poland Central VMs whose memory is below the failed run's observed
18.054 GiB maximum. No resource was created and no research data was uploaded.

## Observed subscription state

- Offer: Azure for Students, active, with Owner access.
- Credit shown in the portal: USD 86 of USD 86 remaining; September cost USD 0.
- Credit expiry shown in the portal: 2027-09-02.
- `Microsoft.Compute` was registered after user authorization.
- Deployment-policy regions: Poland Central, Germany West Central, Denmark East,
  Switzerland North, and Italy North.
- Regional standard-vCPU quota in each checked region: 6, with 0 used at the time
  of the scan.

The capacity snapshot was held only in an isolated temporary Azure CLI directory
and was deleted after logout. Immediately before logout, `az resource list`
returned an empty list. SKU visibility and quota do not guarantee that Azure will
allocate a VM at deployment time.

## CPU findings

Denmark East had 42 quota-and-location-compatible small CPU SKUs with no recorded
location or zone restriction. Relevant four-vCPU options included:

| SKU | vCPU | RAM | Workload fit |
| --- | ---: | ---: | --- |
| `Standard_A4m_v2` | 4 | 32 GiB | Memory-compatible candidate for a bounded raw-run pilot |
| `Standard_DS12_v2` | 4 | 28 GiB | Memory-compatible candidate for a bounded raw-run pilot |
| `Standard_F4s_v2` | 4 | 8 GiB | Insufficient memory for the prior raw run |

The Azure Retail Prices API returned no `Virtual Machines` consumption record for
those exact Denmark East SKUs on 2026-09-10. HDInsight meters returned by broader
queries are a different service and must not be used as VM price estimates. Obtain
an exact portal quote and recheck allocation immediately before any approved pilot.

Poland Central had ten compatible B-series sizes. Zones 1 and 2 were marked
`NotAvailableForSubscription`, so any future allocation must use a permitted zone
or a regional deployment and still pass Azure's deployment-time check.

| SKU | vCPU | RAM | Linux consumption rate | 72 compute-hours | Assessment |
| --- | ---: | ---: | ---: | ---: | --- |
| `Standard_B2as_v2` | 2 | 8 GiB | USD 0.0864/hour | USD 6.22 | Suitable only for light validation |
| `Standard_B4as_v2` | 4 | 16 GiB | USD 0.173/hour | USD 12.46 | Below observed raw-run memory use |

Rates are public retail compute rates, not a bill estimate. Managed disks, network
egress, taxes, and other services are separate. Basv2 uses CPU credits and is
throttled to base performance after credits are exhausted, so it is a poor match
for a sustained multi-day CPU-bound PCMCI fit.

## GPU findings

No checked GPU SKU satisfied both VM-family quota and per-VM vCPU requirements.
Examples included `Standard_NC4as_T4_v3` with family quota 0,
`Standard_NV4as_v4` requiring four vCPUs with family quota 2, and
`Standard_NV6ads_A10_v5` with family quota 0. Denmark East exposed no small GPU
candidate in the scan. A GPU plan therefore requires a different subscription or
approved quota change; it must not be inferred from SKU visibility alone.

Microsoft documents that only paid subscriptions can request quota increases, so
quota expansion is not assumed for this student offer. Low-priority and Spot price
records were excluded from the recommendation because offer support and eviction
behavior were not established for this workload.

## Safe pilot gate

Before creating anything, require all of the following:

1. User approval for the exact region, SKU, maximum runtime, and credit budget.
2. A current portal price and successful deployment-time capacity check.
3. A synthetic or explicitly approved transferable input; no provider-restricted
   source data upload by default.
4. Immutable input/output hashes and a tested checkpoint/restart boundary.
5. Automatic shutdown plus a budget alert. Stop and deallocate the VM after the
   bounded check; storage can continue billing after compute is deallocated.

## Sources

- Azure Retail Prices API, Poland Central `Standard_B2as_v2` and
  `Standard_B4as_v2`, queried 2026-09-10:
  <https://prices.azure.com/api/retail/prices>
- Microsoft Learn, Basv2 CPU-credit behavior and size specifications:
  <https://learn.microsoft.com/azure/virtual-machines/sizes/general-purpose/basv2-series>
- Microsoft Learn, quota-increase prerequisites (paid subscriptions only):
  <https://learn.microsoft.com/azure/quotas/quickstart-increase-quota-portal>
- Microsoft Learn, Azure for Students credit exhaustion and expiry behavior:
  <https://learn.microsoft.com/azure/cost-management-billing/manage/azurestudents-subscription-disabled>
