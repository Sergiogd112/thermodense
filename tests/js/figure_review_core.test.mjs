import assert from "node:assert/strict";
import test from "node:test";

import {
  DECISIONS,
  buildReviewManifest,
  draftStorageKey,
  normalizeImportedManifest,
  normalizeDecision,
  physicalPixels,
  resolveAssetUrl,
} from "../../src/thermodense/figure_review/review-core.mjs";

const figureSet = {
  figureSetVersion: "set-v1",
  figures: [{
    id: "fig-1",
    title: "Figure",
    sha256: "preview",
    publicationSrc: "figures/figure.pdf",
    publicationSha256: "publication",
    publicationFormat: "application/pdf",
    printWidthCm: 8.5,
    claimCardIds: ["claim-1"],
  }],
  claims: [{ id: "claim-1", text: "Claim" }],
  profiles: { agu: { name: "AGU" } },
};

test("legacy supplement imports as appendix and round-trips publication identity", () => {
  const imported = normalizeImportedManifest({
    manifestVersion: "0.4-prototype",
    figureSetVersion: "set-v1",
    profile: "agu",
    figures: [{
      id: "fig-1",
      decision: "supplement",
      comments: [{ type: "caption", text: "Tighten caption" }],
      contentSha256: "preview",
      printWidthCm: 5,
      publication: {
        path: "figures/figure.pdf",
        sha256: "publication",
        format: "application/pdf",
      },
    }],
    claims: [{ id: "claim-1", verdict: "supported" }],
  }, figureSet);

  assert.equal(imported.figures["fig-1"].decision, "appendix");
  assert.equal(imported.printWidths["fig-1"], 5);
  const exported = buildReviewManifest(
    figureSet,
    { profile: imported.profile, figures: imported.figures, claims: imported.claims },
    imported.printWidths,
    "2026-08-05T00:00:00Z",
  );
  assert.equal(exported.manifestVersion, "1.0");
  assert.equal(exported.figures[0].decision, "appendix");
  assert.equal(exported.figures[0].publication.sha256, "publication");
});

test("original 0.1 manifests require explicit cross-version migration", () => {
  const legacy = {
    manifestVersion: "0.1-prototype",
    figureSetVersion: "set-v0",
    profile: "agu",
    figures: [{
      id: "fig-1",
      decision: "supplement",
      comments: [{ type: "scientific", text: "Keep this evidence" }],
      contentSha256: "legacy-preview",
    }],
    claims: [{ id: "claim-1", verdict: "needs-work" }],
  };

  assert.throws(
    () => normalizeImportedManifest(legacy, figureSet),
    /requires explicit figure-set migration/,
  );
  const migrated = normalizeImportedManifest(
    legacy,
    figureSet,
    { allowLegacyMigration: true },
  );
  assert.equal(migrated.figures["fig-1"].decision, "appendix");
  assert.equal(migrated.figures["fig-1"].comments[0].text, "Keep this evidence");
  assert.equal(migrated.migratedFromVersion, "set-v0");
});

test("incompatible figure sets and artifact identities are rejected", () => {
  assert.throws(
    () => normalizeImportedManifest({
      manifestVersion: "1.0",
      figureSetVersion: "set-v2",
      figures: [],
    }, figureSet),
    /figure-set mismatch/,
  );
  assert.throws(
    () => normalizeImportedManifest({
      manifestVersion: "1.0",
      figureSetVersion: "set-v1",
      figures: [{ id: "fig-1", contentSha256: "changed" }],
    }, figureSet),
    /preview identity mismatch/,
  );
  assert.throws(
    () => normalizeImportedManifest({
      manifestVersion: "2.0",
      figureSetVersion: "set-v1",
      figures: [],
    }, figureSet),
    /unsupported manifest version/,
  );
  assert.throws(
    () => normalizeImportedManifest({
      manifestVersion: "1.0",
      figureSetVersion: "set-v1",
      figures: [],
    }, figureSet),
    /every current figure exactly once/,
  );
  assert.throws(
    () => normalizeImportedManifest({
      manifestVersion: "1.0",
      figureSetVersion: "set-v1",
      figures: [{
        id: "fig-1",
        publication: {
          path: "figures/figure.pdf",
          sha256: "publication",
          format: "application/pdf",
        },
      }],
    }, figureSet),
    /preview identity mismatch/,
  );
  assert.throws(
    () => normalizeImportedManifest({
      manifestVersion: "1.0",
      figureSetVersion: "set-v1",
      figures: [{ id: "fig-1", contentSha256: "preview" }],
    }, figureSet),
    /publication identity missing/,
  );
});

test("draft keys, physical sizing, and route-safe assets are deterministic", () => {
  assert.equal(draftStorageKey("set-v1"), "thermodense.figureReview.draft.v1.set-v1");
  assert.ok(Math.abs(physicalPixels(21) - 793.7007874) < 1e-6);
  assert.equal(
    resolveAssetUrl("https://host.example/figure-review/", "figures/figure.png"),
    "https://host.example/figure-review/figures/figure.png",
  );
  assert.throws(() => resolveAssetUrl("https://host.example/figure-review/", "/figure.png"));
});

test("the maintained decision state machine has exactly five states", () => {
  assert.deepEqual(DECISIONS, ["unreviewed", "include", "appendix", "revise", "exclude"]);
  assert.equal(normalizeDecision("supplement"), "appendix");
  assert.equal(normalizeDecision("unknown"), "unreviewed");
});
