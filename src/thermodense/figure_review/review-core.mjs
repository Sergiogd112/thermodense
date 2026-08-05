export const MANIFEST_VERSION = "1.0";
export const DECISIONS = ["unreviewed", "include", "appendix", "revise", "exclude"];
export const COMMENT_TYPES = ["scientific", "limitation", "caption", "presentation"];
export const VERDICTS = ["supported", "unsupported", "needs-work", "not-assessed"];
export const CSS_PX_PER_CM = 96 / 2.54;

export function normalizeDecision(value) {
  const migrated = value === "supplement" ? "appendix" : value;
  return DECISIONS.includes(migrated) ? migrated : "unreviewed";
}

export function draftStorageKey(figureSetVersion) {
  if (!figureSetVersion) throw new Error("figure-set version is required");
  return `thermodense.figureReview.draft.v1.${figureSetVersion}`;
}

export function physicalPixels(cm, scale = 1) {
  if (!Number.isFinite(cm) || !Number.isFinite(scale) || cm < 0 || scale <= 0) {
    throw new Error("physical dimensions must be finite and non-negative");
  }
  return cm * CSS_PX_PER_CM * scale;
}

export function resolveAssetUrl(documentUrl, assetPath) {
  if (!assetPath || assetPath.startsWith("/") || /^[a-z]+:/i.test(assetPath)) {
    throw new Error("asset paths must be relative to the workbench route");
  }
  return new URL(assetPath, new URL("./", documentUrl)).href;
}

function normalizeComment(comment, figureId) {
  const level = comment?.level === "panel" ? "panel" : "figure";
  return {
    level,
    target: String(comment?.target ?? figureId),
    type: COMMENT_TYPES.includes(comment?.type) ? comment.type : "scientific",
    text: String(comment?.text ?? ""),
    createdAt: String(comment?.createdAt ?? new Date().toISOString()),
  };
}

function verifyPublicationIdentity(imported, current) {
  if (imported.contentSha256 !== current.sha256) {
    throw new Error(`preview identity mismatch for ${current.id}`);
  }
  if (!current.publicationSrc && imported.publication) {
    throw new Error(`unexpected publication identity for ${current.id}`);
  }
  if (!current.publicationSrc) return;
  if (!imported.publication) {
    throw new Error(`publication identity missing for ${current.id}`);
  }
  if (
    imported.publication.path !== current.publicationSrc ||
    imported.publication.sha256 !== current.publicationSha256 ||
    imported.publication.format !== current.publicationFormat
  ) {
    throw new Error(`publication identity mismatch for ${current.id}`);
  }
}

export function normalizeImportedManifest(manifest, figureSet) {
  if (!manifest || !Array.isArray(manifest.figures)) {
    throw new Error("manifest has no figures array");
  }
  const supportedVersion =
    manifest.manifestVersion === MANIFEST_VERSION ||
    /^0\.[2-4]-prototype$/.test(manifest.manifestVersion ?? "");
  if (!supportedVersion) {
    throw new Error(`unsupported manifest version: ${manifest.manifestVersion ?? "none"}`);
  }
  if (manifest.figureSetVersion !== figureSet.figureSetVersion) {
    throw new Error(
      `figure-set mismatch: expected ${figureSet.figureSetVersion}, received ${manifest.figureSetVersion ?? "none"}`,
    );
  }

  const importedFigures = new Map(manifest.figures.map((figure) => [figure.id, figure]));
  const currentIds = new Set(figureSet.figures.map((figure) => figure.id));
  if (
    importedFigures.size !== figureSet.figures.length ||
    manifest.figures.length !== figureSet.figures.length ||
    [...importedFigures.keys()].some((id) => !currentIds.has(id))
  ) {
    throw new Error("manifest must contain every current figure exactly once");
  }
  const figures = {};
  const printWidths = {};
  for (const current of figureSet.figures) {
    const imported = importedFigures.get(current.id);
    verifyPublicationIdentity(imported, current);
    figures[current.id] = {
      decision: normalizeDecision(imported.decision),
      comments: Array.isArray(imported.comments)
        ? imported.comments.map((comment) => normalizeComment(comment, current.id))
        : [],
    };
    printWidths[current.id] =
      Number.isFinite(imported.printWidthCm) && imported.printWidthCm >= 1 && imported.printWidthCm <= 30
        ? imported.printWidthCm
        : Number(current.printWidthCm) || 8.5;
  }

  const importedClaims = new Map((manifest.claims ?? []).map((claim) => [claim.id, claim]));
  const claims = {};
  for (const current of figureSet.claims) {
    const verdict = importedClaims.get(current.id)?.verdict;
    claims[current.id] = { verdict: VERDICTS.includes(verdict) ? verdict : "not-assessed" };
  }

  return {
    profile: figureSet.profiles[manifest.profile] ? manifest.profile : Object.keys(figureSet.profiles)[0],
    figures,
    claims,
    printWidths,
  };
}

export function buildReviewManifest(figureSet, review, printWidths, exportedAt = new Date().toISOString()) {
  return {
    manifestVersion: MANIFEST_VERSION,
    figureSetVersion: figureSet.figureSetVersion,
    profile: review.profile,
    exportedAt,
    figures: figureSet.figures.map((figure) => ({
      id: figure.id,
      title: figure.title,
      decision: normalizeDecision(review.figures[figure.id].decision),
      comments: review.figures[figure.id].comments.map((comment) => ({ ...comment })),
      claimCardIds: figure.claimCardIds,
      contentSha256: figure.sha256,
      printWidthCm: printWidths[figure.id],
      publication: figure.publicationSrc
        ? {
            format: figure.publicationFormat,
            path: figure.publicationSrc,
            sha256: figure.publicationSha256,
          }
        : null,
    })),
    claims: figureSet.claims.map((claim) => ({
      id: claim.id,
      verdict: review.claims[claim.id].verdict,
      text: claim.text,
    })),
  };
}
