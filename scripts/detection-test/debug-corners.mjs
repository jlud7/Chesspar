// Debug: visualize redness, connected components, centroids, and the
// bounding hull they imply. Iterates on the corner-detection algo.

import { createCanvas, loadImage } from "canvas";
import { promises as fs } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import sharp from "sharp";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const ROOT = path.resolve(__dirname, "../..");
const PHOTO_DIR = path.join(ROOT, "Test_Photos");
const OUT_DIR = path.join(__dirname, "debug-out");

const ROTATE_QUARTERS = Number(process.env.ROTATE_QUARTERS ?? "1");
const CROP_LEFT = Number(process.env.CROP_LEFT ?? "0.22");

async function loadOriented(filePath) {
  let pipe = sharp(filePath).rotate();
  if (ROTATE_QUARTERS) pipe = pipe.rotate(ROTATE_QUARTERS * 90);
  const rotatedBuf = await pipe.toBuffer();
  const meta = await sharp(rotatedBuf).metadata();
  const W = meta.width;
  const H = meta.height;
  const left = Math.round(W * CROP_LEFT);
  const buf = await sharp(rotatedBuf)
    .extract({ left, top: 0, width: W - left, height: H })
    .toFormat("jpeg")
    .toBuffer();
  return await loadImage(buf);
}

await fs.rm(OUT_DIR, { recursive: true, force: true });
await fs.mkdir(OUT_DIR, { recursive: true });

const photos = (await fs.readdir(PHOTO_DIR))
  .filter((n) => /\.(jpe?g|png)$/i.test(n))
  .sort();

const onlyFirst = process.env.ONLY_FIRST === "1";
const photosToRun = onlyFirst ? photos.slice(0, 3) : photos;

for (const name of photosToRun) {
  const stem = name.replace(/\.[^.]+$/, "");
  const img = await loadOriented(path.join(PHOTO_DIR, name));

  const w = img.width;
  const h = img.height;
  const c = createCanvas(w, h);
  const ctx = c.getContext("2d");
  ctx.drawImage(img, 0, 0);

  const MAX = 512;
  const scale = Math.min(MAX / w, MAX / h, 1);
  const sw = Math.max(8, Math.round(w * scale));
  const sh = Math.max(8, Math.round(h * scale));
  const sc = createCanvas(sw, sh);
  const sctx = sc.getContext("2d");
  sctx.drawImage(c, 0, 0, sw, sh);
  const sd = sctx.getImageData(0, 0, sw, sh).data;

  // Filter: chess-board red is high R, saturated (max-min large). Wood
  // and skin have moderate redness but LOW saturation, so we use both.
  const red = new Uint8Array(sw * sh);
  for (let i = 0, j = 0; i < sd.length; i += 4, j++) {
    const R = sd[i];
    const G = sd[i + 1];
    const B = sd[i + 2];
    const mx = Math.max(R, G, B);
    const mn = Math.min(R, G, B);
    const sat = mx > 0 ? (mx - mn) / mx : 0;
    const rDom = R - Math.max(G, B);
    red[j] = R > 100 && rDom > 50 && sat > 0.5 ? 1 : 0;
  }

  // Pass 1: find the LARGEST connected component in the (unerorded) red
  // mask — that's "all 32 red squares + thin red border line" merged into
  // one blob. Its bounding box approximates the playing area's extent.
  const ccBig = connectedComponents(red, sw, sh);
  let bigLabel = 0;
  let bigSize = 0;
  for (let lbl = 1; lbl < ccBig.sizes.length; lbl++) {
    if (ccBig.sizes[lbl] > bigSize) {
      bigSize = ccBig.sizes[lbl];
      bigLabel = lbl;
    }
  }
  let bigMinX = Infinity,
    bigMaxX = -Infinity,
    bigMinY = Infinity,
    bigMaxY = -Infinity;
  if (bigLabel) {
    for (let y = 0; y < sh; y++) {
      for (let x = 0; x < sw; x++) {
        if (ccBig.labels[y * sw + x] === bigLabel) {
          if (x < bigMinX) bigMinX = x;
          if (x > bigMaxX) bigMaxX = x;
          if (y < bigMinY) bigMinY = y;
          if (y > bigMaxY) bigMaxY = y;
        }
      }
    }
  }
  const blobBbox =
    bigLabel === 0
      ? null
      : { minX: bigMinX, maxX: bigMaxX, minY: bigMinY, maxY: bigMaxY };

  // Pass 2: erode the redness mask so the thin red border line breaks and
  // each chess square becomes its own connected component → centroids.
  const erodeR = Math.max(2, Math.round(Math.min(sw, sh) / 100));
  const eroded = erode(red, sw, sh, erodeR);

  const { labels, sizes } = connectedComponents(eroded, sw, sh);
  const sorted = [...sizes].slice(1).sort((a, b) => b - a);
  const probe = sorted.slice(0, Math.min(20, sorted.length));
  probe.sort((a, b) => a - b);
  const medianBig = probe[Math.floor(probe.length / 2)] || 0;
  const minAccept = Math.max(40, medianBig * 0.3);
  const maxAccept = medianBig * 4;
  const kept = new Set();
  for (let lbl = 1; lbl < sizes.length; lbl++) {
    if (sizes[lbl] >= minAccept && sizes[lbl] <= maxAccept) kept.add(lbl);
  }

  const centroids = new Map();
  for (const lbl of kept) centroids.set(lbl, { x: 0, y: 0, n: 0 });
  for (let y = 0; y < sh; y++) {
    for (let x = 0; x < sw; x++) {
      const lbl = labels[y * sw + x];
      const c = centroids.get(lbl);
      if (c) {
        c.x += x;
        c.y += y;
        c.n++;
      }
    }
  }
  const points = [...centroids.values()].map((c) => ({
    x: c.x / c.n,
    y: c.y / c.n,
    n: c.n,
  }));

  const vis = createCanvas(sw, sh);
  const vctx = vis.getContext("2d");
  const vImg = vctx.createImageData(sw, sh);
  for (let y = 0; y < sh; y++) {
    for (let x = 0; x < sw; x++) {
      const j = y * sw + x;
      const lbl = labels[j];
      const isKept = lbl > 0 && kept.has(lbl);
      const isRed = red[j] === 1;
      const oi = j * 4;
      if (isKept) {
        vImg.data[oi] = 255;
        vImg.data[oi + 1] = 0;
        vImg.data[oi + 2] = 0;
      } else if (isRed) {
        vImg.data[oi] = 80;
        vImg.data[oi + 1] = 60;
        vImg.data[oi + 2] = 60;
      } else {
        vImg.data[oi] = 30;
        vImg.data[oi + 1] = 30;
        vImg.data[oi + 2] = 30;
      }
      vImg.data[oi + 3] = 255;
    }
  }
  vctx.putImageData(vImg, 0, 0);
  vctx.fillStyle = "rgba(0, 255, 0, 1)";
  for (const p of points) {
    vctx.beginPath();
    vctx.arc(p.x, p.y, 3, 0, Math.PI * 2);
    vctx.fill();
  }
  await fs.writeFile(
    path.join(OUT_DIR, `${stem}_components.png`),
    vis.toBuffer("image/png"),
  );

  // Also save a centroids-on-original visualization.
  {
    const cv = createCanvas(w, h);
    const cvctx = cv.getContext("2d");
    cvctx.drawImage(img, 0, 0);
    for (const p of points) {
      cvctx.fillStyle = "rgba(0, 255, 0, 0.95)";
      cvctx.beginPath();
      cvctx.arc(p.x / scale, p.y / scale, 8, 0, Math.PI * 2);
      cvctx.fill();
    }
    await fs.writeFile(
      path.join(OUT_DIR, `${stem}_centroids.png`),
      cv.toBuffer("image/png"),
    );
  }

  // Fit a chess-board grid to the centroids and read corners off the fit.
  const grid = fitGrid(points, blobBbox);

  const oc = createCanvas(w, h);
  const octx = oc.getContext("2d");
  octx.drawImage(img, 0, 0);
  if (grid) {
    const corners = grid.cornersInDownscaledSpace;
    octx.strokeStyle = "rgba(0, 255, 255, 0.95)";
    octx.lineWidth = Math.max(4, w / 200);
    octx.beginPath();
    for (let i = 0; i < 4; i++) {
      const p = corners[i];
      const X = p.x / scale;
      const Y = p.y / scale;
      if (i === 0) octx.moveTo(X, Y);
      else octx.lineTo(X, Y);
    }
    octx.closePath();
    octx.stroke();
    const lbls = ["TL", "TR", "BR", "BL"];
    for (let i = 0; i < 4; i++) {
      const p = corners[i];
      const X = p.x / scale;
      const Y = p.y / scale;
      octx.fillStyle = "rgba(0,255,255,1)";
      octx.beginPath();
      octx.arc(X, Y, Math.max(10, w / 80), 0, Math.PI * 2);
      octx.fill();
      octx.font = `${Math.max(20, w / 30)}px sans-serif`;
      octx.fillStyle = "white";
      octx.strokeStyle = "black";
      octx.lineWidth = 4;
      octx.strokeText(lbls[i], X + 12, Y - 12);
      octx.fillText(lbls[i], X + 12, Y - 12);
    }

    // Also draw all 9x9 grid intersections for verification.
    octx.fillStyle = "rgba(255, 255, 0, 0.7)";
    for (let f = 0; f <= 8; f++) {
      for (let r = 0; r <= 8; r++) {
        const p = grid.boardToImage(f, r);
        octx.beginPath();
        octx.arc(p.x / scale, p.y / scale, 4, 0, Math.PI * 2);
        octx.fill();
      }
    }
  }
  await fs.writeFile(
    path.join(OUT_DIR, `${stem}_quad.png`),
    oc.toBuffer("image/png"),
  );

  console.log(
    `${stem}: kept ${kept.size} components, grid=${grid ? "fit" : "fail"}`,
  );
}

// Find the rotation that puts the centroid cloud on an axis-aligned
// integer grid. Score: number of centroid pairs whose Δu and Δv are both
// near integer multiples of sqw (so they sit on a shared row or column).
function findGridAngleRobust(centroids, meanX, meanY) {
  // Estimate sqw from nearest-neighbour distances (diag of red squares ≈
  // sqrt(2) sqw, but the centroids in adjacent rows of red are diag, so
  // the smallest NN distance is sqrt(2) * sqw).
  let sumMinD = 0;
  let countMin = 0;
  for (let i = 0; i < centroids.length; i++) {
    let best = Infinity;
    for (let j = 0; j < centroids.length; j++) {
      if (i === j) continue;
      const dx = centroids[i].x - centroids[j].x;
      const dy = centroids[i].y - centroids[j].y;
      const d = Math.hypot(dx, dy);
      if (d > 1 && d < best) best = d;
    }
    if (best < Infinity) {
      sumMinD += best;
      countMin++;
    }
  }
  const meanNN = countMin > 0 ? sumMinD / countMin : 50;
  const sqw = meanNN / Math.SQRT2;

  // Scan θ in [-30°, 30°] and pick the angle that produces the most
  // INLIER centroids (centroids whose distance to the nearest grid
  // intersection is < sqw/4).
  let bestTheta = 0;
  let bestInliers = -Infinity;
  for (let degTwentieth = -200; degTwentieth <= 200; degTwentieth++) {
    const theta = (degTwentieth / 20) * (Math.PI / 180); // 0.05° steps
    const inliers = countGridInliers(centroids, meanX, meanY, theta, sqw);
    if (inliers > bestInliers) {
      bestInliers = inliers;
      bestTheta = theta;
    }
  }
  return bestTheta;
}

function countGridInliers(centroids, meanX, meanY, theta, sqw) {
  const cosT = Math.cos(theta);
  const sinT = Math.sin(theta);
  // Find the U-offset and V-offset (origin) that maximises inliers.
  const us = [],
    vs = [];
  for (const c of centroids) {
    const dx = c.x - meanX;
    const dy = c.y - meanY;
    us.push(dx * cosT + dy * sinT);
    vs.push(-dx * sinT + dy * cosT);
  }
  // For each U, residual fraction r_u = (u / sqw) mod 1 in [-0.5, 0.5].
  // Histogram the residuals into 20 bins and find the peak — that's the
  // best origin offset. Inliers are those within 0.15 of the peak.
  function inliersAxis(values) {
    const BINS = 20;
    const hist = new Array(BINS).fill(0);
    for (const v of values) {
      const f = ((v / sqw) % 1 + 1) % 1; // [0, 1)
      hist[Math.min(BINS - 1, Math.floor(f * BINS))]++;
    }
    let bestBin = 0;
    let bestCount = 0;
    for (let i = 0; i < BINS; i++) {
      if (hist[i] > bestCount) {
        bestCount = hist[i];
        bestBin = i;
      }
    }
    const peakF = (bestBin + 0.5) / BINS;
    let inliers = 0;
    for (const v of values) {
      const f = ((v / sqw) % 1 + 1) % 1;
      const d = Math.min(Math.abs(f - peakF), 1 - Math.abs(f - peakF));
      if (d < 0.15) inliers++;
    }
    return inliers;
  }
  return inliersAxis(us) + inliersAxis(vs);
}

// Fit a chess board to the detected red-square centroids, with the
// playing-area bbox from the largest red blob as an anchor for the
// edges (since heavy piece occlusion can hide an entire rank/file from
// the centroid signal).
function fitGrid(centroids, blobBbox) {
  if (centroids.length < 6) return null;

  const meanX = avg(centroids.map((c) => c.x));
  const meanY = avg(centroids.map((c) => c.y));
  // Find the rotation that minimises the residual when snapping centroids
  // to an axis-aligned grid. Score = number of inlier centroids whose
  // (U, V) both land near integer multiples of sqw.
  const theta = findGridAngleRobust(centroids, meanX, meanY);
  const cosT = Math.cos(theta);
  const sinT = Math.sin(theta);
  const uvPoints = centroids.map((c) => {
    const dx = c.x - meanX;
    const dy = c.y - meanY;
    return {
      u: dx * cosT + dy * sinT,
      v: -dx * sinT + dy * cosT,
    };
  });

  // 2) Estimate the square width from median nearest-neighbour distance.
  //    In any single row, adjacent red squares are spaced by 2 * sqw.
  //    Across rows, the same file's red squares are 2 * sqw apart in Y.
  //    The NEAREST neighbours of a red square are the diagonal red squares
  //    at 1*sqw in both X and Y → distance sqrt(2) * sqw.
  const nnDistances = [];
  for (let i = 0; i < uvPoints.length; i++) {
    let best = Infinity;
    for (let j = 0; j < uvPoints.length; j++) {
      if (i === j) continue;
      const du = uvPoints[i].u - uvPoints[j].u;
      const dv = uvPoints[i].v - uvPoints[j].v;
      const d = Math.hypot(du, dv);
      if (d < best) best = d;
    }
    if (best < Infinity) nnDistances.push(best);
  }
  nnDistances.sort((a, b) => a - b);
  const nnMedian = nnDistances[Math.floor(nnDistances.length / 2)];
  // nnMedian is distance to nearest diagonal red square = sqrt(2) * sqw.
  const sqw = nnMedian / Math.SQRT2;

  // 3) Build a histogram of U coordinates with bins of width sqw/3.
  //    The peaks correspond to file positions. Snap each centroid's U to
  //    the nearest file center, then average to refine.
  const uVals = uvPoints.map((p) => p.u);
  const vVals = uvPoints.map((p) => p.v);
  const uMin = Math.min(...uVals);
  const uMax = Math.max(...uVals);
  const vMin = Math.min(...vVals);
  const vMax = Math.max(...vVals);

  // Each centroid is at u = (file + 0.5) * sqw + offset for some file.
  // We don't know which integer file each centroid corresponds to. But
  // they're spaced by integer multiples of sqw. Find the modular offset
  // (origin) that best fits all centroids.
  function residualForOrigin(uOrigin) {
    let total = 0;
    for (const u of uVals) {
      const k = Math.round((u - uOrigin) / sqw);
      const fit = uOrigin + k * sqw;
      const d = u - fit;
      total += d * d;
    }
    return total;
  }
  // Search uOrigin in [-sqw/2, sqw/2] relative to uMin to find best fit.
  let bestUOrigin = uMin;
  let bestUResid = Infinity;
  const STEPS = 40;
  for (let i = 0; i < STEPS; i++) {
    const candidate = uMin - sqw / 2 + (sqw * i) / STEPS;
    const r = residualForOrigin(candidate);
    if (r < bestUResid) {
      bestUResid = r;
      bestUOrigin = candidate;
    }
  }
  // bestUOrigin is now "u position of file index 0" for whichever file
  // is the leftmost-detected. Determine the actual leftmost integer index
  // (could be 0 or higher), then snap.
  // Each centroid's U has integer step k from bestUOrigin: u = bestUOrigin + k * sqw.
  // The minimum k across centroids gives us the leftmost integer index used.
  // Pick the inlier window of 8 consecutive integer K's that captures the
  // most centroids — outliers (label noise) end up outside.
  const allKsU = uVals.map((u) => Math.round((u - bestUOrigin) / sqw));
  const { startK: minK, endK: maxK } = pickBestKWindow(allKsU);
  if (maxK - minK > 7) {
    return null;
  }
  const uniqueFiles = maxK - minK + 1;
  const missingFilesCount = 8 - uniqueFiles;
  // Decide which side files are missing on. Anchor to the BLOB BBOX:
  // blob's leftmost red pixel is at the LEFT EDGE of the leftmost visible
  // file's square. Compute how far that is from the leftmost detected
  // centroid (in sqw units).
  let leftFileOffset = 0;
  if (blobBbox) {
    const blobLeftU =
      (blobBbox.minX - meanX) * cosT + (blobBbox.minY - meanY) * sinT;
    const blobLeftRight =
      (blobBbox.maxX - meanX) * cosT + (blobBbox.maxY - meanY) * sinT;
    const blobLeftEnd = Math.min(blobLeftU, blobLeftRight);
    const xRaw = minK - (blobLeftEnd - bestUOrigin) / sqw - 0.5;
    leftFileOffset = Math.max(0, Math.min(missingFilesCount, Math.round(xRaw)));
  }
  // Whatever the bbox doesn't account for, distribute across the OPPOSITE
  // edge — but never let the playing area shift the centroids out of the
  // 0..7 file range.
  const leftFileK = minK - leftFileOffset;
  const leftU = bestUOrigin + leftFileK * sqw - 0.5 * sqw;
  const rightU = leftU + 8 * sqw;

  // Same for V axis.
  function residualForOriginV(vOrigin) {
    let total = 0;
    for (const v of vVals) {
      const k = Math.round((v - vOrigin) / sqw);
      const fit = vOrigin + k * sqw;
      const d = v - fit;
      total += d * d;
    }
    return total;
  }
  let bestVOrigin = vMin;
  let bestVResid = Infinity;
  for (let i = 0; i < STEPS; i++) {
    const candidate = vMin - sqw / 2 + (sqw * i) / STEPS;
    const r = residualForOriginV(candidate);
    if (r < bestVResid) {
      bestVResid = r;
      bestVOrigin = candidate;
    }
  }
  const allKsV = vVals.map((v) => Math.round((v - bestVOrigin) / sqw));
  const { startK: minKv, endK: maxKv } = pickBestKWindow(allKsV);
  if (maxKv - minKv > 7) {
    return null;
  }
  const uniqueRanks = maxKv - minKv + 1;
  const missingRanksCount = 8 - uniqueRanks;
  // For piece-occluded boards in starting position, the heavy black back
  // rank obliterates redness in rank 8 (top) more than the white back
  // rank does in rank 1 (bottom). Default: missing ranks are at the TOP.
  // Also use the blob bbox: if blob extends past the topmost centroid by
  // more than 0.5 sqw, it means the blob includes obscured-but-partial
  // squares of rank 8 → no extra offset needed.
  let topRankOffset = missingRanksCount;
  if (blobBbox) {
    const blobTopV =
      -(blobBbox.minX - meanX) * sinT + (blobBbox.minY - meanY) * cosT;
    const blobBottomV =
      -(blobBbox.maxX - meanX) * sinT + (blobBbox.maxY - meanY) * cosT;
    const blobTopEnd = Math.min(blobTopV, blobBottomV);
    const yRaw = minKv - (blobTopEnd - bestVOrigin) / sqw - 0.5;
    const bboxBasedOffset = Math.max(0, Math.round(yRaw));
    // Take the LARGER offset — if either signal indicates missing ranks
    // at the top, honour it.
    topRankOffset = Math.max(topRankOffset, bboxBasedOffset);
    topRankOffset = Math.min(topRankOffset, missingRanksCount);
  }
  const topRankK = minKv - topRankOffset;
  const topV = bestVOrigin + topRankK * sqw - 0.5 * sqw;
  const bottomV = topV + 8 * sqw;

  // Convert the 4 corners (leftU/topV, rightU/topV, ...) back to image
  // coordinates by undoing the rotation and centering.
  function uvToImage(u, v) {
    const dx = u * cosT - v * sinT;
    const dy = u * sinT + v * cosT;
    return { x: dx + meanX, y: dy + meanY };
  }
  const tl = uvToImage(leftU, topV);
  const tr = uvToImage(rightU, topV);
  const br = uvToImage(rightU, bottomV);
  const bl = uvToImage(leftU, bottomV);

  function boardToImage(f, r) {
    // (f, r) in [0, 8] mapping to playing area:
    // f=0 -> leftU, f=8 -> rightU, linearly.
    // r=0 -> topV, r=8 -> bottomV.
    const u = leftU + (f / 8) * (rightU - leftU);
    const v = topV + (r / 8) * (bottomV - topV);
    return uvToImage(u, v);
  }

  return {
    cornersInDownscaledSpace: [tl, tr, br, bl],
    boardToImage,
    sqw,
  };
}

// Find the 8-wide window of integer K values that contains the most points.
// Returns the [startK, endK] inclusive range (endK = startK + 7 if 8+ points
// span at least 8 unique K's; otherwise a smaller window of the actual span).
function pickBestKWindow(ks) {
  if (ks.length === 0) return { startK: 0, endK: 0 };
  const sortedKs = [...ks].sort((a, b) => a - b);
  const counts = new Map();
  for (const k of sortedKs) counts.set(k, (counts.get(k) ?? 0) + 1);
  const uniqueKs = [...counts.keys()].sort((a, b) => a - b);
  let bestStart = uniqueKs[0];
  let bestCount = 0;
  for (const startK of uniqueKs) {
    let count = 0;
    for (let dk = 0; dk < 8; dk++) {
      count += counts.get(startK + dk) ?? 0;
    }
    if (count > bestCount) {
      bestCount = count;
      bestStart = startK;
    }
  }
  // Tighten to the actually-occupied window inside [bestStart, bestStart+7].
  let actualMin = Infinity;
  let actualMax = -Infinity;
  for (let dk = 0; dk < 8; dk++) {
    if (counts.has(bestStart + dk)) {
      actualMin = Math.min(actualMin, bestStart + dk);
      actualMax = Math.max(actualMax, bestStart + dk);
    }
  }
  if (actualMin === Infinity) return { startK: bestStart, endK: bestStart };
  return { startK: actualMin, endK: actualMax };
}

function avg(xs) {
  let s = 0;
  for (const x of xs) s += x;
  return s / xs.length;
}

function convexHull(points) {
  if (points.length < 3) return [...points];
  const sorted = [...points].sort((a, b) => a.x - b.x || a.y - b.y);
  const lower = [];
  for (const p of sorted) {
    while (
      lower.length >= 2 &&
      cross(lower[lower.length - 2], lower[lower.length - 1], p) <= 0
    ) {
      lower.pop();
    }
    lower.push(p);
  }
  const upper = [];
  for (let i = sorted.length - 1; i >= 0; i--) {
    const p = sorted[i];
    while (
      upper.length >= 2 &&
      cross(upper[upper.length - 2], upper[upper.length - 1], p) <= 0
    ) {
      upper.pop();
    }
    upper.push(p);
  }
  lower.pop();
  upper.pop();
  return lower.concat(upper);
}

function cross(o, a, b) {
  return (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x);
}

function simplifyToQuadrilateral(hull) {
  if (hull.length <= 4) return [...hull];
  const pts = [...hull];
  while (pts.length > 4) {
    let bestIdx = 0;
    let bestLoss = Infinity;
    for (let i = 0; i < pts.length; i++) {
      const prev = pts[(i - 1 + pts.length) % pts.length];
      const next = pts[(i + 1) % pts.length];
      const tri = Math.abs(
        prev.x * (pts[i].y - next.y) +
          pts[i].x * (next.y - prev.y) +
          next.x * (pts[i].y - prev.y),
      );
      if (tri < bestLoss) {
        bestLoss = tri;
        bestIdx = i;
      }
    }
    pts.splice(bestIdx, 1);
  }
  return pts;
}

function orderClockwise(quad) {
  if (quad.length !== 4) return quad;
  const cx = quad.reduce((s, p) => s + p.x, 0) / 4;
  const cy = quad.reduce((s, p) => s + p.y, 0) / 4;
  const withAngle = quad.map((p) => ({
    p,
    angle: Math.atan2(p.y - cy, p.x - cx),
  }));
  withAngle.sort((a, b) => a.angle - b.angle);
  return [withAngle[0].p, withAngle[1].p, withAngle[2].p, withAngle[3].p];
}

function erode(mask, w, h, r) {
  const out = new Uint8Array(w * h);
  for (let y = r; y < h - r; y++) {
    for (let x = r; x < w - r; x++) {
      let allSet = 1;
      for (let dy = -r; dy <= r && allSet; dy++) {
        for (let dx = -r; dx <= r; dx++) {
          if (mask[(y + dy) * w + (x + dx)] === 0) {
            allSet = 0;
            break;
          }
        }
      }
      out[y * w + x] = allSet;
    }
  }
  return out;
}

function dilate(mask, w, h, r) {
  const out = new Uint8Array(w * h);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      if (mask[y * w + x] === 0) continue;
      const y0 = Math.max(0, y - r);
      const y1 = Math.min(h - 1, y + r);
      const x0 = Math.max(0, x - r);
      const x1 = Math.min(w - 1, x + r);
      for (let yy = y0; yy <= y1; yy++) {
        const row = yy * w;
        for (let xx = x0; xx <= x1; xx++) out[row + xx] = 1;
      }
    }
  }
  return out;
}

function connectedComponents(mask, w, h) {
  const labels = new Int32Array(w * h);
  const sizes = [0];
  let next = 1;
  const stack = [];
  for (let i = 0; i < mask.length; i++) {
    if (mask[i] === 0 || labels[i] !== 0) continue;
    const label = next++;
    sizes.push(0);
    stack.push(i);
    while (stack.length > 0) {
      const idx = stack.pop();
      if (labels[idx] !== 0) continue;
      labels[idx] = label;
      sizes[label]++;
      const x = idx % w;
      const y = (idx - x) / w;
      if (x > 0 && mask[idx - 1] && labels[idx - 1] === 0) stack.push(idx - 1);
      if (x < w - 1 && mask[idx + 1] && labels[idx + 1] === 0)
        stack.push(idx + 1);
      if (y > 0 && mask[idx - w] && labels[idx - w] === 0) stack.push(idx - w);
      if (y < h - 1 && mask[idx + w] && labels[idx + w] === 0)
        stack.push(idx + w);
    }
  }
  return { labels, sizes };
}

console.log("Done");
