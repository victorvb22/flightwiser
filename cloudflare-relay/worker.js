/**
 * Restricted relay for the Flightwiser backend's OpenSky calls.
 *
 * Why this exists: Render's outbound network cannot reach opensky-network.org
 * at all — confirmed a full connection timeout (never even completes a TCP
 * handshake) on both opensky-network.org and auth.opensky-network.org, while
 * the exact same hosts respond normally from a normal connection. This looks
 * like a Render-specific IP-range block (documented: OpenSky blocks known
 * hyperscaler/cloud-hosting ranges to fight abuse) rather than a general
 * outage. Verified empirically that a request routed through Cloudflare's
 * network reaches both hosts successfully — this Worker just sits in that
 * gap, deployed once, for free (Workers free tier: 100k requests/day).
 *
 * Deliberately NOT a general-purpose open proxy: only these two fixed
 * OpenSky hosts are reachable through it, and every request must carry the
 * right secret header — so even if this Worker's URL leaks (it's referenced,
 * not the secret, in the backend's committed config), it's not useful as a
 * public anonymizing relay for anything else.
 *
 * Setup (Cloudflare dashboard):
 *   1. Workers & Pages -> Create -> paste this file's contents in the editor.
 *   2. Settings -> Variables -> add RELAY_SECRET (mark it as a secret, not a
 *      plain text var) -> any long random string.
 *   3. Set the SAME value as OPENSKY_RELAY_SECRET on the backend host
 *      (Render), and the Worker's own URL as OPENSKY_RELAY_URL there.
 */

const TARGETS = {
  "/relay-auth": "https://auth.opensky-network.org",
  "/relay-api": "https://opensky-network.org",
};

export default {
  async fetch(request, env) {
    if (request.headers.get("X-Relay-Secret") !== env.RELAY_SECRET) {
      return new Response("Forbidden", { status: 403 });
    }

    const url = new URL(request.url);
    const prefix = Object.keys(TARGETS).find((p) => url.pathname.startsWith(p));
    if (!prefix) {
      return new Response("Not found", { status: 404 });
    }

    const targetUrl = TARGETS[prefix] + url.pathname.slice(prefix.length) + url.search;

    const headers = new Headers(request.headers);
    headers.delete("host");
    headers.delete("x-relay-secret");

    const upstream = await fetch(targetUrl, {
      method: request.method,
      headers,
      body: ["GET", "HEAD"].includes(request.method) ? undefined : await request.arrayBuffer(),
    });

    return new Response(upstream.body, {
      status: upstream.status,
      headers: upstream.headers,
    });
  },
};
