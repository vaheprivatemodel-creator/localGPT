/**
 * Native Next.js streaming proxy for the rag-api SSE endpoint.
 *
 * Why this exists: the default `rewrites()` proxy buffers responses in
 * Next.js dev, which means SSE bytes from `localhost:8001/chat/stream` are
 * not flushed to the browser until the upstream connection closes. That makes
 * the UI appear hung even though the rag-api finished long ago.
 *
 * This handler manually forwards the request to the rag-api, then streams the
 * response body back to the client unbuffered. Headers are set so any reverse
 * proxy in front of Next (or buffering middlewares) get the hint to flush.
 */

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

const RAG_API = process.env.RAG_API_URL || 'http://localhost:8001';

export async function POST(req: Request) {
  const body = await req.text();

  const headers: Record<string, string> = {
    'Content-Type': req.headers.get('content-type') || 'application/json',
    Accept: 'text/event-stream',
  };
  const auth = req.headers.get('authorization');
  if (auth) headers.Authorization = auth;

  const upstream = await fetch(`${RAG_API}/chat/stream`, {
    method: 'POST',
    headers,
    body,
    // @ts-expect-error — Node fetch supports duplex but TS types lag.
    duplex: 'half',
  });

  if (!upstream.ok || !upstream.body) {
    return new Response(
      JSON.stringify({ error: `upstream ${upstream.status}` }),
      { status: upstream.status || 502, headers: { 'Content-Type': 'application/json' } },
    );
  }

  return new Response(upstream.body, {
    status: 200,
    headers: {
      'Content-Type': 'text/event-stream; charset=utf-8',
      'Cache-Control': 'no-cache, no-transform',
      Connection: 'keep-alive',
      'X-Accel-Buffering': 'no',
    },
  });
}

export async function OPTIONS() {
  return new Response(null, {
    status: 204,
    headers: {
      'Access-Control-Allow-Origin': '*',
      'Access-Control-Allow-Methods': 'POST, OPTIONS',
      'Access-Control-Allow-Headers': 'Content-Type, Authorization',
    },
  });
}
