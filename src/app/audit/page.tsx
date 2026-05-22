"use client"

import { useEffect, useState, useCallback } from "react"
import { chatAPI, AuditEntry } from "@/lib/api"
import { ShieldCheck, Download, Filter, ChevronLeft, ChevronRight, AlertTriangle, CheckCircle, XCircle, FileText } from "lucide-react"
import Link from "next/link"

const PAGE_SIZE = 25

function formatDate(iso: string) {
  if (!iso) return "—"
  return new Date(iso).toLocaleString([], {
    month: "short", day: "numeric", year: "numeric",
    hour: "2-digit", minute: "2-digit",
  })
}

function KbGapBadge({ flagged }: { flagged: number }) {
  if (!flagged) return null
  return (
    <span className="inline-flex items-center gap-1 text-xs bg-yellow-900/50 text-yellow-300 border border-yellow-700 rounded px-1.5 py-0.5">
      <AlertTriangle className="w-3 h-3" /> KB Gap
    </span>
  )
}

function ReviewedBadge({ at, by }: { at: string | null; by: string | null }) {
  if (!at) {
    return (
      <span className="inline-flex items-center gap-1 text-xs bg-gray-800 text-gray-400 border border-gray-700 rounded px-1.5 py-0.5">
        <XCircle className="w-3 h-3" /> Pending
      </span>
    )
  }
  return (
    <span className="inline-flex items-center gap-1 text-xs bg-green-900/50 text-green-300 border border-green-700 rounded px-1.5 py-0.5" title={`by ${by} on ${formatDate(at)}`}>
      <CheckCircle className="w-3 h-3" /> {by}
    </span>
  )
}

function EntryRow({ entry, onReviewed }: { entry: AuditEntry; onReviewed: (id: string) => void }) {
  const [expanded, setExpanded] = useState(false)
  const [reviewing, setReviewing] = useState(false)
  const [reviewed, setReviewed] = useState({ at: entry.reviewed_at, by: entry.reviewed_by })

  const handleReview = async () => {
    if (reviewed.at || reviewing) return
    setReviewing(true)
    try {
      await chatAPI.markReviewed(entry.id)
      const now = new Date().toISOString()
      setReviewed({ at: now, by: "Attorney" })
      onReviewed(entry.id)
    } finally {
      setReviewing(false)
    }
  }

  const hasGap = !!entry.kb_gap_flagged
  const srcDocs = Array.isArray(entry.source_documents) ? entry.source_documents : []

  return (
    <div className={`border-b border-gray-800 ${hasGap ? "border-l-2 border-l-yellow-600" : ""}`}>
      <div
        className="flex items-start gap-3 px-4 py-3 cursor-pointer hover:bg-gray-900/60 transition-colors"
        onClick={() => setExpanded(e => !e)}
      >
        <FileText className="w-4 h-4 mt-0.5 text-gray-500 flex-shrink-0" />

        <div className="flex-1 min-w-0">
          <div className="flex flex-wrap items-center gap-2 mb-1">
            <span className="text-xs text-gray-400">{formatDate(entry.created_at)}</span>
            {entry.used_rag ? (
              <span className="text-xs bg-blue-900/40 text-blue-300 border border-blue-800 rounded px-1.5 py-0.5">RAG</span>
            ) : (
              <span className="text-xs bg-gray-800 text-gray-500 border border-gray-700 rounded px-1.5 py-0.5">Direct</span>
            )}
            <KbGapBadge flagged={entry.kb_gap_flagged} />
            {srcDocs.length > 0 && (
              <span className="text-xs text-gray-500">{srcDocs.length} source{srcDocs.length !== 1 ? "s" : ""}</span>
            )}
          </div>
          <p className="text-sm text-gray-100 font-medium truncate">{entry.user_query}</p>
          {!expanded && (
            <p className="text-xs text-gray-400 mt-0.5 line-clamp-2">{entry.ai_response.slice(0, 200)}{entry.ai_response.length > 200 ? "…" : ""}</p>
          )}
        </div>

        <div className="flex-shrink-0 flex flex-col items-end gap-2">
          <ReviewedBadge at={reviewed.at} by={reviewed.by} />
          {!reviewed.at && (
            <button
              onClick={e => { e.stopPropagation(); handleReview() }}
              disabled={reviewing}
              className="flex items-center gap-1 text-xs px-2 py-1 rounded border border-green-700 text-green-400 hover:bg-green-900/40 transition-colors disabled:opacity-50"
            >
              <ShieldCheck className="w-3 h-3" />
              {reviewing ? "Saving…" : "Mark Reviewed"}
            </button>
          )}
        </div>
      </div>

      {expanded && (
        <div className="px-11 pb-4 space-y-4">
          <div>
            <p className="text-xs font-semibold text-gray-400 uppercase mb-1">Question</p>
            <p className="text-sm text-gray-200 whitespace-pre-wrap">{entry.user_query}</p>
          </div>
          <div>
            <p className="text-xs font-semibold text-gray-400 uppercase mb-1">AI Answer</p>
            <div className="text-sm text-gray-200 whitespace-pre-wrap bg-gray-900 rounded p-3 border border-gray-800 max-h-72 overflow-y-auto">
              {entry.ai_response}
            </div>
          </div>
          {srcDocs.length > 0 && (
            <div>
              <p className="text-xs font-semibold text-gray-400 uppercase mb-1">Sources ({srcDocs.length})</p>
              <div className="space-y-1">
                {srcDocs.slice(0, 5).map((s: any, i: number) => (
                  <div key={i} className="text-xs text-gray-300 bg-gray-900 rounded px-2 py-1 border border-gray-800 truncate">
                    [{i + 1}] {s.document_id?.split("_").slice(1).join("_") || "source"} — {String(s.text || "").slice(0, 120)}…
                  </div>
                ))}
                {srcDocs.length > 5 && <p className="text-xs text-gray-500">+{srcDocs.length - 5} more</p>}
              </div>
            </div>
          )}
          {reviewed.at && (
            <p className="text-xs text-green-400">
              ✓ Reviewed by {reviewed.by} on {formatDate(reviewed.at)}
            </p>
          )}
        </div>
      )}
    </div>
  )
}

export default function AuditPage() {
  const [entries, setEntries] = useState<AuditEntry[]>([])
  const [total, setTotal] = useState(0)
  const [page, setPage] = useState(0)
  const [filterReviewed, setFilterReviewed] = useState<"all" | "yes" | "no">("all")
  const [filterGap, setFilterGap] = useState(false)
  const [loading, setLoading] = useState(true)
  const [pendingCount, setPendingCount] = useState(0)

  const load = useCallback(async () => {
    setLoading(true)
    try {
      const reviewed = filterReviewed === "all" ? null : filterReviewed === "yes"
      const { entries: rows, total: t } = await chatAPI.getAuditLog({
        limit: PAGE_SIZE,
        offset: page * PAGE_SIZE,
        reviewed,
      })
      const filtered = filterGap ? rows.filter(r => r.kb_gap_flagged) : rows
      setEntries(filtered)
      setTotal(t)
      // Count pending reviews for the badge
      const { total: pending } = await chatAPI.getAuditLog({ limit: 1, reviewed: false })
      setPendingCount(pending)
    } finally {
      setLoading(false)
    }
  }, [page, filterReviewed, filterGap])

  useEffect(() => { load() }, [load])

  const totalPages = Math.ceil(total / PAGE_SIZE)

  return (
    <div className="flex flex-col h-screen bg-black text-gray-100">
      {/* Header */}
      <header className="flex items-center gap-4 px-6 py-4 border-b border-gray-800 flex-shrink-0">
        <Link href="/" className="text-gray-400 hover:text-white transition-colors">
          <ChevronLeft className="w-5 h-5" />
        </Link>
        <div className="flex items-center gap-2">
          <ShieldCheck className="w-5 h-5 text-green-400" />
          <h1 className="text-lg font-semibold">Audit Log</h1>
        </div>
        {pendingCount > 0 && (
          <span className="ml-1 bg-yellow-700 text-yellow-100 text-xs rounded-full px-2 py-0.5">
            {pendingCount} pending review
          </span>
        )}
        <div className="flex-1" />
        <a
          href={chatAPI.getAuditExportUrl()}
          download="audit_log.csv"
          className="flex items-center gap-1.5 text-sm px-3 py-1.5 rounded border border-gray-700 text-gray-300 hover:bg-gray-800 transition-colors"
        >
          <Download className="w-4 h-4" /> Export CSV
        </a>
      </header>

      {/* Filters */}
      <div className="flex items-center gap-3 px-6 py-3 border-b border-gray-800 bg-gray-950 flex-shrink-0">
        <Filter className="w-4 h-4 text-gray-500" />
        <span className="text-sm text-gray-400">Filter:</span>
        <div className="flex gap-2">
          {(["all", "no", "yes"] as const).map(v => (
            <button
              key={v}
              onClick={() => { setFilterReviewed(v); setPage(0) }}
              className={`text-xs px-3 py-1 rounded border transition-colors ${
                filterReviewed === v
                  ? "bg-white text-black border-white"
                  : "border-gray-700 text-gray-400 hover:border-gray-500"
              }`}
            >
              {v === "all" ? "All" : v === "no" ? "Pending review" : "Reviewed"}
            </button>
          ))}
        </div>
        <button
          onClick={() => { setFilterGap(g => !g); setPage(0) }}
          className={`flex items-center gap-1 text-xs px-3 py-1 rounded border transition-colors ${
            filterGap
              ? "bg-yellow-900/60 text-yellow-300 border-yellow-700"
              : "border-gray-700 text-gray-400 hover:border-gray-500"
          }`}
        >
          <AlertTriangle className="w-3 h-3" /> KB Gap only
        </button>
        <span className="ml-auto text-xs text-gray-500">{total} total entries</span>
      </div>

      {/* Entries */}
      <div className="flex-1 overflow-y-auto">
        {loading ? (
          <div className="flex items-center justify-center h-32 text-gray-500">Loading…</div>
        ) : entries.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-40 gap-2 text-gray-500">
            <ShieldCheck className="w-8 h-8 opacity-30" />
            <p className="text-sm">No entries found</p>
          </div>
        ) : (
          entries.map(e => (
            <EntryRow key={e.id} entry={e} onReviewed={() => load()} />
          ))
        )}
      </div>

      {/* Pagination */}
      {totalPages > 1 && (
        <div className="flex items-center justify-center gap-4 px-6 py-3 border-t border-gray-800 flex-shrink-0">
          <button
            disabled={page === 0}
            onClick={() => setPage(p => p - 1)}
            className="p-1.5 rounded hover:bg-gray-800 disabled:opacity-30 transition-colors"
          >
            <ChevronLeft className="w-4 h-4" />
          </button>
          <span className="text-sm text-gray-400">
            Page {page + 1} of {totalPages}
          </span>
          <button
            disabled={page >= totalPages - 1}
            onClick={() => setPage(p => p + 1)}
            className="p-1.5 rounded hover:bg-gray-800 disabled:opacity-30 transition-colors"
          >
            <ChevronRight className="w-4 h-4" />
          </button>
        </div>
      )}
    </div>
  )
}
