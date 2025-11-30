import { useState } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'

interface Source {
  url: string
  title: string
  section: string | null
  snippet: string
  char_start: number
  char_end: number
  score: number
}

interface ChatResponse {
  answer_text: string
  sources: Source[]
  confidence: 'low' | 'medium' | 'high'
  query_embedding_similarity: number[]
  follow_up_questions?: string[]
}

interface Message {
  id: number
  role: 'user' | 'assistant'
  content: string
  sources?: Source[]
  follow_up_questions?: string[]
}

const SUGGESTION_CHIPS = [
  {
    text: 'Find IRS Forms (1040, W‑2, 1099)',
    // rgb(67,166,246)
    color: 'from-[#43A6F6] to-[#2E8FE3]',
    bgColor: 'bg-[#43A6F6]',
    hoverColor: 'hover:from-[#2E8FE3] hover:to-[#257ACB]',
  },
  {
    text: 'Deduction & credit guidance',
    // rgb(177,102,142)
    color: 'from-[#B1668E] to-[#9B4F77]',
    bgColor: 'bg-[#B1668E]',
    hoverColor: 'hover:from-[#9B4F77] hover:to-[#844263]',
  },
  {
    text: 'Corporate filing help',
    // rgb(240,162,78)
    color: 'from-[#F0A24E] to-[#DC8E3B]',
    bgColor: 'bg-[#F0A24E]',
    hoverColor: 'hover:from-[#DC8E3B] hover:to-[#C97F33]',
  },
  {
    text: 'Estimate quarterly tax payments',
    // rgb(111,168,116)
    color: 'from-[#6FA874] to-[#5B9362]',
    bgColor: 'bg-[#6FA874]',
    hoverColor: 'hover:from-[#5B9362] hover:to-[#4E8255]',
  },
  {
    text: ' What is the Child Tax Credit?',
    // rgb(124,115,184)
    color: 'from-[#7C73B8] to-[#675EA4]',
    bgColor: 'bg-[#7C73B8]',
    hoverColor: 'hover:from-[#675EA4] hover:to-[#5B5497]',
  },
  {
    text: '1099 vs W‑2 rules',
    // rgb(102,166,152)
    color: 'from-[#66A698] to-[#4F907F]',
    bgColor: 'bg-[#66A698]',
    hoverColor: 'hover:from-[#4F907F] hover:to-[#3E7E6E]',
  },
]

export function App() {
  const [query, setQuery] = useState('')
  const [hasSubmitted, setHasSubmitted] = useState(false)
  const [userQuery, setUserQuery] = useState('')
  const [response, setResponse] = useState<ChatResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [expandedSources, setExpandedSources] = useState<Set<string>>(new Set())
  const [messages, setMessages] = useState<Message[]>([])

  function safeHost(u: string) {
    try {
      return new URL(u).hostname
    } catch {
      return u
    }
  }

  // Helpers to cleanly format source labels and canonical URLs
  function canonicalizeUrl(u: string): string {
    try {
      const url = new URL(u)
      return `${url.origin}${url.pathname}`
    } catch {
      return u
    }
  }

  function slugToTitleCase(slug: string): string {
    return slug
      .replace(/[-_]+/g, ' ')
      .replace(/\s+/g, ' ')
      .trim()
      .replace(/\b\w/g, (c) => c.toUpperCase())
  }

  function extractPublicationFromUrl(u: string): string | null {
    try {
      const path = new URL(u).pathname.toLowerCase()
      const m1 = path.match(/\/publications\/p(\d+)/i)
      if (m1) return `Publication ${m1[1]}`
      const m2 = path.match(/\/pub\/(?:irs-pdf\/)?p(\d+)/i)
      if (m2) return `Publication ${m2[1]}`
      return null
    } catch {
      return null
    }
  }

  function deriveTitleFromUrl(u: string): string {
    try {
      const url = new URL(u)
      const seg = url.pathname.split('/').filter(Boolean).pop() || ''
      const pub = extractPublicationFromUrl(u)
      if (pub) return pub
      if (!seg) return url.hostname
      return slugToTitleCase(decodeURIComponent(seg))
    } catch {
      return u
    }
  }

  function cleanSourceLabel(source: Source): string {
    // Hard overrides for known publications regardless of incoming title
    const path = (() => {
      try { return new URL(source.url).pathname.toLowerCase() } catch { return '' }
    })()
    if (/\/p505(\b|\/|#|\?|$)/.test(path) || /\/publications\/p505/.test(path)) {
      return 'Publication 505 — Tax Withholding and Estimated Tax'
    }
    if (/\/p575(\b|\/|#|\?|$)/.test(path) || /\/publications\/p575/.test(path)) {
      return 'Publication 575 — Pension and Annuity Income'
    }
    if (/\/p926(\b|\/|#|\?|$)/.test(path) || /\/publications\/p926/.test(path)) {
      return 'Publication 926 — Household Employer’s Tax Guide'
    }
    const pub = extractPublicationFromUrl(source.url)
    let t = (source.title || '').replace(/\s+/g, ' ').trim()
    // Remove noisy trailing excerpt markers or quotes artifacts
    t = t.replace(/\s+—\s*excerpt:.*$/i, '').replace(/\s+-\s*excerpt:.*$/i, '')
    // If metadata has a clean title, prefer it
    if (t) return t
    // Otherwise build from URL
    if (pub) return pub
    return deriveTitleFromUrl(source.url)
  }

  async function handleSubmit(submittedQuery: string) {
    if (!submittedQuery.trim()) return

    setHasSubmitted(true)
    setUserQuery(submittedQuery)
    setQuery('')
    setResponse(null)
    setError(null)
    setLoading(true)
    setExpandedSources(new Set())
    setMessages((prev) => [
      ...prev,
      { id: Date.now(), role: 'user', content: submittedQuery },
    ])

    try {
      const historyPayload = [...messages, { role: 'user', content: submittedQuery }].map((m) => ({
        role: m.role,
        content: m.content,
      }))
      const res = await fetch('/v1/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: submittedQuery, json: true, history: historyPayload }),
      })

      if (!res.ok) throw new Error(`HTTP ${res.status}`)

      const raw = await res.json()
      const normalized: ChatResponse = {
        answer_text:
          (raw?.answer_text ?? raw?.answer ?? raw?.output ?? raw?.text ?? '').toString(),
        sources: Array.isArray(raw?.sources) ? raw.sources : [],
        confidence: (raw?.confidence === 'low' || raw?.confidence === 'high') ? raw.confidence : 'medium',
        query_embedding_similarity: Array.isArray(raw?.query_embedding_similarity)
          ? raw.query_embedding_similarity
          : [],
        follow_up_questions: Array.isArray(raw?.follow_up_questions) ? raw.follow_up_questions : [],
      }

      if (!normalized.answer_text) {
        normalized.answer_text = 'No answer text returned.'
      }

      setResponse(normalized)
      setMessages((prev) => [
        ...prev,
        {
          id: Date.now() + 1,
          role: 'assistant',
          content: normalized.answer_text,
          sources: normalized.sources,
          follow_up_questions: normalized.follow_up_questions || [],
        },
      ])
    } catch (e: any) {
      setError(e.message || 'An error occurred')
      setMessages((prev) => [
        ...prev,
        {
          id: Date.now() + 2,
          role: 'assistant',
          content: `Error: ${e.message || 'An error occurred'}`,
        },
      ])
    } finally {
      setLoading(false)
    }
  }

  function handleChipClick(chipText: string) {
    setQuery(chipText)
    handleSubmit(chipText)
  }


  function toggleSource(key: string) {
    const next = new Set(expandedSources)
    if (next.has(key)) next.delete(key)
    else next.add(key)
    setExpandedSources(next)
  }

  return (
    <div className="min-h-screen bg-white flex flex-col">
      {!hasSubmitted ? (
        // Homepage - Before Query
        <div className="flex-1 flex flex-col items-center justify-center px-4 py-16 md:py-20 lg:py-24">
          {/* Main Title */}
          <h1 className="text-3xl md:text-4xl font-medium text-gray-900 mb-10 md:mb-12 text-center tracking-tight">
            What can I help with?
          </h1>

          {/* Input Bar */}
          <div className="w-full max-w-3xl mb-12 md:mb-16">
            <div className="relative bg-white rounded-full shadow-md hover:shadow-lg focus-within:shadow-lg focus-within:ring-2 focus-within:ring-blue-500/10 transition-all duration-300 border border-gray-100">
              <div className="flex items-center px-6 md:px-7 py-3.5 md:py-4">
                <input
                  type="text"
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' && !e.shiftKey) {
                      e.preventDefault()
                      handleSubmit(query)
                    }
                  }}
                  placeholder="Ask anything about IRS tax information"
                  className="flex-1 outline-none text-gray-900 placeholder-gray-400 text-base md:text-lg bg-transparent font-normal"
                />
                <button
                  onClick={() => handleSubmit(query)}
                  disabled={!query.trim()}
                  className="ml-3 text-white bg-gradient-to-r from-gray-800 to-gray-900 px-4 py-2 rounded-full font-semibold shadow-md hover:from-gray-900 hover:to-black disabled:opacity-40 disabled:cursor-not-allowed transition-all duration-200"
                  aria-label="Send"
                >
                  <svg
                    className="w-5 h-5"
                    viewBox="0 0 24 24"
                    fill="none"
                    xmlns="http://www.w3.org/2000/svg"
                  >
                    <path d="M5 12h12M13 5l7 7-7 7" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
                  </svg>
                </button>
              </div>
            </div>
          </div>

          {/* Suggestion Chips */}
          <div className="w-full max-w-5xl grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3.5 md:gap-4">
            {SUGGESTION_CHIPS.map((chip, index) => (
              <button
                key={index}
                onClick={() => handleChipClick(chip.text)}
                className={`group relative overflow-hidden bg-gradient-to-r ${chip.color} ${chip.hoverColor} text-white px-5 md:px-6 py-3 md:py-3.5 rounded-full font-medium text-sm md:text-sm flex items-center justify-center gap-2 shadow-md hover:shadow-lg active:scale-[0.98] transition-all duration-300 border border-white/20`}
              >
                <svg
                  className="w-4 h-4 md:w-4 md:h-4 flex-shrink-0"
                  fill="currentColor"
                  viewBox="0 0 20 20"
                >
                  <path d="M9.049 2.927c.3-.921 1.603-.921 1.902 0l1.07 3.292a1 1 0 00.95.69h3.462c.969 0 1.371 1.24.588 1.81l-2.8 2.034a1 1 0 00-.364 1.118l1.07 3.292c.3.921-.755 1.688-1.54 1.118l-2.8-2.034a1 1 0 00-1.175 0l-2.8 2.034c-.784.57-1.838-.197-1.539-1.118l1.07-3.292a1 1 0 00-.364-1.118L2.98 8.72c-.783-.57-.38-1.81.588-1.81h3.461a1 1 0 00.951-.69l1.07-3.292z" />
                </svg>
                <span className="text-center">{chip.text}</span>
              </button>
            ))}
          </div>
        </div>
      ) : (
        // Chat Interface - After Query
        <div className="flex-1 flex flex-col max-w-4xl w-full mx-auto px-4 md:px-6 py-8 md:py-10 bg-white">
          <div className="space-y-6 pb-24 md:pb-28">
            {messages.map((m, idx) => (
              m.role === 'user' ? (
                <div className="flex justify-end" key={m.id}>
                  <div className="bg-gradient-to-r from-gray-800 to-gray-900 text-white px-6 py-4 md:px-7 md:py-4.5 rounded-3xl rounded-tr-md max-w-[85%] md:max-w-[75%] shadow-lg border border-gray-700/50">
                    <p className="text-sm md:text-base leading-relaxed font-normal">{m.content}</p>
                  </div>
                </div>
              ) : (
                <div className="flex justify-start" key={m.id}>
                  <div className="bg-white border border-gray-200/80 px-6 py-5 md:px-7 md:py-6 rounded-3xl rounded-tl-md max-w-[85%] md:max-w-[75%] shadow-lg w-full">
                    <div className="space-y-5">
                      <div className="md-content max-w-none text-gray-800">
                        <ReactMarkdown remarkPlugins={[remarkGfm]}>
                          {m.content}
                        </ReactMarkdown>
                      </div>
                      {m.sources && m.sources.length > 0 && (
                        <div className="space-y-3">
                          <h4 className="text-sm font-semibold text-gray-800">Sources</h4>
                          <ul className="space-y-3">
                            {(() => {
                              const byUrl = new Map<string, Source>()
                              for (const s of m.sources) {
                                const link = canonicalizeUrl(s.url)
                                if (!byUrl.has(link)) byUrl.set(link, { ...s, url: link })
                              }
                              const unique = Array.from(byUrl.values())
                              return unique.map((source, index) => {
                                const label = cleanSourceLabel(source)
                                const link = source.url
                                return (
                                  <li key={`${m.id}-${index}`} className="text-sm">
                                    <div className="leading-snug">
                                      <a
                                        href={link}
                                        target="_blank"
                                        rel="noopener noreferrer"
                                        className="font-semibold text-blue-700 hover:text-blue-800 underline-offset-2 hover:underline"
                                        title={label}
                                      >
                                        {label}
                                      </a>
                                    </div>
                                    <div className="text-xs text-gray-600 mt-0.5 break-all">{link}</div>
                                  </li>
                                )
                              })
                            })()}
                          </ul>
                        </div>
                      )}
                      {m.follow_up_questions && m.follow_up_questions.length > 0 && (
                        <div className="pt-1">
                          <h4 className="text-sm font-semibold text-gray-800 mb-2">Related questions</h4>
                          <div className="rounded-xl border border-gray-200 bg-gray-50/50 p-3 md:p-4">
                            <div className="grid grid-cols-1 sm:grid-cols-2 gap-2.5">
                              {m.follow_up_questions.map((q, i) => (
                                <button
                                  key={i}
                                  onClick={() => handleSubmit(q)}
                                  title={q}
                                  className="group w-full px-3.5 py-2 text-sm rounded-xl border border-gray-200/80 bg-white hover:bg-gray-50 text-gray-700 hover:text-gray-900 shadow-sm hover:shadow transition-all duration-200 focus:outline-none focus:ring-2 focus:ring-blue-500/30 focus:border-gray-300 flex items-start gap-2 text-left"
                                >
                                  <span className="inline-flex items-center justify-center w-6 h-6 rounded-full bg-gray-50 text-gray-600 border border-gray-200 flex-shrink-0 mt-0.5">
                                    <svg className="w-3.5 h-3.5" viewBox="0 0 24 24" fill="none" stroke="currentColor">
                                      <circle cx="11" cy="11" r="7" strokeWidth="2" />
                                      <path d="M21 21l-4.35-4.35" strokeWidth="2" strokeLinecap="round" />
                                    </svg>
                                  </span>
                                  <span className="flex-1 whitespace-normal break-words leading-snug">
                                    {q}
                                  </span>
                                  <svg className="w-4 h-4 text-gray-400 group-hover:text-gray-600 mt-0.5 flex-shrink-0" viewBox="0 0 24 24" fill="none" stroke="currentColor">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 5l7 7-7 7" />
                                  </svg>
                                </button>
                              ))}
                            </div>
                          </div>
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              )
            ))}
            {loading && (
              <div className="flex justify-start">
                <div className="bg-white border border-gray-200/80 px-6 py-5 md:px-7 md:py-6 rounded-3xl rounded-tl-md max-w-[85%] md:max-w-[75%] shadow-lg">
                  <div className="flex items-center gap-3 text-gray-600">
                    <div className="flex gap-1.5">
                      <div className="w-2 h-2 bg-blue-500 rounded-full animate-bounce" style={{ animationDelay: '0ms' }}></div>
                      <div className="w-2 h-2 bg-blue-500 rounded-full animate-bounce" style={{ animationDelay: '150ms' }}></div>
                      <div className="w-2 h-2 bg-blue-500 rounded-full animate-bounce" style={{ animationDelay: '300ms' }}></div>
                    </div>
                    <span className="text-sm font-medium">IRS Assistant is thinking…</span>
                  </div>
                </div>
              </div>
            )}
            {error && (
              <div className="flex justify-start">
                <div className="text-red-600 bg-red-50 border border-red-200 rounded-lg p-4">
                  <p className="font-semibold mb-1.5">Error</p>
                  <p className="text-sm">{error}</p>
                </div>
              </div>
            )}
          </div>

          {/* Composer (fixed bottom) */}
          <div className="fixed left-0 right-0 bottom-4 z-40 px-4 md:px-6">
            <div className="relative bg-white rounded-2xl shadow-lg border border-gray-200 max-w-3xl mx-auto">
              <div className="flex items-center px-4 md:px-5 py-3.5">
                <input
                  type="text"
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' && !e.shiftKey) {
                      e.preventDefault()
                      if (query.trim()) handleSubmit(query.trim())
                    }
                  }}
                  placeholder="Ask another IRS question…"
                  className="flex-1 outline-none text-gray-900 placeholder-gray-400 bg-transparent text-base"
                />
                <button
                  onClick={() => query.trim() && handleSubmit(query.trim())}
                  disabled={!query.trim() || loading}
                  className="ml-3 text-white bg-gradient-to-r from-gray-800 to-gray-900 px-4 py-2 rounded-xl font-semibold shadow-lg hover:from-gray-900 hover:to-black disabled:opacity-40 disabled:cursor-not-allowed transition-all duration-200"
                  aria-label="Send"
                >
                  <svg
                    className="w-5 h-5"
                    viewBox="0 0 24 24"
                    fill="none"
                    xmlns="http://www.w3.org/2000/svg"
                  >
                    <path d="M5 12h12M13 5l7 7-7 7" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
                  </svg>
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
