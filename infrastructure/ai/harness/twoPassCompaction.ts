import type { ModelMessage } from 'ai';
import { findSafeCompactionSplitIndex } from '../contextCompaction';

/**
 * Notes are tiny relative to the message prefix they summarize, but only when
 * the producer closure is released. `pending` holds the in-flight production
 * promise (whose closure retains the full prefix) and is nulled as soon as it
 * settles; `note` keeps the settled string for reuse. The cache is additionally
 * bounded so a long-lived app cannot accumulate entries from many sessions.
 */
interface CacheEntry {
  modelId: string;
  prefixLength: number;
  fingerprint: string;
  note: string | null;
  pending: Promise<string> | null;
}

const MAX_ENTRIES = 32;

function canonicalize(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(canonicalize);
  if (!value || typeof value !== 'object') return value;
  return Object.fromEntries(
    Object.entries(value as Record<string, unknown>)
      .sort(([left], [right]) => left.localeCompare(right))
      .map(([key, entry]) => [key, canonicalize(entry)]),
  );
}

export function fingerprintMessages(messages: ModelMessage[]): string {
  const value = JSON.stringify(canonicalize(messages));
  let hash = 0x811c9dc5;
  for (let index = 0; index < value.length; index += 1) {
    hash ^= value.charCodeAt(index);
    hash = Math.imul(hash, 0x01000193);
  }
  return (hash >>> 0).toString(36);
}

export class TwoPassCompactionCache {
  private readonly entries = new Map<string, CacheEntry>();

  start(
    chatSessionId: string,
    modelId: string,
    messages: ModelMessage[],
    producer: (prefix: ModelMessage[]) => Promise<string>,
  ): boolean {
    const current = this.entries.get(chatSessionId);
    if (
      current
      && current.modelId === modelId
      && messages.length >= current.prefixLength
      && fingerprintMessages(messages.slice(0, current.prefixLength)) === current.fingerprint
    ) {
      return false;
    }
    const protectedTail = Math.max(10, Math.ceil(messages.length * 0.05));
    const prefixLength = findSafeCompactionSplitIndex(messages, protectedTail);
    if (prefixLength <= 0) return false;
    const prefix = messages.slice(0, prefixLength);
    const fingerprint = fingerprintMessages(prefix);
    if (
      current
      && current.modelId === modelId
      && current.prefixLength === prefixLength
      && current.fingerprint === fingerprint
    ) {
      return false;
    }
    const entry: CacheEntry = {
      modelId,
      prefixLength,
      fingerprint,
      note: null,
      pending: producer(prefix).then(note => note.slice(0, 12_000)),
    };
    entry.pending = entry.pending.then(
      (note) => {
        // Drop the producer closure (which retains the full prefix array) the
        // moment the note settles; only the summarized string is kept.
        if (this.entries.get(chatSessionId) === entry) {
          entry.note = note;
          entry.pending = null;
        }
        return note;
      },
      () => {
        if (this.entries.get(chatSessionId) === entry) {
          entry.note = '';
          entry.pending = null;
        }
        return '';
      },
    );
    this.entries.set(chatSessionId, entry);
    // Bounded cache: evict the oldest insertion so failed/abandoned sessions
    // cannot pin their prefixes here for the app's lifetime.
    while (this.entries.size > MAX_ENTRIES) {
      const oldestKey = this.entries.keys().next().value;
      if (oldestKey === undefined) break;
      this.entries.delete(oldestKey);
    }
    return true;
  }

  async consume(
    chatSessionId: string,
    modelId: string,
    messages: ModelMessage[],
  ): Promise<{ note: string; prefixLength: number } | undefined> {
    const entry = this.entries.get(chatSessionId);
    if (!entry || entry.modelId !== modelId || messages.length < entry.prefixLength) {
      this.entries.delete(chatSessionId);
      return undefined;
    }
    const prefix = messages.slice(0, entry.prefixLength);
    if (fingerprintMessages(prefix) !== entry.fingerprint) {
      this.entries.delete(chatSessionId);
      return undefined;
    }
    const note = entry.pending ? await entry.pending : entry.note;
    return note ? { note, prefixLength: entry.prefixLength } : undefined;
  }

  clear(chatSessionId: string): void {
    this.entries.delete(chatSessionId);
  }
}

export const globalTwoPassCompactionCache = new TwoPassCompactionCache();
