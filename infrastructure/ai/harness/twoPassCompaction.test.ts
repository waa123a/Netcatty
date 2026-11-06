import assert from 'node:assert/strict';
import test from 'node:test';
import type { ModelMessage } from 'ai';
import { TwoPassCompactionCache, fingerprintMessages } from './twoPassCompaction';

test('fingerprintMessages includes complete tool arguments canonically', () => {
  const a: ModelMessage[] = [{
    role: 'assistant',
    content: [{ type: 'tool-call', toolCallId: '1', toolName: 'terminal_execute', input: { b: 2, a: 1 } }],
  }];
  const reordered: ModelMessage[] = [{
    role: 'assistant',
    content: [{ type: 'tool-call', toolCallId: '1', toolName: 'terminal_execute', input: { a: 1, b: 2 } }],
  }];
  const changed: ModelMessage[] = [{
    role: 'assistant',
    content: [{ type: 'tool-call', toolCallId: '1', toolName: 'terminal_execute', input: { a: 9, b: 2 } }],
  }];

  assert.equal(fingerprintMessages(a), fingerprintMessages(reordered));
  assert.notEqual(fingerprintMessages(a), fingerprintMessages(changed));
});

test('TwoPassCompactionCache reuses only an unchanged model and prefix', async () => {
  const cache = new TwoPassCompactionCache();
  const messages = Array.from({ length: 20 }, (_, index) => ({
    role: index % 2 ? 'assistant' : 'user',
    content: `message ${index}`,
  })) as ModelMessage[];
  cache.start('chat-1', 'model-a', messages, async () => 'NOTE1');

  const hit = await cache.consume('chat-1', 'model-a', messages);
  assert.equal(hit?.note, 'NOTE1');
  assert.ok((hit?.prefixLength ?? 0) < messages.length);
  assert.equal(await cache.consume('chat-1', 'model-b', messages), undefined);
});

test('TwoPassCompactionCache serves a settled note without the producer', async () => {
  const cache = new TwoPassCompactionCache();
  const messages = Array.from({ length: 20 }, (_, index) => ({
    role: index % 2 ? 'assistant' : 'user',
    content: `message ${index}`,
  })) as ModelMessage[];
  cache.start('chat-1', 'model-a', messages, async () => 'LATE');
  // Let the production promise settle before consuming.
  await new Promise((resolve) => setImmediate(resolve));

  const hit = await cache.consume('chat-1', 'model-a', messages);
  assert.equal(hit?.note, 'LATE');
});

test('TwoPassCompactionCache bounds its retained entries', async () => {
  const cache = new TwoPassCompactionCache();
  const messages = Array.from({ length: 20 }, (_, index) => ({
    role: index % 2 ? 'assistant' : 'user',
    content: `message ${index}`,
  })) as ModelMessage[];
  for (let index = 0; index < 40; index += 1) {
    cache.start(`chat-${index}`, 'model-a', messages, async () => `NOTE${index}`);
  }
  // The oldest entries were evicted once the cache exceeded its bound.
  assert.equal(await cache.consume('chat-0', 'model-a', messages), undefined);
  assert.equal((await cache.consume('chat-39', 'model-a', messages))?.note, 'NOTE39');
});
