// Run once per model/backend. The server is private to the benchmark.
// node scripts/benchmark-meeting-summary.mjs --server <llama-server> --model <gguf> --output <new.json>
// Add --device Vulkan0 --layers 99 to measure GPU inference.
import { spawn } from "node:child_process";
import { randomBytes } from "node:crypto";
import { once } from "node:events";
import { open, readFile, writeFile } from "node:fs/promises";
import { createServer } from "node:net";
import path from "node:path";
import { setTimeout as sleep } from "node:timers/promises";
import { parseArgs } from "node:util";
import contextPolicy from "../src/helpers/llamaContextPolicy.js";

const { values } = parseArgs({
  options: Object.fromEntries(
    ["server", "model", "output", "device", "layers"].map((key) => [key, { type: "string" }])
  ),
});
if (!values.server || !values.model || !values.output) {
  throw new Error(
    "Supply --server, --model and --output. For GPU also supply --device and --layers."
  );
}
const layers = Number(values.layers ?? (values.device ? 99 : 0));
if (!Number.isSafeInteger(layers) || layers < 0 || Boolean(values.device) !== layers > 0) {
  throw new Error("A GPU test needs a device and a positive layer count. CPU uses zero layers.");
}
const fixture = JSON.parse(
  await readFile(new URL("../test/fixtures/meeting-summary-pt-br.json", import.meta.url), "utf8")
);
const prompt = fixture.prompt + "\n\n" + fixture.transcript.join("\n");

const listener = createServer();
listener.listen(0, "127.0.0.1");
await once(listener, "listening");
const port = listener.address().port;
await new Promise((resolve, reject) =>
  listener.close((error) => (error ? reject(error) : resolve()))
);
const base = `http://127.0.0.1:${port}`;
const serverFile = path.resolve(values.server);
const modelFile = path.resolve(values.model);
const output = path.resolve(values.output);
const log = await open(output + ".log", "wx");
const apiKey = randomBytes(32).toString("hex");
const args = [
  "--model",
  modelFile,
  "--alias",
  "meeting-summary",
  "--host",
  "127.0.0.1",
  "--port",
  String(port),
  "--threads",
  "4",
  "--threads-batch",
  "4",
  "--parallel",
  "1",
  "--ctx-size",
  String(contextPolicy.BASELINE_CONTEXT_SIZE),
  "--jinja",
  "--log-verbosity",
  "4",
  "--cache-ram",
  "0",
  "--n-gpu-layers",
  String(layers),
];
if (values.device) args.push("--device", values.device);

const start = performance.now();
const child = spawn(serverFile, args, {
  cwd: path.dirname(serverFile),
  env: { ...process.env, LLAMA_API_KEY: apiKey },
  stdio: ["ignore", log.fd, log.fd],
  windowsHide: true,
});
let spawnError;
child.on("error", (error) => {
  spawnError = error;
});
const closed = new Promise((resolve) => child.once("close", resolve));
const stop = () => child.kill();
process.once("SIGINT", stop);
process.once("SIGTERM", stop);
const ms = (since) => Math.round(performance.now() - since);

async function generate(text, maxTokens) {
  const request = {
    model: "meeting-summary",
    messages: [{ role: "user", content: text }],
    temperature: 0,
    seed: 42,
    max_tokens: maxTokens,
    cache_prompt: false,
    chat_template_kwargs: { enable_thinking: false },
  };
  const begin = performance.now();
  const response = await fetch(base + "/v1/chat/completions", {
    method: "POST",
    headers: { "Content-Type": "application/json", Authorization: `Bearer ${apiKey}` },
    body: JSON.stringify(request),
    signal: AbortSignal.timeout(300000),
  });
  if (!response.ok) throw new Error(`Completion HTTP ${response.status}: ${await response.text()}`);
  const body = await response.json();
  const wallMs = ms(begin);
  const choice = body.choices[0];
  if (!choice.message.content?.trim() || choice.finish_reason !== "stop") {
    throw new Error(`Incomplete answer (${choice.finish_reason})`);
  }
  return { wallMs, usage: body.usage, timings: body.timings, text: choice.message.content.trim() };
}

try {
  while (true) {
    if (spawnError) throw spawnError;
    if (child.exitCode !== null || child.signalCode)
      throw new Error("Server exited before it was ready: " + output + ".log");
    const healthy = await fetch(base + "/health", {
      headers: { Authorization: `Bearer ${apiKey}` },
      signal: AbortSignal.timeout(1500),
    })
      .then((r) => r.ok)
      .catch(() => false);
    if (healthy) break;
    if (ms(start) > 180000) throw new Error("Server startup exceeded three minutes");
    await sleep(250);
  }
  const startupMs = ms(start);
  console.log(`Ready in ${startupMs} ms: ${path.basename(modelFile)} on ${values.device ?? "CPU"}`);
  await generate("Escreva apenas a palavra pronto.", 32);
  const runs = [];
  for (let index = 0; index < 3; index++) {
    const result = await generate(prompt, 512);
    runs.push(result);
    console.log(
      `Run ${index + 1}: ${result.wallMs} ms, ${result.usage.completion_tokens} output tokens`
    );
  }
  await writeFile(
    output,
    JSON.stringify(
      {
        model: path.basename(modelFile),
        device: values.device ?? "CPU",
        layers,
        contextSize: contextPolicy.BASELINE_CONTEXT_SIZE,
        startupMs,
        runs,
      },
      null,
      2
    ) + "\n",
    { flag: "wx" }
  );
} finally {
  if (child.exitCode === null && !child.signalCode) child.kill();
  await closed;
  process.off("SIGINT", stop);
  process.off("SIGTERM", stop);
  await log.close();
}
