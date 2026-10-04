/**
 * Live provider for a NInfer OpenAI-compatible inference server.
 *
 * Registers the `ninfer` provider. The model catalog is fetched from the
 * server's /v1/models during a network model refresh (like the built-in
 * llama.cpp integration). NInfer serves one resident model per server, so a
 * model only appears in the picker while the server reports it: when the
 * server is down or the model is unloaded, the catalog goes empty and the
 * model disappears. The last good catalog is persisted and restored offline,
 * so a previously loaded model stays visible across restarts while the
 * server is away.
 *
 * Run `/ninfer` in an interactive session to contact the server, check which
 * model is loaded, and refresh the catalog (same role as `/llama`).
 *
 * Models are flagged reasoning-capable with the qwen-chat-template thinking
 * format, so pi sends chat_template_kwargs.enable_thinking and the
 * thinking-level picker works.
 *
 * Server configuration (first match wins):
 *   1. NINFER_BASE_URL / NINFER_API_KEY environment variables
 *   2. ~/.pi/agent/ninfer.json  →  { "baseUrl": "...", "apiKey": "..." }
 *   3. /ninfer <url>            →  sets and persists the URL at runtime
 *   4. default: http://127.0.0.1:8080
 */
import { readFileSync } from "node:fs";
import { writeFile } from "node:fs/promises";
import { homedir } from "node:os";
import { join } from "node:path";
import { stream, streamSimple } from "@earendil-works/pi-ai/compat";
import { isModelType } from "@earendil-works/pi-ai";
import type {
	ApiKeyCredential,
	AuthContext,
	AuthResult,
	Model,
	Provider,
	ProviderStreamOptions,
	RefreshModelsContext,
} from "@earendil-works/pi-ai";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";

const NINFER_PROVIDER_ID = "ninfer";
const DEFAULT_SERVER_URL = "http://127.0.0.1:8080";
const FETCH_TIMEOUT_MS = 4000;
/** Max output tokens per request. /v1/models does not report it; adjust here. */
const MAX_OUTPUT_TOKENS = 32768;

/**
 * Thinking levels exposed in the picker. The Qwen chat template only switches
 * thinking on/off (any non-off level sends enable_thinking: true), so levels
 * are coarse; xhigh/max are kept out of the picker (null).
 */
const THINKING_LEVEL_MAP = {
	off: "off",
	minimal: "minimal",
	low: "low",
	medium: "medium",
	high: "high",
	xhigh: null,
	max: null,
} as const;

function agentDir(): string {
	return process.env.PI_AGENT_DIR ?? join(homedir(), ".pi", "agent");
}

function configPath(): string {
	return join(agentDir(), "ninfer.json");
}

interface NinferConfig {
	baseUrl?: string;
	apiKey?: string;
}

function readConfigFile(): NinferConfig {
	try {
		// Sync read at extension load only; the file is tiny.
		const raw = readFileSync(configPath(), "utf8");
		if (!raw) return {};
		const parsed = JSON.parse(raw) as NinferConfig;
		return parsed && typeof parsed === "object" ? parsed : {};
	} catch {
		return {};
	}
}

function normalizeServerUrl(raw: string | undefined): string | undefined {
	const value = raw?.trim();
	if (!value) return undefined;
	return value.replace(/\/+$/, "");
}

// --- Mutable server configuration (env > config file > default) ---
let serverUrl: string = DEFAULT_SERVER_URL;
let apiKey: string = "none";

function loadConfig(): void {
	const file = readConfigFile();
	serverUrl =
		normalizeServerUrl(process.env.NINFER_BASE_URL) ??
		normalizeServerUrl(file.baseUrl) ??
		DEFAULT_SERVER_URL;
	apiKey = process.env.NINFER_API_KEY ?? file.apiKey ?? "none";
}

async function persistConfig(baseUrl: string): Promise<void> {
	const existing = readConfigFile();
	const next: NinferConfig = { ...existing, baseUrl };
	await writeFile(configPath(), `${JSON.stringify(next, null, 2)}\n`, "utf8");
}

interface NinferModelEntry {
	id?: string;
	/** Configured context ceiling; present for the loaded model. */
	max_model_len?: number;
}

function isLoadedModel(entry: NinferModelEntry): entry is { id: string; max_model_len?: number } {
	return typeof entry?.id === "string" && entry.id.length > 0;
}

function toModel(entry: NinferModelEntry, baseUrl: string): Model<"openai-completions"> {
	const contextWindow =
		typeof entry.max_model_len === "number" && entry.max_model_len > 0 ? entry.max_model_len : MAX_OUTPUT_TOKENS;
	return {
		id: entry.id,
		name: entry.id,
		api: "openai-completions",
		provider: NINFER_PROVIDER_ID,
		baseUrl,
		reasoning: true,
		thinkingLevelMap: { ...THINKING_LEVEL_MAP },
		input: ["text"],
		cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
		contextWindow,
		maxTokens: MAX_OUTPUT_TOKENS,
		compat: {
			supportsStore: false,
			supportsDeveloperRole: false,
			supportsReasoningEffort: false,
			supportsUsageInStreaming: true,
			supportsStrictMode: false,
			maxTokensField: "max_tokens",
			thinkingFormat: "qwen-chat-template",
		},
	};
}

async function fetchEntries(
	serverUrl: string,
	apiKey: string,
	signal: AbortSignal,
): Promise<NinferModelEntry[]> {
	const res = await fetch(`${serverUrl}/v1/models`, {
		signal: AbortSignal.any([signal, AbortSignal.timeout(FETCH_TIMEOUT_MS)]),
		headers: { Authorization: `Bearer ${apiKey}` },
	});
	if (!res.ok) throw new Error(`HTTP ${res.status}`);
	const body = (await res.json()) as { data?: NinferModelEntry[] };
	return (body.data ?? []).filter(isLoadedModel);
}

export default function (pi: ExtensionAPI) {
	loadConfig();
	let models: readonly Model<"openai-completions">[] = [];

	const provider: Provider<"openai-completions"> = {
		id: NINFER_PROVIDER_ID,
		name: "ninfer",
		// Always reflects the current server (env / config file / /ninfer).
		get baseUrl() {
			return `${serverUrl}/v1`;
		},
		auth: {
			apiKey: {
				name: "NInfer server",
				check: async () => ({
					type: "api_key" as const,
					source: `server ${serverUrl}`,
				}),
				resolve: async (): Promise<AuthResult> => ({
					auth: { apiKey, baseUrl: `${serverUrl}/v1` },
					env: { NINFER_BASE_URL: serverUrl },
					source: "ninfer config",
				}),
			},
		},
		getModels: () => models,
		refreshModels: async (context: RefreshModelsContext): Promise<void> => {
			// Offline phase: restore the persisted catalog so a previously
			// loaded model stays visible while the server is unreachable
			// (same behavior as the built-in llama.cpp integration).
			if (!context.allowNetwork) {
				const stored = context.stored?.models.filter(
					(model) =>
						model.provider === NINFER_PROVIDER_ID &&
						isModelType(model, "chat") &&
						model.api === "openai-completions",
				) as Model<"openai-completions">[] | undefined;
				if (stored && stored.length > 0) {
					await context.publish({
						update: () => {
							models = stored;
						},
					});
				}
				return;
			}

			// Network phase: fetch the live catalog from the configured server.
			let refreshed: Model<"openai-completions">[];
			try {
				const entries = await fetchEntries(serverUrl, apiKey, context.signal);
				refreshed = entries.map((entry) => toModel(entry, `${serverUrl}/v1`));
			} catch {
				// Server unreachable or model unloaded: keep the restored
				// catalog (or nothing, if there is no previous one). The next
				// refresh retries.
				return;
			}
			if (context.signal.aborted) return;
			await context.publish({
				persist: { models: refreshed, checkedAt: Date.now() },
				update: () => {
					models = refreshed;
				},
			});
		},
		stream: (model, context, options) => stream(model, context, options as ProviderStreamOptions | undefined),
		streamSimple: (model, context, options) => streamSimple(model, context, options),
	};

	pi.registerProvider(provider);

	/**
	 * Contact the NInfer server, check which model is loaded, and refresh the
	 * catalog. Mirrors the built-in `/llama` command's explicit network
	 * refresh (startup refreshes are cache-only by design).
	 *
	 *   /ninfer                 → refresh with the configured server
	 *   /ninfer http://host:8080→ set + persist the server URL, then refresh
	 */
	pi.registerCommand("ninfer", {
		description: "Check the NInfer server and refresh the model catalog (/ninfer <url> to set the server)",
		handler: async (args, ctx) => {
			if (ctx.mode !== "tui") {
				ctx.ui.notify("/ninfer is available in interactive mode", "warning");
				return;
			}

			const arg = args.trim();
			if (arg) {
				const url = normalizeServerUrl(arg);
				if (!url || !/^https?:\/\//i.test(url)) {
					ctx.ui.notify(`Invalid NInfer URL: ${arg}`, "error");
					return;
				}
				serverUrl = url;
				try {
					await persistConfig(url);
					ctx.ui.notify(`NInfer server set to ${url} (saved to ${configPath()})`);
				} catch (error) {
					ctx.ui.notify(`NInfer server set to ${url} for this session (could not save)`, "warning");
					void error;
				}
			}

			try {
				const entries = await fetchEntries(serverUrl, apiKey, AbortSignal.timeout(10_000));
				if (entries.length === 0) {
					ctx.ui.notify(`No model loaded on ${serverUrl}`, "warning");
				}
				const result = await ctx.modelRegistry.refresh({
					providers: [NINFER_PROVIDER_ID],
					// /ninfer already contacted the configured server, so keep this
					// refresh live even in PI_OFFLINE.
					allowNetwork: true,
					signal: AbortSignal.timeout(15_000),
				});
				if (result.aborted) throw new Error("Model catalog refresh timed out.");
				const refreshError = result.errors.get(NINFER_PROVIDER_ID);
				if (refreshError) throw refreshError;
				for (const entry of entries) {
					ctx.ui.notify(
						`Loaded ${entry.id}${typeof entry.max_model_len === "number" ? ` (context ${entry.max_model_len})` : ""}`,
					);
				}
			} catch (error) {
				const message =
					error instanceof Error && /fetch failed|timeout|network/i.test(`${error.name} ${error.message}`)
						? `Could not connect to ${serverUrl}.`
						: error instanceof Error
							? error.message
							: String(error);
				ctx.ui.notify(message, "error");
			}
		},
	});
}
