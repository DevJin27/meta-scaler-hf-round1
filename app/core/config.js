import { config as loadDotenv } from "dotenv";
import { z } from "zod";

loadDotenv();

const envSchema = z.object({
  NODE_ENV: z.enum(["development", "test", "production"]).default("development"),
  PORT: z.coerce.number().int().positive().default(10000),
  HOST: z.string().trim().min(1).default("0.0.0.0"),
  REDIS_URL: z.string().trim().min(1, "REDIS_URL is required."),
  ADMIN_SECRET: z.string().trim().min(1, "ADMIN_SECRET is required."),
  CLIENT_ORIGINS: z.string().trim().optional(),
  SOCKET_PATH: z.string().trim().min(1).default("/socket.io"),
  LOG_LEVEL: z.string().trim().min(1).default("info"),
  ROUND_DURATION_MS: z.coerce.number().int().min(1000).default(10000),
  INTER_ROUND_DELAY_MS: z.coerce.number().int().min(0).default(3000),
  AUTO_ADVANCE_ROUNDS: z
    .string()
    .trim()
    .optional()
    .transform((value) => value !== "false"),
  REDIS_KEY_PREFIX: z.string().trim().min(1).default("signal-game"),
  PING_INTERVAL_MS: z.coerce.number().int().min(1000).default(25000),
  PING_TIMEOUT_MS: z.coerce.number().int().min(1000).default(20000),
  LIFECYCLE_TICK_MS: z.coerce.number().int().min(100).default(500),
  LOCK_TTL_MS: z.coerce.number().int().min(1000).default(4000),
  MAX_TEAMS: z.coerce.number().int().min(1).max(10).default(10),
});

export function loadConfig(env = process.env) {
  const parsed = envSchema.safeParse(env);

  if (!parsed.success) {
    const message = parsed.error.issues
      .map((issue) => `${issue.path.join(".") || "env"}: ${issue.message}`)
      .join("; ");
    throw new Error(`Invalid environment configuration: ${message}`);
  }

  const values = parsed.data;
  const allowedOrigins = values.CLIENT_ORIGINS
    ? values.CLIENT_ORIGINS.split(",")
        .map((value) => value.trim())
        .filter(Boolean)
    : ["*"];

  return {
    env: values.NODE_ENV,
    port: values.PORT,
    host: values.HOST,
    redisUrl: values.REDIS_URL,
    adminSecret: values.ADMIN_SECRET,
    allowedOrigins,
    socketPath: values.SOCKET_PATH,
    logLevel: values.LOG_LEVEL,
    roundDurationMs: values.ROUND_DURATION_MS,
    interRoundDelayMs: values.INTER_ROUND_DELAY_MS,
    autoAdvanceRounds: values.AUTO_ADVANCE_ROUNDS,
    redisKeyPrefix: values.REDIS_KEY_PREFIX,
    pingIntervalMs: values.PING_INTERVAL_MS,
    pingTimeoutMs: values.PING_TIMEOUT_MS,
    lifecycleTickMs: values.LIFECYCLE_TICK_MS,
    lockTtlMs: values.LOCK_TTL_MS,
    maxTeams: values.MAX_TEAMS,
  };
}
