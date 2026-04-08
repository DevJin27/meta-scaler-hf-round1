import { randomUUID } from "node:crypto";

import { createError } from "../core/errors.js";

const RELEASE_LOCK_SCRIPT = `
if redis.call("GET", KEYS[1]) == ARGV[1] then
  return redis.call("DEL", KEYS[1])
end
return 0
`;

export class DistributedLockService {
  constructor({ redis, config, logger }) {
    this.redis = redis;
    this.config = config;
    this.logger = logger;
  }

  lockKey(name) {
    return `${this.config.redisKeyPrefix}:lock:${name}`;
  }

  async acquire(name, ttlMs = this.config.lockTtlMs) {
    const token = randomUUID();
    const key = this.lockKey(name);
    const result = await this.redis.set(key, token, "PX", ttlMs, "NX");

    if (result !== "OK") {
      return null;
    }

    return { key, token };
  }

  async release(handle) {
    if (!handle) {
      return;
    }

    await this.redis.eval(RELEASE_LOCK_SCRIPT, 1, handle.key, handle.token);
  }

  async withLock(name, task, options = {}) {
    const ttlMs = options.ttlMs ?? this.config.lockTtlMs;
    const attempts = options.attempts ?? 1;
    const retryDelayMs = options.retryDelayMs ?? 50;
    let handle = null;

    for (let attempt = 0; attempt < attempts; attempt += 1) {
      handle = await this.acquire(name, ttlMs);
      if (handle) {
        break;
      }

      if (attempt < attempts - 1) {
        await new Promise((resolve) => setTimeout(resolve, retryDelayMs));
      }
    }

    if (!handle) {
      if (options.returnNullOnBusy) {
        return null;
      }

      throw createError("RESOURCE_BUSY", {
        message: "The game state is busy. Please retry.",
        statusCode: 423,
      });
    }

    try {
      return await task();
    } finally {
      await this.release(handle);
    }
  }
}
