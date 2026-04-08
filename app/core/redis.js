import { Redis } from "ioredis";

export async function createRedisClients(config, logger) {
  const baseOptions = {
    lazyConnect: true,
    maxRetriesPerRequest: null,
    enableReadyCheck: true,
    retryStrategy(attempts) {
      return Math.min(attempts * 200, 3000);
    },
  };

  const command = new Redis(config.redisUrl, baseOptions);
  const pub = new Redis(config.redisUrl, baseOptions);
  const sub = new Redis(config.redisUrl, baseOptions);

  for (const client of [command, pub, sub]) {
    client.on("error", (error) => {
      logger.error({ err: error }, "redis client error");
    });
  }

  await Promise.all([command.connect(), pub.connect(), sub.connect()]);

  return { command, pub, sub };
}

export async function closeRedisClients(redisClients) {
  await Promise.allSettled([
    redisClients.command.quit(),
    redisClients.pub.quit(),
    redisClients.sub.quit(),
  ]);
}
