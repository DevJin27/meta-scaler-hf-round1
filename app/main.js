import http from "node:http";

import { createAdapter } from "@socket.io/redis-adapter";

import { createHttpApp } from "./api/routes.js";
import { loadConfig } from "./core/config.js";
import { createLogger } from "./core/logger.js";
import { closeRedisClients, createRedisClients } from "./core/redis.js";
import { createSocketServer } from "./sockets/index.js";
import { DistributedLockService } from "./services/distributed-lock.service.js";
import { GameEngineService } from "./services/game-engine.service.js";
import { GameRepositoryService } from "./services/game-repository.service.js";
import { LifecycleService } from "./services/lifecycle.service.js";

export async function createApplication(options = {}) {
  const config = options.config ?? loadConfig(options.env);
  const logger = options.logger ?? createLogger(config);
  const redisClients = options.redisClients ?? (await createRedisClients(config, logger));
  const runtime = {
    startedAt: Date.now(),
    shuttingDown: false,
  };

  const repository = new GameRepositoryService({
    redis: redisClients.command,
    config,
  });
  await repository.init();

  const lockService = new DistributedLockService({
    redis: redisClients.command,
    config,
    logger,
  });

  const engine = new GameEngineService({
    repository,
    lockService,
    config,
    logger,
    now: options.now,
  });

  const buildPublicSnapshot = () => engine.buildSnapshot();

  let socketServerRef = null;

  const emitDomainEvent = async (event) => {
    if (!socketServerRef) {
      return;
    }

    const { io, broadcastPublicState } = socketServerRef;

    switch (event.type) {
      case "round:evaluated":
        io.emit("round:locked", {
          round: event.payload.round,
          lockedAt: event.payload.evaluatedAt,
        });
        io.emit("round:evaluated", event.payload);
        io.emit("round:results", event.payload);
        io.emit("leaderboard:updated", event.payload.leaderboard);
        break;
      case "round:started":
        io.emit("round:started", event.payload);
        break;
      case "round:paused":
        io.emit("round:paused", event.payload);
        break;
      case "round:resumed":
        io.emit("round:resumed", event.payload);
        break;
      case "round:submission-status":
        io.emit("round:submission-status", {
          accepted: event.payload.accepted,
          teamId: event.payload.teamId,
          round: event.payload.round,
          submittedAt: event.payload.submittedAt,
        });
        break;
      case "game:finished":
        io.emit("game:finished", event.payload);
        io.emit("leaderboard:updated", event.payload.leaderboard);
        break;
      default:
        break;
    }

    await broadcastPublicState();
  };

  const app = createHttpApp({
    logger,
    config,
    engine,
    runtime,
    redisClients,
    broadcastPublicSnapshot: async () => {
      if (socketServerRef) {
        await socketServerRef.broadcastPublicState();
      }
    },
    emitDomainEvent,
  });

  const httpServer = http.createServer(app);

  const socketServer = createSocketServer({
    httpServer,
    config,
    logger,
    engine,
    emitDomainEvent,
    buildPublicSnapshot,
  });
  socketServerRef = socketServer;

  socketServer.io.adapter(
    createAdapter(redisClients.pub, redisClients.sub, {
      publishOnSpecificResponseChannel: true,
    }),
  );

  engine.setPresenceResolver(async (teamId) => {
    const sockets = await socketServer.io.in(`team:${teamId}`).allSockets();
    return sockets.size;
  });

  const lifecycle = new LifecycleService({
    engine,
    config,
    logger,
    onEvent: emitDomainEvent,
  });
  let listening = false;

  async function start() {
    lifecycle.start();

    await new Promise((resolve) => {
      httpServer.listen(config.port, config.host, resolve);
    });
    listening = true;

    logger.info(
      {
        host: config.host,
        port: config.port,
        socketPath: config.socketPath,
      },
      "server listening",
    );
  }

  async function close(reason = "shutdown") {
    if (runtime.shuttingDown) {
      return;
    }

    runtime.shuttingDown = true;
    lifecycle.stop();

    socketServer.io.emit("game:error", {
      code: "SERVER_SHUTTING_DOWN",
      message: "The server is shutting down.",
      details: { reason },
    });

    await new Promise((resolve) => setTimeout(resolve, 25));
    await new Promise((resolve) => socketServer.io.close(resolve));
    if (listening) {
      await new Promise((resolve, reject) => {
        httpServer.close((error) => {
          if (error) {
            reject(error);
            return;
          }
          resolve();
        });
      });
      listening = false;
    }

    if (!options.redisClients) {
      await closeRedisClients(redisClients);
    }
  }

  for (const signal of ["SIGINT", "SIGTERM"]) {
    process.on(signal, async () => {
      logger.info({ signal }, "received shutdown signal");
      try {
        await close(signal);
        process.exit(0);
      } catch (error) {
        logger.error({ err: error, signal }, "graceful shutdown failed");
        process.exit(1);
      }
    });
  }

  return {
    app,
    httpServer,
    io: socketServer.io,
    engine,
    lifecycle,
    config,
    logger,
    start,
    close,
  };
}
