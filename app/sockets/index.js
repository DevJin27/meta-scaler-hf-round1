import { Server } from "socket.io";

import { createError } from "../core/errors.js";
import { normalizeDecision, normalizeTeamJoin } from "../models/contracts.js";
import { serializeError } from "../core/errors.js";

const success = (data) => ({ ok: true, data });
const failure = (error) => ({ ok: false, error: serializeError(error) });
const noop = () => {};

function isOriginAllowed(origin, allowedOrigins) {
  if (!origin || allowedOrigins.includes("*")) {
    return true;
  }

  return allowedOrigins.includes(origin);
}

export function createSocketServer({
  httpServer,
  config,
  logger,
  engine,
  emitDomainEvent,
  buildPublicSnapshot,
}) {
  const io = new Server(httpServer, {
    path: config.socketPath,
    transports: ["websocket", "polling"],
    pingInterval: config.pingIntervalMs,
    pingTimeout: config.pingTimeoutMs,
    serveClient: false,
    allowRequest: (request, callback) => {
      const allowed = isOriginAllowed(request.headers.origin, config.allowedOrigins);
      callback(null, allowed);
    },
    cors: {
      origin(origin, callback) {
        callback(null, isOriginAllowed(origin, config.allowedOrigins));
      },
      methods: ["GET", "POST"],
    },
  });

  async function emitViewerSnapshot(socket) {
    const snapshot = await engine.buildSnapshot({
      teamId: socket.data.teamId ?? null,
      isAdmin: Boolean(socket.data.isAdmin),
    });
    socket.emit("game:snapshot", snapshot);
  }

  async function broadcastPublicState() {
    const snapshot = await buildPublicSnapshot();
    io.emit("game:snapshot", snapshot);
    io.emit("game:state", snapshot);
  }

  async function runHandler(socket, ack, handler) {
    const respond = typeof ack === "function" ? ack : noop;

    try {
      const result = await handler();
      respond(success(result));
      return result;
    } catch (error) {
      logger.warn(
        {
          err: error,
          socketId: socket.id,
          teamId: socket.data.teamId ?? null,
        },
        "socket handler failed",
      );
      respond(failure(error));
      socket.emit("game:error", serializeError(error));
      return null;
    }
  }

  io.engine.on("connection_error", (error) => {
    logger.warn({ err: error }, "socket connection error");
  });

  io.on("connection", async (socket) => {
    socket.data = {
      ...socket.data,
      isAdmin: false,
      teamId: null,
    };

    logger.info(
      {
        socketId: socket.id,
        transport: socket.conn.transport.name,
      },
      "socket connected",
    );

    const handshakeTeamId = socket.handshake.auth?.teamId;
    const handshakeName = socket.handshake.auth?.name;
    const handshakeAdminSecret = socket.handshake.auth?.adminSecret;

    if (handshakeAdminSecret && String(handshakeAdminSecret) === config.adminSecret) {
      socket.data.isAdmin = true;
      socket.join("admins");
    }

    if (handshakeTeamId && handshakeName) {
      try {
        const payload = normalizeTeamJoin({
          teamId: handshakeTeamId,
          name: handshakeName,
        });
        const result = await engine.joinTeam(payload);
        socket.data.teamId = result.team.teamId;
        socket.join(`team:${result.team.teamId}`);
      } catch (error) {
        logger.warn({ err: error, socketId: socket.id }, "handshake team join failed");
      }
    }

    await emitViewerSnapshot(socket);
    await broadcastPublicState();

    socket.on("state:request", async (_payload, ack) => {
      await runHandler(socket, ack, async () => {
        const snapshot = await engine.buildSnapshot({
          teamId: socket.data.teamId ?? null,
          isAdmin: Boolean(socket.data.isAdmin),
        });
        await emitViewerSnapshot(socket);
        return snapshot;
      });
    });

    socket.on("admin:authenticate", async (payload = {}, ack) => {
      await runHandler(socket, ack, async () => {
        engine.assertAdminSecret(payload.secret);
        socket.data.isAdmin = true;
        socket.join("admins");
        const snapshot = await engine.buildSnapshot({ isAdmin: true });
        await emitViewerSnapshot(socket);
        return {
          authenticated: true,
          snapshot,
        };
      });
    });

    async function handleTeamJoin(payload = {}, ack) {
      await runHandler(socket, ack, async () => {
        const joinPayload = normalizeTeamJoin(payload);
        const result = await engine.joinTeam(joinPayload);

        if (socket.data.teamId && socket.data.teamId !== result.team.teamId) {
          socket.leave(`team:${socket.data.teamId}`);
        }

        socket.data.teamId = result.team.teamId;
        socket.join(`team:${result.team.teamId}`);

        await broadcastPublicState();
        await emitViewerSnapshot(socket);

        return result;
      });
    }

    socket.on("team:join", handleTeamJoin);

    socket.on("team:register", async (payload = {}, ack) => {
      return handleTeamJoin(payload, ack);
    });

    async function handleDecision(payload = {}, ack) {
      await runHandler(socket, ack, async () => {
        if (!socket.data.teamId) {
          throw createError("INVALID_TEAM", {
            message: "Join a team before submitting a decision.",
          });
        }

        const result = await engine.submitDecision({
          teamId: socket.data.teamId,
          ...normalizeDecision(payload),
        });

        socket.emit("round:submission-status", result);
        await emitDomainEvent({
          type: "round:submission-status",
          payload: result,
        });
        await emitViewerSnapshot(socket);

        return result;
      });
    }

    socket.on("team:decision", handleDecision);
    socket.on("team:submit", handleDecision);

    socket.on("admin:start-game", async (_payload, ack) => {
      await runHandler(socket, ack, async () => {
        if (!socket.data.isAdmin) {
          throw createError("AUTH_REQUIRED");
        }
        const result = await engine.startGame();
        await emitDomainEvent({ type: "round:started", payload: result });
        return result;
      });
    });

    socket.on("admin:reset-game", async (_payload, ack) => {
      await runHandler(socket, ack, async () => {
        if (!socket.data.isAdmin) {
          throw createError("AUTH_REQUIRED");
        }
        const snapshot = await engine.resetGame();
        await broadcastPublicState();
        return snapshot;
      });
    });

    socket.on("admin:next-round", async (_payload, ack) => {
      await runHandler(socket, ack, async () => {
        if (!socket.data.isAdmin) {
          throw createError("AUTH_REQUIRED");
        }
        const events = await engine.advanceRoundNow();
        for (const event of events) {
          await emitDomainEvent(event);
        }
        return { events };
      });
    });

    socket.on("admin:pause-round", async (_payload, ack) => {
      await runHandler(socket, ack, async () => {
        if (!socket.data.isAdmin) {
          throw createError("AUTH_REQUIRED");
        }
        const result = await engine.pauseRound();
        await emitDomainEvent({ type: "round:paused", payload: result });
        return result;
      });
    });

    socket.on("admin:resume-round", async (_payload, ack) => {
      await runHandler(socket, ack, async () => {
        if (!socket.data.isAdmin) {
          throw createError("AUTH_REQUIRED");
        }
        const result = await engine.resumeRound();
        await emitDomainEvent({ type: "round:resumed", payload: result });
        return result;
      });
    });

    socket.on("disconnect", async (reason) => {
      logger.info(
        {
          socketId: socket.id,
          reason,
          teamId: socket.data.teamId ?? null,
        },
        "socket disconnected",
      );
      await broadcastPublicState();
    });
  });

  return {
    io,
    broadcastPublicState,
    emitViewerSnapshot,
  };
}
