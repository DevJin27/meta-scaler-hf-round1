import { asyncRoute, attachErrorHandler, createBaseApp } from "../core/http.js";
import { createError } from "../core/errors.js";
import {
  normalizeAdminAuth,
  normalizeDecision,
  normalizeTeamJoin,
} from "../models/contracts.js";

function readAdminSecret(req) {
  return (
    req.header("x-admin-secret") ||
    req.body?.secret ||
    req.query?.secret ||
    ""
  );
}

function ok(res, data) {
  res.json({ ok: true, data });
}

export function createHttpApp({
  logger,
  config,
  engine,
  runtime,
  redisClients,
  broadcastPublicSnapshot,
  emitDomainEvent,
}) {
  const app = createBaseApp({ logger, allowedOrigins: config.allowedOrigins });

  app.get(
    ["/health", "/api/health"],
    asyncRoute(async (_req, res) => {
      const [state, redisStatus] = await Promise.all([
        engine.repository.getState(),
        redisClients.command.ping(),
      ]);

      ok(res, {
        status: runtime.shuttingDown ? "shutting-down" : "ok",
        redis: redisStatus,
        phase: state.phase,
        round: state.round,
        totalRounds: state.totalRounds,
        remainingMs: engine.getRemainingMs(state),
        uptimeMs: Date.now() - runtime.startedAt,
      });
    }),
  );

  app.get(
    ["/state", "/api/state"],
    asyncRoute(async (req, res) => {
      const snapshot = await engine.buildSnapshot({
        teamId: typeof req.query.teamId === "string" ? req.query.teamId : null,
      });

      ok(res, snapshot);
    }),
  );

  app.get(
    ["/leaderboard", "/api/leaderboard"],
    asyncRoute(async (_req, res) => {
      const snapshot = await engine.buildSnapshot();
      ok(res, snapshot.leaderboard);
    }),
  );

  app.post(
    ["/teams/join", "/api/teams/join", "/api/team/join"],
    asyncRoute(async (req, res) => {
      const payload = normalizeTeamJoin(req.body);
      const result = await engine.joinTeam(payload);
      await broadcastPublicSnapshot();
      ok(res, result);
    }),
  );

  app.post(
    ["/teams/:teamId/decision", "/api/teams/:teamId/decision", "/teams/:teamId/submit", "/api/teams/:teamId/submit"],
    asyncRoute(async (req, res) => {
      const payload = normalizeDecision(req.body);
      const result = await engine.submitDecision({
        teamId: req.params.teamId,
        ...payload,
      });

      await Promise.all([
        broadcastPublicSnapshot(),
        emitDomainEvent({
          type: "round:submission-status",
          payload: result,
        }),
      ]);

      ok(res, result);
    }),
  );

  app.post(
    ["/admin/auth", "/api/admin/auth"],
    asyncRoute(async (req, res) => {
      const payload = normalizeAdminAuth({ secret: readAdminSecret(req) });
      engine.assertAdminSecret(payload.secret);
      ok(res, {
        authenticated: true,
        snapshot: await engine.buildSnapshot({ isAdmin: true }),
      });
    }),
  );

  app.post(
    ["/admin/game/start", "/api/admin/game/start"],
    asyncRoute(async (req, res) => {
      const payload = normalizeAdminAuth({ secret: readAdminSecret(req) });
      engine.assertAdminSecret(payload.secret);
      const result = await engine.startGame();
      await emitDomainEvent({ type: "round:started", payload: result });
      ok(res, result);
    }),
  );

  app.post(
    ["/admin/game/reset", "/api/admin/game/reset"],
    asyncRoute(async (req, res) => {
      const payload = normalizeAdminAuth({ secret: readAdminSecret(req) });
      engine.assertAdminSecret(payload.secret);
      const snapshot = await engine.resetGame();
      await broadcastPublicSnapshot();
      ok(res, snapshot);
    }),
  );

  app.post(
    ["/admin/game/next-round", "/api/admin/game/next-round"],
    asyncRoute(async (req, res) => {
      const payload = normalizeAdminAuth({ secret: readAdminSecret(req) });
      engine.assertAdminSecret(payload.secret);
      const events = await engine.advanceRoundNow();
      for (const event of events) {
        await emitDomainEvent(event);
      }
      ok(res, { events });
    }),
  );

  app.post(
    ["/admin/game/pause", "/api/admin/game/pause"],
    asyncRoute(async (req, res) => {
      const payload = normalizeAdminAuth({ secret: readAdminSecret(req) });
      engine.assertAdminSecret(payload.secret);
      const result = await engine.pauseRound();
      await emitDomainEvent({ type: "round:paused", payload: result });
      ok(res, result);
    }),
  );

  app.post(
    ["/admin/game/resume", "/api/admin/game/resume"],
    asyncRoute(async (req, res) => {
      const payload = normalizeAdminAuth({ secret: readAdminSecret(req) });
      engine.assertAdminSecret(payload.secret);
      const result = await engine.resumeRound();
      await emitDomainEvent({ type: "round:resumed", payload: result });
      ok(res, result);
    }),
  );

  app.use((_req, _res, next) => {
    next(createError("INVALID_INPUT", { message: "Route not found.", statusCode: 404 }));
  });

  attachErrorHandler(app, logger);

  return app;
}
