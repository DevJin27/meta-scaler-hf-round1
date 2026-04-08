import assert from "node:assert/strict";
import test from "node:test";

import RedisMock from "ioredis-mock";

import { createLogger } from "../app/core/logger.js";
import { DistributedLockService } from "../app/services/distributed-lock.service.js";
import { GameEngineService } from "../app/services/game-engine.service.js";
import { GameRepositoryService } from "../app/services/game-repository.service.js";

function createTestContext() {
  const redis = new RedisMock();
  let now = 1_700_000_000_000;
  const config = {
    env: "test",
    port: 0,
    host: "0.0.0.0",
    redisUrl: "redis://test",
    adminSecret: "test-secret",
    allowedOrigins: ["*"],
    socketPath: "/socket.io",
    logLevel: "silent",
    roundDurationMs: 10_000,
    interRoundDelayMs: 0,
    autoAdvanceRounds: false,
    redisKeyPrefix: `signal-game-test-${Math.random().toString(36).slice(2)}`,
    pingIntervalMs: 25_000,
    pingTimeoutMs: 20_000,
    lifecycleTickMs: 250,
    lockTtlMs: 4_000,
    maxTeams: 10,
  };
  const logger = createLogger(config);
  const repository = new GameRepositoryService({ redis, config });
  const lockService = new DistributedLockService({ redis, config, logger });
  const engine = new GameEngineService({
    repository,
    lockService,
    config,
    logger,
    now: () => now,
  });

  engine.setPresenceResolver(async () => 0);

  return {
    redis,
    repository,
    engine,
    setNow(value) {
      now = value;
    },
    now() {
      return now;
    },
  };
}

test("duplicate submissions are rejected for the same team and round", async (t) => {
  const context = createTestContext();
  t.after(async () => {
    await context.redis.quit();
  });

  await context.repository.init();
  await context.engine.joinTeam({ teamId: "team-alpha", name: "Team Alpha" });
  await context.engine.startGame();

  const first = await context.engine.submitDecision({
    teamId: "team-alpha",
    decision: "TRADE",
  });

  assert.equal(first.accepted, true);

  await assert.rejects(
    context.engine.submitDecision({
      teamId: "team-alpha",
      decision: "IGNORE",
    }),
    /already submitted/i,
  );
});

test("alpha rounds reward trade and penalize ignore or missing decisions", async (t) => {
  const context = createTestContext();
  t.after(async () => {
    await context.redis.quit();
  });

  await context.repository.init();
  await context.engine.joinTeam({ teamId: "team-alpha", name: "Team Alpha" });
  await context.engine.joinTeam({ teamId: "team-beta", name: "Team Beta" });
  await context.engine.joinTeam({ teamId: "team-gamma", name: "Team Gamma" });

  const started = await context.engine.startGame();
  assert.equal(started.round, 1);

  await context.engine.submitDecision({ teamId: "team-alpha", decision: "TRADE" });
  await context.engine.submitDecision({ teamId: "team-beta", decision: "IGNORE" });

  context.setNow(context.now() + 11_000);
  const evaluation = await context.engine.evaluateExpiredRound();

  assert.equal(evaluation.signal.correctDecision, "TRADE");

  const byTeam = Object.fromEntries(evaluation.results.map((result) => [result.teamId, result]));
  assert.equal(byTeam["team-alpha"].delta, 180);
  assert.equal(byTeam["team-alpha"].verdict, "CORRECT_ALPHA");
  assert.equal(byTeam["team-beta"].delta, -100);
  assert.equal(byTeam["team-beta"].verdict, "MISS_ALPHA");
  assert.equal(byTeam["team-gamma"].delta, -100);
  assert.equal(byTeam["team-gamma"].verdict, "MISS_ALPHA");
});

test("non-alpha rounds reward ignore and penalize bad trades", async (t) => {
  const context = createTestContext();
  t.after(async () => {
    await context.redis.quit();
  });

  await context.repository.init();
  await context.engine.joinTeam({ teamId: "team-alpha", name: "Team Alpha" });
  await context.engine.joinTeam({ teamId: "team-beta", name: "Team Beta" });
  await context.engine.startGame();

  context.setNow(context.now() + 11_000);
  await context.engine.evaluateExpiredRound();

  const events = await context.engine.advanceRoundNow();
  assert.equal(events.at(-1)?.type, "round:started");

  await context.engine.submitDecision({ teamId: "team-alpha", decision: "IGNORE", round: 2 });
  await context.engine.submitDecision({ teamId: "team-beta", decision: "TRADE", round: 2 });

  context.setNow(context.now() + 11_000);
  const evaluation = await context.engine.evaluateExpiredRound();
  const byTeam = Object.fromEntries(evaluation.results.map((result) => [result.teamId, result]));

  assert.equal(evaluation.signal.correctDecision, "IGNORE");
  assert.equal(byTeam["team-alpha"].delta, 100);
  assert.equal(byTeam["team-alpha"].verdict, "CORRECT_IGNORE");
  assert.equal(byTeam["team-beta"].delta, -91);
  assert.equal(byTeam["team-beta"].verdict, "WRONG");
});
