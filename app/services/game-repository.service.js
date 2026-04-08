import { SIGNALS } from "../models/signals.js";

const SUBMIT_DECISION_SCRIPT = `
local phase = redis.call("HGET", KEYS[1], "phase")
if phase ~= "live" then
  return "ROUND_NOT_ACTIVE"
end

local inputLocked = redis.call("HGET", KEYS[1], "inputLocked")
if inputLocked == "1" then
  return "ROUND_CLOSED"
end

local stateRound = tonumber(redis.call("HGET", KEYS[1], "round") or "0")
if stateRound ~= tonumber(ARGV[2]) then
  return "ROUND_NOT_ACTIVE"
end

local roundEndsAt = tonumber(redis.call("HGET", KEYS[1], "roundEndsAt") or "0")
if tonumber(ARGV[1]) > roundEndsAt then
  return "ROUND_CLOSED"
end

if redis.call("HEXISTS", KEYS[2], ARGV[3]) == 1 then
  return "ALREADY_SUBMITTED"
end

redis.call("HSET", KEYS[2], ARGV[3], ARGV[4])
redis.call("PEXPIRE", KEYS[2], 604800000)
return "OK"
`;

function defaultState(roundDurationMs) {
  return {
    phase: "idle",
    round: 0,
    totalRounds: SIGNALS.length,
    currentSignalId: "",
    roundDurationMs,
    roundStartedAt: 0,
    roundEndsAt: 0,
    inputLocked: true,
    pauseRemainingMs: 0,
    autoAdvanceAt: 0,
    lastUpdatedAt: 0,
  };
}

function parseState(raw, roundDurationMs) {
  const initial = defaultState(roundDurationMs);

  return {
    phase: raw.phase || initial.phase,
    round: Number(raw.round || initial.round),
    totalRounds: Number(raw.totalRounds || initial.totalRounds),
    currentSignalId: raw.currentSignalId || initial.currentSignalId,
    roundDurationMs: Number(raw.roundDurationMs || initial.roundDurationMs),
    roundStartedAt: Number(raw.roundStartedAt || initial.roundStartedAt),
    roundEndsAt: Number(raw.roundEndsAt || initial.roundEndsAt),
    inputLocked: raw.inputLocked === "1",
    pauseRemainingMs: Number(raw.pauseRemainingMs || initial.pauseRemainingMs),
    autoAdvanceAt: Number(raw.autoAdvanceAt || initial.autoAdvanceAt),
    lastUpdatedAt: Number(raw.lastUpdatedAt || initial.lastUpdatedAt),
  };
}

function serializeState(state) {
  return {
    phase: state.phase,
    round: String(state.round),
    totalRounds: String(state.totalRounds),
    currentSignalId: state.currentSignalId || "",
    roundDurationMs: String(state.roundDurationMs),
    roundStartedAt: String(state.roundStartedAt || 0),
    roundEndsAt: String(state.roundEndsAt || 0),
    inputLocked: state.inputLocked ? "1" : "0",
    pauseRemainingMs: String(state.pauseRemainingMs || 0),
    autoAdvanceAt: String(state.autoAdvanceAt || 0),
    lastUpdatedAt: String(state.lastUpdatedAt || 0),
  };
}

function parseJsonMap(rawMap) {
  return Object.entries(rawMap).map(([key, value]) => {
    try {
      return [key, JSON.parse(value)];
    } catch {
      return [key, null];
    }
  });
}

export class GameRepositoryService {
  constructor({ redis, config }) {
    this.redis = redis;
    this.config = config;
    this.keys = {
      state: `${config.redisKeyPrefix}:state`,
      teams: `${config.redisKeyPrefix}:teams`,
      lastEvaluation: `${config.redisKeyPrefix}:last-evaluation`,
      roundSubmissions: (round) => `${config.redisKeyPrefix}:round:${round}:submissions`,
    };
  }

  async init() {
    const exists = await this.redis.exists(this.keys.state);
    if (!exists) {
      await this.redis.hset(this.keys.state, serializeState(defaultState(this.config.roundDurationMs)));
    }
  }

  async getState() {
    await this.init();
    const raw = await this.redis.hgetall(this.keys.state);
    return parseState(raw, this.config.roundDurationMs);
  }

  async saveState(state) {
    await this.redis.hset(this.keys.state, serializeState(state));
  }

  async patchState(partialState) {
    const current = await this.getState();
    const nextState = {
      ...current,
      ...partialState,
    };
    await this.saveState(nextState);
    return nextState;
  }

  async listTeams() {
    const raw = await this.redis.hgetall(this.keys.teams);
    return parseJsonMap(raw)
      .map(([, team]) => team)
      .filter(Boolean);
  }

  async getTeam(teamId) {
    const raw = await this.redis.hget(this.keys.teams, teamId);
    return raw ? JSON.parse(raw) : null;
  }

  async countTeams() {
    return this.redis.hlen(this.keys.teams);
  }

  async saveTeam(team) {
    await this.redis.hset(this.keys.teams, team.teamId, JSON.stringify(team));
  }

  async saveTeams(teams) {
    if (!teams.length) {
      return;
    }

    const pipeline = this.redis.pipeline();
    for (const team of teams) {
      pipeline.hset(this.keys.teams, team.teamId, JSON.stringify(team));
    }
    await pipeline.exec();
  }

  async getSubmissions(round) {
    const raw = await this.redis.hgetall(this.keys.roundSubmissions(round));
    return Object.fromEntries(
      parseJsonMap(raw).filter(([, value]) => value !== null),
    );
  }

  async clearRoundSubmissions(round) {
    await this.redis.del(this.keys.roundSubmissions(round));
  }

  async clearAllRoundSubmissions() {
    const keys = await this.redis.keys(`${this.config.redisKeyPrefix}:round:*:submissions`);
    if (keys.length) {
      await this.redis.del(...keys);
    }
  }

  async getLastEvaluation() {
    const raw = await this.redis.get(this.keys.lastEvaluation);
    return raw ? JSON.parse(raw) : null;
  }

  async saveLastEvaluation(payload) {
    await this.redis.set(this.keys.lastEvaluation, JSON.stringify(payload));
  }

  async clearLastEvaluation() {
    await this.redis.del(this.keys.lastEvaluation);
  }

  async submitDecision({ teamId, round, decision, submittedAt }) {
    const response = await this.redis.eval(
      SUBMIT_DECISION_SCRIPT,
      2,
      this.keys.state,
      this.keys.roundSubmissions(round),
      String(submittedAt),
      String(round),
      teamId,
      JSON.stringify({
        teamId,
        decision,
        round,
        submittedAt,
      }),
    );

    return response;
  }
}
